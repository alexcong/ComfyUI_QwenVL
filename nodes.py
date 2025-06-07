"""ComfyUI custom nodes for Qwen2.5-VL and Qwen2.5 models."""
import os
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np
import folder_paths
import subprocess
import uuid
import tempfile


def tensor_to_pil(image_tensor, batch_index=0) -> Image:
    """
    Converts an image tensor to a PIL Image object.

    Args:
        image_tensor (torch.Tensor): The input tensor, expected to be in the
                                     shape [batch, height, width, channels].
        batch_index (int): The index of the image in the batch to convert.

    Returns:
        PIL.Image.Image: The converted PIL Image.
    """
    # Convert tensor of shape [batch, height, width, channels] at the batch_index to PIL Image
    image_tensor = image_tensor[batch_index].unsqueeze(0)
    i = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


class Qwen2VL:
    """
    ComfyUI node for Qwen2.5-VL (Vision-Language) models.
    This node can process text, image, and optionally video inputs to generate text responses.
    It handles model loading, preprocessing of inputs, inference, and postprocessing.
    """
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types, names, and default values for the ComfyUI node.
        This method is crucial for ComfyUI to render the node's UI and manage its inputs.
        """
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "Qwen2.5-VL-3B-Instruct",
                        "Qwen2.5-VL-7B-Instruct",
                        "SkyCaptioner-V1",
                    ],
                    {"default": "Qwen2.5-VL-3B-Instruct"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                "seed": ("INT", {"default": -1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"

    def inference(
        self,
        text,
        model,
        quantization,
        keep_model_loaded,
        temperature,
        max_new_tokens,
        seed,
        image=None,
        video_path=None,
    ):
        """
        Performs inference using the Qwen2.5-VL model.

        Args:
            text (str): The primary text prompt for the model.
            model (str): The specific Qwen2.5-VL model checkpoint to use.
            quantization (str): The quantization method to apply ("none", "4bit", "8bit").
            keep_model_loaded (bool): If True, keeps the model loaded in memory after inference.
            temperature (float): Sampling temperature for generation.
            max_new_tokens (int): Maximum number of new tokens to generate.
            seed (int): Random seed for generation (-1 for random).
            image (torch.Tensor, optional): An image tensor to provide as visual context.
            video_path (str, optional): Path to a video file to provide as visual context.

        Returns:
            tuple: A tuple containing a single string, which is either the generated text
                   response from the model or an error message if inference fails.

        The method handles:
        1. Model and processor loading (if not already loaded).
        2. Preprocessing of text, image (if provided), and video (if provided).
           - Video is processed using ffmpeg to extract frames or represent the video.
        3. Combining inputs into a format suitable for the Qwen2.5-VL model.
        4. Running the model inference.
        5. Postprocessing the generated tokens into a readable string.
        6. Cleaning up temporary files (e.g., processed video).
        7. Optionally unloading the model from memory.
        """
        if seed != -1:
            torch.manual_seed(seed)

        if model.startswith("Qwen"):
            model_id = f"qwen/{model}"
        else:
            model_id = f"Skywork/{model}"
        # put downloaded model to model/LLM dir
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "LLM", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
            )

        if self.processor is None:
            # Define min_pixels and max_pixels:
            # Images will be resized to maintain their aspect ratio
            # within the range of min_pixels and max_pixels.
            min_pixels = 256*28*28
            max_pixels = 1024*28*28 

            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        if self.model is None:
            # Load the model on the available device(s)
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                quantization_config = None

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
            )

        with torch.no_grad():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                }
            ]

            processed_video_path = None  # Initialize
            try:
                if video_path:
                    print("deal video_path", video_path)
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video_file:
                        processed_video_path = tmp_video_file.name

                    ffmpeg_command = [
                        "ffmpeg",
                        "-i", video_path,
                        "-vf", "fps=1,scale='min(256,iw)':min'(256,ih)':force_original_aspect_ratio=decrease",
                        "-c:v", "libx264",
                        "-preset", "fast",
                        "-crf", "18",
                        processed_video_path
                    ]
                    subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True) # Added capture_output and text for better error handling if needed

                    messages[0]["content"].insert(0, {
                        "type": "video",
                        "video": processed_video_path,
                    })
                elif image is not None: # Ensure image is not None before processing
                    print("deal image")
                    pil_image = tensor_to_pil(image)
                    messages[0]["content"].insert(0, {
                        "type": "image",
                        "image": pil_image,
                    })
                # If neither video_path nor image is provided, messages[0]["content"] will only contain text.

                text_prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                print("deal messages", messages) # For debugging
                image_inputs, video_inputs_processed = process_vision_info(messages) # Renamed to avoid conflict

                model_inputs = self.processor(
                    text=[text_prompt], # Corrected variable name
                    images=image_inputs,
                    videos=video_inputs_processed, # Corrected variable name
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")

                generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                result = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                    # temperature is not a param for batch_decode, it's for generate
                )
                # The temperature parameter is typically used in the `generate` method, not `batch_decode`.
                # If it's intended for `generate`, it should be passed there.
                # For now, removing from `batch_decode` as it's not a valid arg.

                if not keep_model_loaded:
                    del self.processor
                    del self.model
                    self.processor = None
                    self.model = None
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                return result

            except subprocess.CalledProcessError as e:
                # Specific error handling for ffmpeg failure
                error_message = f"Error processing video with ffmpeg: {e}\nStderr: {e.stderr}"
                print(error_message)
                return (error_message,)
            except FileNotFoundError:
                # Specific error handling if ffmpeg is not found
                error_message = "Error: ffmpeg not found. Please ensure ffmpeg is installed and in your PATH."
                print(error_message)
                return (error_message,)
            except Exception as e:
                # General error handling for other exceptions during the try block
                error_message = f"Error during Qwen2VL inference: {str(e)}"
                print(error_message)
                return (error_message,)
            finally:
                if processed_video_path and os.path.exists(processed_video_path):
                    os.remove(processed_video_path)
                    print(f"Cleaned up temporary video file: {processed_video_path}")


class Qwen2:
    """
    ComfyUI node for Qwen2.5 text generation models.
    This node takes a system prompt and a user prompt to generate text responses using
    various Qwen2.5 language model checkpoints. It handles model loading, tokenization,
    inference, and decoding.
    """
    def __init__(self):
        self.model_checkpoint = None
        self.tokenizer = None
        self.model = None
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types, names, and default values for the ComfyUI node.
        This method allows ComfyUI to construct the node's interface and manage its data flow.
        """
        return {
            "required": {
                "system": (
                    "STRING",
                    {
                        "default": "You are a helpful assistant.",
                        "multiline": True,
                    },
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "Qwen2.5-3B-Instruct",
                        "Qwen2.5-7B-Instruct",
                        "Qwen2.5-14B-Instruct",
                        "Qwen2.5-32B-Instruct",
                    ],
                    {"default": "Qwen2.5-7B-Instruct"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),  # add quantization type selection
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                "seed": ("INT", {"default": -1}),  # add seed parameter, default is -1
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"

    def inference(
        self,
        system,
        prompt,
        model,
        quantization,
        keep_model_loaded,
        temperature,
        max_new_tokens,
        seed,
    ):
        """
        Performs text generation using the specified Qwen2.5 model.

        Args:
            system (str): The system prompt to guide the model's behavior.
            prompt (str): The user's prompt for which a response is generated.
            model (str): The specific Qwen2.5 model checkpoint to use.
            quantization (str): The quantization method ("none", "4bit", "8bit").
            keep_model_loaded (bool): Whether to keep the model in memory after inference.
            temperature (float): Sampling temperature for generation.
            max_new_tokens (int): Maximum number of new tokens to generate.
            seed (int): Random seed for generation (-1 for random).

        Returns:
            tuple: A tuple containing a single string, which is either the generated text
                   response or an error message if inference fails (e.g., empty prompt).

        The method handles:
        1. Checking for an empty prompt.
        2. Model and tokenizer loading (if not already loaded), with support for quantization.
        3. Applying the chat template to combine system and user prompts.
        4. Tokenizing the input text.
        5. Running the model inference to generate token IDs.
        6. Decoding the generated tokens back into a string.
        7. Optionally unloading the model and tokenizer from memory.
        """
        if not prompt.strip():
            return ("Error: Prompt input is empty.",)

        if seed != -1:
            torch.manual_seed(seed)
        model_id = f"qwen/{model}"
        # put downloaded model to model/LLM dir
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "LLM", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
            )

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        if self.model is None:
            # Load the model on the available device(s)
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                quantization_config = None

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
            )

        with torch.no_grad():
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer([text], return_tensors="pt").to("cuda")

            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = self.tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
                temperature=temperature,
            )

            if not keep_model_loaded:
                del self.tokenizer
                del self.model
                self.tokenizer = None
                self.model = None
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            return result
