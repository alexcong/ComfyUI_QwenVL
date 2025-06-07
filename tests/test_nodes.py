import unittest
from unittest.mock import patch, MagicMock

# Add path to import nodes from parent directory
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Attempt to import the node classes
# These imports might require mocks if they trigger heavy initializations,
# but INPUT_TYPES is a classmethod and __init__ might be simple enough or mockable.
from nodes import Qwen2VL, Qwen2, tensor_to_pil # tensor_to_pil might be needed if INPUT_TYPES uses it, or for other tests

class TestQwen2VLNode(unittest.TestCase):

    def test_input_types(self):
        inputs = Qwen2VL.INPUT_TYPES()
        self.assertIn("required", inputs)
        required_keys = ["text", "model", "quantization", "keep_model_loaded", "temperature", "max_new_tokens", "seed"]
        for key in required_keys:
            self.assertIn(key, inputs["required"])

        # Check a specific default value as an example
        self.assertEqual(inputs["required"]["model"][1]["default"], "Qwen2.5-VL-3B-Instruct")
        self.assertEqual(inputs["required"]["quantization"][0], ["none", "4bit", "8bit"])

    @patch('nodes.torch.cuda.is_available')
    @patch('nodes.torch.cuda.get_device_capability')
    def test_initialization(self, mock_get_device_capability, mock_is_available):
        # Mock CUDA checks during initialization
        mock_is_available.return_value = False # Simulate CPU environment or basic GPU
        mock_get_device_capability.return_value = (7, 5) # Simulate a GPU that doesn't support bf16

        node = Qwen2VL()
        self.assertIsNone(node.model_checkpoint)
        self.assertIsNone(node.processor)
        self.assertIsNone(node.model)
        self.assertFalse(node.bf16_support) # Based on mock_is_available=False

    @patch('nodes.torch.manual_seed')
    @patch('nodes.folder_paths')
    @patch('nodes.os.path.exists')
    @patch('nodes.snapshot_download')
    @patch('nodes.AutoProcessor.from_pretrained')
    @patch('nodes.Qwen2_5_VLForConditionalGeneration.from_pretrained')
    @patch('nodes.process_vision_info')
    @patch('nodes.tensor_to_pil') # Though not directly used in text-only, it's part of the class
    def test_qwen2vl_inference_text_only(self, mock_tensor_to_pil, mock_process_vision_info, mock_qwen_model_load, mock_auto_processor_load, mock_snapshot_download, mock_os_exists, mock_folder_paths, mock_torch_seed):
        # Setup mocks
        mock_folder_paths.models_dir = "/mock/models"
        mock_os_exists.return_value = True # Assume model exists

        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "formatted_text_prompt"
        # Mock the __call__ method of the processor instance
        mock_processor_call_result = MagicMock()
        mock_processor.return_value = mock_processor_call_result
        mock_auto_processor_load.return_value = mock_processor

        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock() # Mocked generated IDs
        mock_qwen_model_load.return_value = mock_model

        # For batch_decode, it's called on the processor instance
        mock_processor.batch_decode.return_value = ["decoded_text"]

        mock_process_vision_info.return_value = (None, None) # No vision inputs for text-only

        node = Qwen2VL()
        node.bf16_support = False # Control for test predictability

        # Call inference
        result = node.inference(
            text="hello world",
            model="Qwen2.5-VL-3B-Instruct",
            quantization="none",
            keep_model_loaded=True,
            temperature=0.7,
            max_new_tokens=50,
            seed=42
        )

        # Assertions
        mock_torch_seed.assert_called_with(42)
        mock_os_exists.assert_called_once_with(os.path.join("/mock/models", "Qwen2.5-VL-3B-Instruct"))
        mock_auto_processor_load.assert_called_once()
        mock_qwen_model_load.assert_called_once()

        expected_messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'hello world'}]}]
        mock_processor.apply_chat_template.assert_called_with(expected_messages, tokenize=False, add_generation_prompt=True)
        mock_process_vision_info.assert_called_with(expected_messages)

        # Check that the processor was called with the correct arguments
        mock_processor.assert_called_with(
            text=["formatted_text_prompt"], images=None, videos=None, padding=True, return_tensors="pt"
        )
        mock_model.generate.assert_called_once()
        mock_processor.batch_decode.assert_called_once()
        self.assertEqual(result, ["decoded_text"])

    @patch('nodes.torch.manual_seed')
    @patch('nodes.folder_paths')
    @patch('nodes.os.path.exists')
    @patch('nodes.snapshot_download')
    @patch('nodes.AutoProcessor.from_pretrained')
    @patch('nodes.Qwen2_5_VLForConditionalGeneration.from_pretrained')
    @patch('nodes.process_vision_info')
    @patch('nodes.tensor_to_pil')
    def test_qwen2vl_inference_with_image(self, mock_tensor_to_pil, mock_process_vision_info, mock_qwen_model_load, mock_auto_processor_load, mock_snapshot_download, mock_os_exists, mock_folder_paths, mock_torch_seed):
        mock_folder_paths.models_dir = "/mock/models"
        mock_os_exists.return_value = True

        mock_pil_image = MagicMock(spec=Image.Image)
        mock_tensor_to_pil.return_value = mock_pil_image

        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "formatted_prompt_with_image"
        mock_processor.return_value = MagicMock() # Mock processor() call
        mock_auto_processor_load.return_value = mock_processor
        mock_processor.batch_decode.return_value = ["decoded_image_text"]

        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock()
        mock_qwen_model_load.return_value = mock_model

        # Assume process_vision_info will receive the PIL image and return appropriate structures
        mock_image_inputs_structured = MagicMock()
        mock_process_vision_info.return_value = (mock_image_inputs_structured, None)

        node = Qwen2VL()
        node.bf16_support = False

        mock_image_tensor = MagicMock(spec=torch.Tensor)

        result = node.inference(
            text="describe this image",
            model="Qwen2.5-VL-3B-Instruct",
            quantization="none",
            keep_model_loaded=True,
            temperature=0.7,
            max_new_tokens=60,
            seed=43,
            image=mock_image_tensor
        )

        mock_torch_seed.assert_called_with(43)
        mock_tensor_to_pil.assert_called_once_with(mock_image_tensor)

        expected_messages_image = [{'role': 'user', 'content': [{'type': 'image', 'image': mock_pil_image}, {'type': 'text', 'text': 'describe this image'}]}]
        # Order of image and text in content list matters for `process_vision_info`
        # The current code inserts image at index 0 of content list.
        mock_processor.apply_chat_template.assert_called_with(expected_messages_image, tokenize=False, add_generation_prompt=True)
        mock_process_vision_info.assert_called_with(expected_messages_image)

        mock_processor.assert_called_with(
            text=["formatted_prompt_with_image"], images=mock_image_inputs_structured, videos=None, padding=True, return_tensors="pt"
        )
        mock_model.generate.assert_called_once()
        mock_processor.batch_decode.assert_called_once()
        self.assertEqual(result, ["decoded_image_text"])

    @patch('nodes.tempfile.NamedTemporaryFile')
    @patch('nodes.os.remove')
    @patch('nodes.subprocess.run')
    @patch('nodes.torch.manual_seed')
    @patch('nodes.folder_paths')
    @patch('nodes.os.path.exists')
    @patch('nodes.snapshot_download')
    @patch('nodes.AutoProcessor.from_pretrained')
    @patch('nodes.Qwen2_5_VLForConditionalGeneration.from_pretrained')
    @patch('nodes.process_vision_info')
    def test_qwen2vl_inference_with_video(self, mock_process_vision_info, mock_qwen_model_load, mock_auto_processor_load, mock_snapshot_download, mock_os_exists, mock_folder_paths, mock_torch_seed, mock_subprocess_run, mock_os_remove, mock_tempfile_named):
        mock_folder_paths.models_dir = "/mock/models"
        mock_os_exists.return_value = True # Model path exists

        # Mock for tempfile
        mock_temp_file = MagicMock()
        mock_temp_file.name = "/tmp/fake_video.mp4"
        mock_tempfile_named.return_value.__enter__.return_value = mock_temp_file


        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "formatted_prompt_with_video"
        mock_processor.return_value = MagicMock() # Mock processor() call
        mock_auto_processor_load.return_value = mock_processor
        mock_processor.batch_decode.return_value = ["decoded_video_text"]

        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock()
        mock_qwen_model_load.return_value = mock_model

        mock_video_inputs_structured = MagicMock()
        mock_process_vision_info.return_value = (None, mock_video_inputs_structured) # Image is None

        node = Qwen2VL()
        node.bf16_support = False

        result = node.inference(
            text="what is in the video?",
            model="Qwen2.5-VL-7B-Instruct",
            quantization="none",
            keep_model_loaded=True,
            temperature=0.5,
            max_new_tokens=100,
            seed=44,
            video_path="/path/to/real/video.mp4"
        )

        mock_torch_seed.assert_called_with(44)
        mock_tempfile_named.assert_called_once_with(suffix=".mp4", delete=False)

        expected_ffmpeg_command_parts = ["ffmpeg", "-i", "/path/to/real/video.mp4", mock_temp_file.name]
        # Check if subprocess.run was called, and its first few arguments (command)
        called_command = mock_subprocess_run.call_args[0][0]
        self.assertEqual(called_command[0:3], expected_ffmpeg_command_parts[0:3]) # ffmpeg -i /path/to/real/video.mp4
        self.assertEqual(called_command[-1], mock_temp_file.name) # Output path is the temp file
        mock_subprocess_run.assert_called_once_with(unittest.mock.ANY, check=True, capture_output=True, text=True)


        expected_messages_video = [{'role': 'user', 'content': [{'type': 'video', 'video': mock_temp_file.name}, {'type': 'text', 'text': 'what is in the video?'}]}]
        mock_processor.apply_chat_template.assert_called_with(expected_messages_video, tokenize=False, add_generation_prompt=True)
        mock_process_vision_info.assert_called_with(expected_messages_video)

        mock_processor.assert_called_with(
            text=["formatted_prompt_with_video"], images=None, videos=mock_video_inputs_structured, padding=True, return_tensors="pt"
        )
        mock_model.generate.assert_called_once()
        mock_processor.batch_decode.assert_called_once()
        mock_os_remove.assert_called_once_with(mock_temp_file.name)
        self.assertEqual(result, ["decoded_video_text"])

    @patch('nodes.torch.cuda.empty_cache')
    @patch('nodes.torch.cuda.ipc_collect')
    @patch('nodes.folder_paths')
    @patch('nodes.os.path.exists')
    @patch('nodes.AutoProcessor.from_pretrained')
    @patch('nodes.Qwen2_5_VLForConditionalGeneration.from_pretrained')
    @patch('nodes.process_vision_info') # Still need to mock this even if not primary
    def test_qwen2vl_keep_model_loaded_false(self, mock_process_vision_info, mock_qwen_model_load, mock_auto_processor_load, mock_os_exists, mock_folder_paths, mock_ipc_collect, mock_empty_cache):
        mock_folder_paths.models_dir = "/mock/models"
        mock_os_exists.return_value = True

        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = "prompt"
        mock_processor.return_value = MagicMock()
        mock_auto_processor_load.return_value = mock_processor
        mock_processor.batch_decode.return_value = ["text_output"]

        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock()
        mock_qwen_model_load.return_value = mock_model

        mock_process_vision_info.return_value = (None, None)

        node = Qwen2VL()
        node.bf16_support = False

        # Ensure model and processor are initially set by mocks
        node.model = mock_model
        node.processor = mock_processor

        node.inference(text="test", model="Qwen2.5-VL-3B-Instruct", quantization="none", keep_model_loaded=False, temperature=0.7, max_new_tokens=10, seed=1)

        self.assertIsNone(node.model)
        self.assertIsNone(node.processor)
        mock_empty_cache.assert_called_once()
        mock_ipc_collect.assert_called_once()


class TestQwen2Node(unittest.TestCase):

    def test_input_types(self):
        inputs = Qwen2.INPUT_TYPES()
        self.assertIn("required", inputs)
        required_keys = ["system", "prompt", "model", "quantization", "keep_model_loaded", "temperature", "max_new_tokens", "seed"]
        for key in required_keys:
            self.assertIn(key, inputs["required"])

        self.assertEqual(inputs["required"]["model"][1]["default"], "Qwen2.5-7B-Instruct")
        self.assertEqual(inputs["required"]["prompt"][1]["default"], "")

    @patch('nodes.torch.cuda.is_available')
    @patch('nodes.torch.cuda.get_device_capability')
    def test_initialization(self, mock_get_device_capability, mock_is_available):
        mock_is_available.return_value = True
        mock_get_device_capability.return_value = (8, 0) # Simulate a GPU that supports bf16

        node = Qwen2()
        self.assertIsNone(node.model_checkpoint)
        self.assertIsNone(node.tokenizer)
        self.assertIsNone(node.model)
        self.assertTrue(node.bf16_support) # Based on mocks

    @patch('nodes.torch.cuda.is_available')
    @patch('nodes.torch.cuda.get_device_capability')
    def test_empty_prompt_in_inference(self, mock_get_device_capability, mock_is_available):
        mock_is_available.return_value = False # Simulate CPU
        mock_get_device_capability.return_value = (7,5) # Not relevant for this test but good practice
        node = Qwen2()
        # We are not testing the full inference, just the initial prompt check
        result = node.inference(system="System prompt", prompt=" ", model="Qwen2.5-7B-Instruct", quantization="none", keep_model_loaded=False, temperature=0.7, max_new_tokens=512, seed=-1)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0], "Error: Prompt input is empty.")

    @patch('nodes.torch.manual_seed')
    @patch('nodes.folder_paths')
    @patch('nodes.os.path.exists')
    @patch('nodes.snapshot_download')
    @patch('nodes.AutoTokenizer.from_pretrained')
    @patch('nodes.AutoModelForCausalLM.from_pretrained')
    def test_qwen2_inference_basic(self, mock_model_load, mock_tokenizer_load, mock_snapshot_download, mock_os_exists, mock_folder_paths, mock_torch_seed):
        mock_folder_paths.models_dir = "/mock/models_qwen2"
        mock_os_exists.return_value = True # Assume model exists

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted_chat_prompt"
        # Mock the __call__ method of the tokenizer instance
        mock_tokenizer_call_result = MagicMock()
        mock_tokenizer_call_result.input_ids = MagicMock() # Mock input_ids attribute for zipping
        mock_tokenizer.return_value = mock_tokenizer_call_result
        mock_tokenizer_load.return_value = mock_tokenizer
        mock_tokenizer.batch_decode.return_value = ["decoded_qwen2_text"]


        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock() # Mocked generated IDs
        mock_model_load.return_value = mock_model

        node = Qwen2()
        node.bf16_support = True # Control for test

        result = node.inference(
            system="You are an AI.",
            prompt="Hello Qwen2",
            model="Qwen2.5-7B-Instruct",
            quantization="none",
            keep_model_loaded=True,
            temperature=0.6,
            max_new_tokens=70,
            seed=45
        )

        mock_torch_seed.assert_called_with(45)
        mock_os_exists.assert_called_once_with(os.path.join("/mock/models_qwen2", "Qwen2.5-7B-Instruct"))
        mock_tokenizer_load.assert_called_once()
        mock_model_load.assert_called_once()

        expected_messages = [
            {"role": "system", "content": "You are an AI."},
            {"role": "user", "content": "Hello Qwen2"},
        ]
        mock_tokenizer.apply_chat_template.assert_called_with(expected_messages, tokenize=False, add_generation_prompt=True)

        # Check that the tokenizer was called with the correct arguments
        # In Qwen2, tokenizer is called with [text]
        mock_tokenizer.assert_called_with(["formatted_chat_prompt"], return_tensors="pt")
        mock_model.generate.assert_called_once()
        mock_tokenizer.batch_decode.assert_called_once()
        self.assertEqual(result, ["decoded_qwen2_text"])

    @patch('nodes.torch.cuda.empty_cache')
    @patch('nodes.torch.cuda.ipc_collect')
    @patch('nodes.folder_paths')
    @patch('nodes.os.path.exists')
    @patch('nodes.AutoTokenizer.from_pretrained')
    @patch('nodes.AutoModelForCausalLM.from_pretrained')
    def test_qwen2_keep_model_loaded_false(self, mock_model_load, mock_tokenizer_load, mock_os_exists, mock_folder_paths, mock_ipc_collect, mock_empty_cache):
        mock_folder_paths.models_dir = "/mock/models_qwen2"
        mock_os_exists.return_value = True

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.return_value = MagicMock(input_ids=MagicMock()) # for inputs.input_ids
        mock_tokenizer_load.return_value = mock_tokenizer
        mock_tokenizer.batch_decode.return_value = ["text_output"]

        mock_model = MagicMock()
        mock_model.generate.return_value = MagicMock()
        mock_model_load.return_value = mock_model

        node = Qwen2()
        node.bf16_support = True

        # Ensure model and tokenizer are initially set by mocks
        node.model = mock_model
        node.tokenizer = mock_tokenizer

        node.inference(system="sys", prompt="user prompt", model="Qwen2.5-7B-Instruct", quantization="none", keep_model_loaded=False, temperature=0.7, max_new_tokens=10, seed=1)

        self.assertIsNone(node.model)
        self.assertIsNone(node.tokenizer)
        mock_empty_cache.assert_called_once()
        mock_ipc_collect.assert_called_once()


if __name__ == '__main__':
    unittest.main()
