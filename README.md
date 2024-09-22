
# ComfyUI Qwen2-VL wrapper

## Sample workflow
You can find a sample [workflow](workflow/Qwen2VL.json) here.

![alt text](workflow/comfy_workflow.png)

Additionally, you can use Qwen2.5 for text generation
![alt text](workflow/comfy_workflow2.png)

A sample [workflow](workflow/qwen25.json) using both nodes

## Installation
Install from ComfyUI Manager, search for `Qwen2-VL wrapper for ComfyUI`

To install ComfyUI_QwenVL in `ComfyUI\custom_nodes\`, follow these steps:

1. *Clone the repository*:
    ```bash
    git clone https://github.com/alexcong/ComfyUI_QwenVL.git
    ```

2. *Navigate to the cloned directory*:
    ```bash
    cd ComfyUI_QwenVL
    ```

3. *Install the required dependencies*:
    ```bash
    pip install -r requirements.txt
    ```

## Qwen2-VL models location
Models will be downloaded to `ComfyUI\models\LLM\`
