from .nodes import *

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Qwen2VL": Qwen2VL,
    "Qwen2.5": Qwen2
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen2VL": "Qwen2VL",
    "Qwen2.5": "Qwen2.5",
}
