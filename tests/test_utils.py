import torch
from PIL import Image
import numpy as np

# Assuming tensor_to_pil is in nodes.py and accessible.
# If nodes.py is in the parent directory, Python's import mechanisms
# might require specific sys.path adjustments when running tests directly,
# or proper packaging. For now, we'll assume it can be imported.
# A common way to handle imports from parent directory for tests:
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nodes import tensor_to_pil

def test_conversion_rgb_image():
    # Create a sample RGB image tensor: shape (batch, height, width, channels)
    # The function expects [batch, height, width, channels]
    # and internally uses batch_index=0 and squeezes.
    # So a (1, 64, 64, 3) tensor is appropriate.
    sample_tensor = torch.rand(1, 64, 64, 3)
    pil_image = tensor_to_pil(sample_tensor)
    assert isinstance(pil_image, Image.Image), "Output should be a PIL Image"
    assert pil_image.mode == "RGB", f"Expected mode RGB, got {pil_image.mode}"
    assert pil_image.size == (64, 64), f"Expected size (64,64), got {pil_image.size}"

def test_conversion_grayscale_image():
    # Create a sample grayscale image tensor: shape (batch, height, width, 1)
    # The current tensor_to_pil function uses .squeeze() at the end,
    # which would remove the channel dimension if it's 1.
    # PIL Image.fromarray for a 2D array creates an 'L' mode image.
    sample_tensor = torch.rand(1, 64, 64, 1)
    pil_image = tensor_to_pil(sample_tensor) # Will be (64,64) after squeeze
    assert isinstance(pil_image, Image.Image), "Output should be a PIL Image"
    # For a 2D numpy array, fromarray typically results in 'L' mode (grayscale)
    assert pil_image.mode == "L", f"Expected mode L (grayscale), got {pil_image.mode}"
    assert pil_image.size == (64, 64), f"Expected size (64,64), got {pil_image.size}"

def test_conversion_batched_tensor_selects_first_image():
    # Create a batch of 2 images, 3 channels (RGB)
    sample_tensor = torch.rand(2, 64, 64, 3)
    # Modify the first image in the batch to be identifiable if needed,
    # but tensor_to_pil hardcodes batch_index=0, so we just check shape/type.
    pil_image = tensor_to_pil(sample_tensor, batch_index=0)
    assert isinstance(pil_image, Image.Image), "Output should be a PIL Image"
    assert pil_image.mode == "RGB", f"Expected mode RGB, got {pil_image.mode}"
    assert pil_image.size == (64, 64), f"Expected size (64,64), got {pil_image.size}"

    # Test with a different batch_index if the function were to support it
    # For now, this part is more of a forward-looking thought as batch_index is hardcoded
    # If you were to test tensor_to_pil(sample_tensor, batch_index=1)
    # you'd expect similar results for the second image.
