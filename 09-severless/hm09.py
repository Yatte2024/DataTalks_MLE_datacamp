"""
ML Zoomcamp 2025 - Homework 9
Full solution code for Q1–Q6 in one Python file.
All comments are written in English.
"""

import subprocess
from io import BytesIO
from urllib import request

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image


# ============================================================
# Shared helpers
# ============================================================

def download_image(url: str) -> Image.Image:
    """Download image from URL and return a PIL Image."""
    with request.urlopen(url) as resp:
        buffer = resp.read()
    return Image.open(BytesIO(buffer))


def prepare_image(img: Image.Image, target_size=(200, 200)) -> Image.Image:
    """Convert image to RGB and resize to the target size."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img.resize(target_size, Image.NEAREST)


def preprocess(img: Image.Image) -> np.ndarray:
    """
    Apply the same preprocessing as in Homework 8.

    IMPORTANT:
    - This function returns a tensor with shape (3, H, W) (no batch dimension).
    - Model expects NCHW: (1, 3, 200, 200), so batch dimension is added later.
    """
    # Convert to numpy array with shape (H, W, C)
    x = np.array(img).astype("float32") / 255.0

    # Use the same mean and std as in HW8 (example values)
    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")

    # Normalize per channel: broadcasting over height and width
    x = (x - mean) / std

    # Change from (H, W, C) to (C, H, W)
    x = np.transpose(x, (2, 0, 1))

    # Shape is now (3, H, W); batch dim will be added outside
    return x


# ============================================================
# Q1 – Inspect ONNX model (input/output names)
# ============================================================

def answer_q1():
    """Print ONNX model input and output names."""
    model = onnx.load("./raw_data/hair_classifier_v1.onnx")


    print("\n=== Q1: ONNX Model Outputs ===")
    for out in model.graph.output:
        print("Output:", out.name)


# ============================================================
# Q2 – Target size
# ============================================================

def answer_q2():
    """Verify that the resized image is 200x200."""
    url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
    img = download_image(url)
    img_resized = prepare_image(img, target_size=(200, 200))

    print("\n=== Q2 ===")
    print("Original size:", img.size)
    print("Resized size:", img_resized.size)
    print("Q2 multiple-choice answer: 200x200")


# ============================================================
# Q3 – First pixel, R channel after preprocessing
# ============================================================

def answer_q3():
    """
    Compute first pixel R after preprocessing.

    Note:
    - preprocess() returns (3, H, W)
    - So the first pixel R is x[0, 0, 0].
    """
    url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
    img = prepare_image(download_image(url), target_size=(200, 200))
    x = preprocess(img)  # shape: (3, 200, 200)

    first_pixel_r = x[0, 0, 0]

    print("\n=== Q3 ===")
    print("First pixel R after preprocessing:", first_pixel_r)
    print("Q3 multiple-choice answer (closest): -1.073")


# ============================================================
# Q4 – Run hair_classifier_v1.onnx on the image
# ============================================================

def answer_q4():
    """
    Run the ONNX model and print prediction.

    IMPORTANT:
    - Input tensor must be NCHW: (1, 3, 200, 200).
    - Do NOT add batch dimension twice.
    """
    url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"

    # Prepare image
    img = prepare_image(download_image(url), target_size=(200, 200))

    # Preprocess -> (3, 200, 200)
    x = preprocess(img)

    # Add batch dimension once -> (1, 3, 200, 200)
    x = np.expand_dims(x, axis=0)

    print("\n=== Q4 ===")
    print("Input tensor shape (should be 1, 3, 200, 200):", x.shape)

    # Load ONNX model
    session = ort.InferenceSession("./raw_data/hair_classifier_v1.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print("Input name:", input_name)
    print("Output name:", output_name)

    # Run model
    pred = session.run([output_name], {input_name: x})[0]
    score = float(pred[0])

    print("Model output:", score)
    print("Q4 multiple-choice answer (closest): 0.89")


# ============================================================
# Q5 – Docker base image size
# ============================================================

def answer_q5():
    """Pull Docker image and display its size."""
    image = "agrigorev/model-2025-hairstyle:v1"

    print("\n=== Q5 ===")
    print("Pulling image (requires Docker installed)...")
    subprocess.run(["docker", "pull", image], check=True)

    result = subprocess.run(
        ["docker", "images", image, "--format", "{{.Size}}"],
        check=True,
        capture_output=True,
        text=True,
    )

    size = result.stdout.strip()
    print("Image size reported by Docker:", size)

# ============================================================
# Q6 – Run hair_classifier_empty.onnx
# ============================================================

def answer_q6():
    """
    Run the 'empty' ONNX model with the same preprocessing.

    IMPORTANT:
    - Same input shape: (1, 3, 200, 200).
    """
    url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"

    img = prepare_image(download_image(url), target_size=(200, 200))
    x = preprocess(img)               # (3, 200, 200)
    x = np.expand_dims(x, axis=0)     # (1, 3, 200, 200)

    print("\n=== Q6 ===")
    print("Input tensor shape (should be 1, 3, 200, 200):", x.shape)

    session = ort.InferenceSession("hair_classifier_empty.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    pred = session.run([output_name], {input_name: x})[0]
    score = float(pred[0])

    print("Empty model output:", score)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":


    answer_q1()
    answer_q2()
    answer_q3()
    answer_q4()


