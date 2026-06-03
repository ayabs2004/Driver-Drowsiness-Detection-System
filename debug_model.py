
import os
import cv2
import numpy as np
import keras
from keras.models import load_model

MODEL_PATH = r"c:\Users\aya\Documents\driver ai project\drowsiness_model.keras"
IMG_SIZE = (64, 64)

def test_loading():
    print(f"Checking model path: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("❌ Model file not found!")
        return False
    
    print("Attempting to load model with compile=False...")
    try:
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully!")
        model.summary()
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def test_inference(model):
    print("Testing inference with dummy data...")
    try:
        dummy_input = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 3).astype(np.float32)
        prediction = model.predict(dummy_input, verbose=0)
        print(f"✅ Inference successful! Prediction shape: {prediction.shape}")
        return True
    except Exception as n:
        print(f"❌ Inference failed: {n}")
        return False

if __name__ == "__main__":
    model = test_loading()
    if model:
        test_inference(model)
