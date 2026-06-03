
import os
import json
import numpy as np
import keras
from keras import layers
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Configuration (matching train_custom_model.py)
IMG_SIZE = (96, 96)
BATCH_SIZE = 32
MODEL_PATH = r"c:\Users\aya\Documents\driver ai project\drowsiness_model.keras"
TEST_DIR = r"c:\Users\aya\Documents\driver ai project\processed_data\test"
TARGET_CLASSES = ['closed_eye', 'closed_mouth', 'open_eye', 'open_mouth']


def draw_ascii_plot(title, data, width=50, height=10):
    if not data:
        return
    
    max_val = max(data)
    min_val = min(data)
    if max_val == min_val:
        max_val += 0.1
        
    print(f"\n{title} (Epochs 1 to {len(data)})")
    print(f"{max_val:.4f} " + "+" + "-" * width)
    
    for h in range(height, -1, -1):
        line = "       │"
        threshold = min_val + (max_val - min_val) * (h / height)
        
        for i in range(width):
            idx = int(i * len(data) / width)
            if idx < len(data) and data[idx] >= threshold:
                line += "*"
            else:
                line += " "
        
        if h == height // 2:
            print(f"{threshold:.4f} {line}")
        elif h == 0:
            print(f"{min_val:.4f} {line}")
        else:
            print(f"       {line}")
    
    print("       " + "└" + "─" * width)

def show_history():
    HISTORY_PATH = r"c:\Users\aya\Documents\driver ai project\history.json"
    if not os.path.exists(HISTORY_PATH):
        print("\nNote: Training history file (history.json) not found.")
        print("    Curves will be available once you run train_custom_model.py again.")
        return

    print("\n" + "="*50)
    print("TRAINING HISTORY CURVES")
    print("="*50)
    
    with open(HISTORY_PATH, 'r') as f:
        history = json.load(f)
    
    if 'accuracy' in history:
        draw_ascii_plot("ACCURACY EVOLUTION", history['accuracy'])
    if 'loss' in history:
        draw_ascii_plot("LOSS EVOLUTION", history['loss'])

def evaluate():
    import json
    print("-" * 50)
    print("MODEL PERFORMANCE REPORT")
    print("-" * 50)

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    # 1. Load Model
    print(f"Loading model: {os.path.basename(MODEL_PATH)}...")
    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load Testing Dataset
    print(f"Loading test data from: {TEST_DIR}...")
    try:
        test_ds = keras.utils.image_dataset_from_directory(
            TEST_DIR,
            labels='inferred',
            label_mode='categorical',
            class_names=TARGET_CLASSES,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=False  # Crucial for confusion matrix
        )
        class_names = test_ds.class_names
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 3. Preprocessing
    rescaling_layer = layers.Rescaling(1./255)
    test_ds = test_ds.map(lambda x, y: (rescaling_layer(x), y))

    # 4. Predictions
    print("Running inference on test set (this may take a moment)...")
    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 5. Metrics Calculation
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    # 6. Confusion Matrix (Text-based)
    print("\n" + "="*50)
    print("CONFUSION MATRIX")
    print("="*50)
    cm = confusion_matrix(y_true, y_pred)
    
    # Pretty print confusion matrix
    header = "True \\ Pred | " + " | ".join([f"{c:10}" for c in class_names])
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:10} | " + " | ".join([f"{val:10}" for val in row])
        print(row_str)

    print("\n" + "="*50)
    accuracy = np.mean(y_true == y_pred)
    print(f"OVERALL ACCURACY: {accuracy:.2%}")
    print("="*50)

if __name__ == "__main__":
    # Ensure sklearn is available, if not, we can't run the report easily
    try:
        import sklearn
    except ImportError:
        print("Installing scikit-learn for metrics calculation...")
        os.system("pip install scikit-learn")
    
    evaluate()
    show_history()
