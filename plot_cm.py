import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR = r"c:\Users\aya\Documents\driver ai project\processed_data"
MODEL_PATH = r"c:\Users\aya\Documents\driver ai project\drowsiness_model.keras"
IMG_SIZE = (96, 96)
TARGET_CLASSES = ['closed_eye', 'closed_mouth', 'open_eye', 'open_mouth']

print("Loading model...")
model = load_model(MODEL_PATH)

print("Loading test data...")
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    os.path.join(BASE_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical',
    classes=TARGET_CLASSES,
    shuffle=False
)

print("Predicting...")
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

# Generate Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Plotting
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=TARGET_CLASSES, 
            yticklabels=TARGET_CLASSES,
            annot_kws={"size": 14})

plt.title('Driver Drowsiness - Test Set Confusion Matrix', fontsize=16, pad=20)
plt.ylabel('True Class', fontsize=14)
plt.xlabel('Predicted Class', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

output_file = "confusion_matrix.png"
plt.savefig(output_file, dpi=300)
print(f"\n✅ Visual confusion matrix saved to: {output_file}")

# Print basic stats to console
print("\nAccuracy on Test Set: {:.2f}%".format(np.mean(y_true == y_pred) * 100))
