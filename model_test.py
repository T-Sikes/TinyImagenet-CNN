#!/usr/bin/env python
"""
model_test.py
Standalone script to load trained model weights and run a full validation check.
"""

import torch, os, gdown
from collections import Counter

from tinyimagenet_cnn import theBestCNN, val_loader  

# --- Device ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def download_model(path="model.pth"):
    if os.path.exists(path):
        print(f"✅ Model file already exists at {path}")
        return

    print("⬇️ Downloading model weights from Google Drive...")
    
    file_id = "19ExAOiFKCkubhvjn-XHpu3Lit3f3-FTA"
    url = f"https://drive.google.com/uc?id={file_id}"

    try:
        gdown.download(url, path, quiet=False)
        print(f"✅ Model downloaded and saved to {path}")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")

# === Model Loading Function ===
def load_model(path="model.pth", num_classes=15):
    download_model(path)
    """
    Load PyTorch model weights from a file.
    Returns the model on the specified device.
    """
    model = theBestCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded from {path}")
    return model

# === Prediction Function ===
def predict(model, images):
    """
    Predict classes for a batch of images.
    Args:
        model: the trained PyTorch model
        images: torch.Tensor of shape (N, C, H, W)
    Returns:
        preds: torch.Tensor of predicted class indices (N,)
    """
    model.eval()
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    return predictions

# === Comprehensive Model Test ===
def comprehensive_model_test(model_path="model.pth"):
    model = load_model(model_path)
    model.eval()
    
    print("🧪 COMPREHENSIVE MODEL DIAGNOSTICS")
    print("=" * 50)
    
    # Full validation accuracy
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            preds = predict(model, images)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_acc = correct / total
    print(f"1. Full Validation Accuracy: {val_acc:.4f} ({correct}/{total})")
    
    # Per-class accuracy
    class_correct = [0] * 15
    class_total = [0] * 15
    
    for pred, label in zip(all_predictions, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
    
    print("\n2. Per-Class Accuracy:")
    weak_classes = []
    for i in range(15):
        acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"   Class {i:2d}: {acc:.3f} ({class_correct[i]}/{class_total[i]})")
        if acc < 0.6:
            weak_classes.append(i)
    
    # Prediction diversity
    unique_preds = len(Counter(all_predictions))
    print(f"\n3. Prediction Diversity: {unique_preds}/15 classes predicted")
    
    # Batch consistency
    print("\n4. Batch Consistency Test:")
    for i in range(3):
        test_images, test_labels = next(iter(val_loader))
        test_images = test_images.to(device)
        test_labels = test_labels.to(device)
        preds = predict(model, test_images)
        unique_in_batch = len(torch.unique(preds))
        print(f"   Batch {i}: {unique_in_batch} unique classes predicted")
    
    # Final assessment
    print("\n5. FINAL ASSESSMENT:")
    if val_acc >= 0.78 and unique_preds >= 12 and len(weak_classes) <= 3:
        print("   ✅ EXCELLENT - High chance of test set success!")
    elif val_acc >= 0.75 and unique_preds >= 10:
        print("   ✅ GOOD - Competitive model")
    else:
        print("   ⚠️ NEEDS IMPROVEMENT - Test set performance uncertain")
        if weak_classes:
            print(f"   Weak classes: {weak_classes}")
        if unique_preds < 10:
            print(f"   Only predicting {unique_preds}/15 classes")
    
    return val_acc, weak_classes

# === Run test when script is executed ===
if __name__ == "__main__":
    final_accuracy, weak_classes = comprehensive_model_test()
    print(f"\n🎯 Validation Accuracy: {final_accuracy:.4f}")
    if weak_classes:
        print(f"⚠️ Weak classes: {weak_classes}")
