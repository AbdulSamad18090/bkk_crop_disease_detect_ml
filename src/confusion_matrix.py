import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

CLASS_NAMES = [
    'Apple Black_spot', 'Apple Brown_spot', 'Apple Normal',
    'Apricot blight leaf disease', 'Apricot Normal', 'Apricot shot_hole',
    'Bean bean rust image', 'Bean Fungal_leaf disease', 'Bean Normal leaf', 'Bean shot_hole',
    'Cherry brown_spot', 'Cherry Leaf Scorch', 'Cherry Normal leaf', 'Cherry purple leaf spot', 'Cherry_shot hole disease',
    'Corn Fungal leaf', 'Corn gray leaf spot', 'Corn holcus_ leaf spot', 'Corn Normal leaf',
    'Fig Blight_leaf disease', 'Fig Brown spot', 'Fig normal leaf', 'Fig_rust leaf',
    'Grape Anthracnose leaf', 'Grape Brown spot leaf', 'Grape Downy mildew leaf', 'Grape Mites_leaf disease', 'Grape Normal_leaf', 'Grape Powdery_mildew leaf', 'Grape shot hole leaf disease',
    'lokat Leaf_spot', 'Lokat Normal leaf',
    'Pear Black spot _ leaf disease', 'Pear fire blight', 'Pear Normal _leaf',
    'persimmons Brown_spot',
    'tomato Fusarium Wilt', 'tomato spider mites', 'tomato verticillium wilt', 'tomato_bacterial_spot', 'tomato_early_blight', 'tomato_healthy_leaf', 'tomato_late_blight', 'tomato_leaf_curl', 'tomato_leaf_miner', 'tomato_leaf_mold', 'tomato_septoria_leaf',
    'Walnut Anthracnose_leaf disease', 'Walnut Blotch_leaf disease', 'Walnut leaf gall mite', 'Walnut Normal_leaf', 'Walnut Shot_hole'
]

def get_args():
    p = argparse.ArgumentParser(description="Generate confusion matrix for crop disease classifier")
    p.add_argument("--data_dir", default="dataset/plantCity/test", help="Path to test dataset directory")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--image_size", type=int, default=224, help="Square image size (e.g. 224)")
    p.add_argument("--model_path", default="models/torch_model.pth", help="Path to the saved model")
    p.add_argument("--output_dir", default="outputs", help="Directory to save confusion matrix image")
    p.add_argument("--normalize", action="store_true", help="Normalize confusion matrix values")
    return p.parse_args()

def main():
    args = get_args()

    if not os.path.exists(args.data_dir):
        raise SystemExit(f"Test data directory not found: {args.data_dir}")

    if not os.path.exists(args.model_path):
        raise SystemExit(f"Model not found at: {args.model_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.model_path}...")
    try:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        raise SystemExit(f"Failed to load model: {e}")

    print(f"Loading test dataset from {args.data_dir}...")

    test_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        test_dataset = datasets.ImageFolder(args.data_dir, transform=test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    except Exception as e:
        raise SystemExit(f"Error loading dataset: {e}")

    class_names = test_dataset.classes
    print(f"Found {len(class_names)} classes")

    # Collect predictions
    print("\nRunning inference...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Predicting", unit="batch"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate accuracy
    accuracy = (all_preds == all_labels).sum() / len(all_labels)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)

    if args.normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    # Plot confusion matrix
    plt.figure(figsize=(25, 20))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 6}
    )
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    # Save confusion matrix
    output_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved to: {output_path}")

    # Save classification report
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(report)

    report_path = os.path.join(args.output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Classification report saved to: {report_path}")

    plt.show()

if __name__ == "__main__":
    main()
