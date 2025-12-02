import argparse
import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
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
    p = argparse.ArgumentParser(description="Test the trained crop disease classifier")
    p.add_argument("--data_dir", default="dataset/plantCity/test", help="Path to test dataset directory")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--image_size", type=int, default=224, help="Square image size (e.g. 224)")
    p.add_argument("--model_path", default="models/torch_model.pth", help="Path to the saved model")
    return p.parse_args()

def main():
    args = get_args()

    if not os.path.exists(args.data_dir):
        raise SystemExit(f"Test data directory not found: {args.data_dir}")

    if not os.path.exists(args.model_path):
        raise SystemExit(f"Model not found at: {args.model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.model_path}...")
    try:
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()
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
    print(f"Found {len(class_names)} classes: {class_names}")

    # Evaluate
    print("\nEvaluating model...")
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    num_batches = len(test_loader)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Testing", unit="batch")):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Log every 10 batches
            if (batch_idx + 1) % 10 == 0:
                current_acc = correct / total
                current_loss = running_loss / total
                print(f"\n  Batch [{batch_idx + 1}/{num_batches}] - Loss: {current_loss:.4f}, Accuracy: {current_acc:.4f}")

    avg_loss = running_loss / total
    accuracy = correct / total

    print(f"\nTest Results:")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

if __name__ == "__main__":
    main()
