import argparse
import logging
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_args():
    p = argparse.ArgumentParser(description="Train a crop disease classifier using PyTorch")
    p.add_argument("--data_dir", default="dataset/plantCity/train", help="Dataset path")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--learning_rate", type=float, default=0.001)
    p.add_argument("--output", default="models/torch_model.pth")
    return p.parse_args()

def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    since = time.time()
    best_acc = 0.0
    best_wts = model.state_dict()

    total_train_batches = len(dataloaders["train"])
    total_val_batches = len(dataloaders["val"])

    logger.info(f"Starting training for {num_epochs} epochs")
    logger.info(f"Train batches: {total_train_batches}, Val batches: {total_val_batches}")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"{'='*50}")

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0
            phase_batches = total_train_batches if phase == "train" else total_val_batches

            # Progress bar for batches
            batch_iterator = tqdm(
                dataloaders[phase],
                desc=f"  {phase.upper():5}",
                total=phase_batches,
                unit="batch",
                ncols=100,
                leave=True
            )

            for batch_idx, (inputs, labels) in enumerate(batch_iterator):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

                # Update progress bar with current loss
                current_loss = running_loss / ((batch_idx + 1) * inputs.size(0))
                batch_iterator.set_postfix(loss=f"{current_loss:.4f}")

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            logger.info(f"  {phase.upper():5} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = model.state_dict()
                logger.info(f"  >> New best model saved! (Acc: {best_acc:.4f})")

        epoch_time = time.time() - epoch_start
        logger.info(f"  Epoch completed in {epoch_time:.1f}s")

    time_elapsed = time.time() - since
    logger.info(f"\n{'='*50}")
    logger.info(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    logger.info(f"Best Validation Accuracy: {best_acc:.4f}")
    logger.info(f"{'='*50}")

    model.load_state_dict(best_wts)
    return model


def main():
    args = get_args()

    logger.info("=" * 50)
    logger.info("Crop Disease Classifier Training")
    logger.info("=" * 50)
    logger.info(f"Configuration:")
    logger.info(f"  Data directory: {args.data_dir}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Image size: {args.image_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Output: {args.output}")

    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        raise SystemExit(f"Data directory not found: {args.data_dir}")

    # -----------------------------
    # 🔥 DATA TRANSFORMS
    # -----------------------------
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Load dataset once
    full_dataset = datasets.ImageFolder(args.data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(args.data_dir, transform=val_transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_ds, _ = torch.utils.data.random_split(full_dataset, [train_size, val_size], generator)
    _, val_ds = torch.utils.data.random_split(val_dataset, [train_size, val_size], generator)

    dataloaders = {
        "train": torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4),
        "val": torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    }

    num_classes = len(full_dataset.classes)
    logger.info(f"Dataset loaded successfully")
    logger.info(f"  Total samples: {len(full_dataset)}")
    logger.info(f"  Training samples: {train_size}")
    logger.info(f"  Validation samples: {val_size}")
    logger.info(f"  Number of classes: {num_classes}")
    logger.info(f"  Classes: {full_dataset.classes}")

    # -----------------------------
    # GPU CHECK
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA version: {torch.version.cuda}")

    # -----------------------------
    # MODEL
    # -----------------------------
    logger.info("Loading MobileNetV2 pretrained model...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze all layers
    for p in model.parameters():
        p.requires_grad = False

    # Replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    logger.info(f"Model classifier replaced for {num_classes} classes")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Train
    logger.info("\nStarting training...")
    model = train_model(model, dataloaders, criterion, optimizer, args.epochs, device)

    # Save
    logger.info(f"\nSaving model to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)

    logger.info(f"Model saved successfully to: {args.output}")


if __name__ == "__main__":
    main()
