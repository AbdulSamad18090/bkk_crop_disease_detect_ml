import argparse
import os
import tensorflow as tf

def get_args():
    p = argparse.ArgumentParser(description="Test the trained crop disease classifier")
    p.add_argument("--data_dir", default="dataset/plantCity/test", help="Path to test dataset directory")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--image_size", type=int, default=224, help="Square image size (e.g. 224)")
    p.add_argument("--model_path", default="models/saved_model", help="Path to the saved model")
    return p.parse_args()

def main():
    args = get_args()
    
    if not os.path.exists(args.data_dir):
        raise SystemExit(f"Test data directory not found: {args.data_dir}")
        
    if not os.path.exists(args.model_path):
        raise SystemExit(f"Model not found at: {args.model_path}")

    print(f"Loading model from {args.model_path}...")
    try:
        model = tf.keras.models.load_model(args.model_path)
    except Exception as e:
        raise SystemExit(f"Failed to load model: {e}")

    print(f"Loading test dataset from {args.data_dir}...")
    image_size = (args.image_size, args.image_size)
    
    try:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            args.data_dir,
            seed=123,
            image_size=image_size,
            batch_size=args.batch_size,
        )
    except ValueError as e:
         raise SystemExit(f"Error loading dataset (directory might be empty or invalid structure): {e}")

    class_names = test_ds.class_names
    print(f"Found {len(class_names)} classes: {class_names}")

    # Evaluate
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(test_ds)
    
    print(f"\nTest Results:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
