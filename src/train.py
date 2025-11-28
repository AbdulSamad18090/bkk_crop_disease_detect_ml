# train.py
import argparse
import os
import tensorflow as tf


def get_args():
    p = argparse.ArgumentParser(description="Train a crop disease classifier using transfer learning")
    p.add_argument("--data_dir", default="dataset/train", help="Path to training dataset directory (organized by class)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--image_size", type=int, default=224, help="Square image size (e.g. 224)")
    p.add_argument("--output", default="models/saved_model", help="Path to save/load the trained model")
    p.add_argument("--fine_tune", type=int, default=0, help="Number of layers from the end of the base model to unfreeze for fine-tuning (0 = none)")
    return p.parse_args()


def main():
    args = get_args()
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        raise SystemExit(f"Data directory not found: {data_dir}")

    image_size = (args.image_size, args.image_size)

    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=image_size,
        batch_size=args.batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=image_size,
        batch_size=args.batch_size,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # Check if saved model exists
    if os.path.exists(args.output):
        print(f"Loading existing model from {args.output}")
        model = tf.keras.models.load_model(args.output)
    else:
        print("Creating a new model")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(args.image_size, args.image_size, 3),
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False

        inputs = tf.keras.Input(shape=(args.image_size, args.image_size, 3))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

    model.summary()

    # Train
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    # Optional fine-tuning
    if args.fine_tune > 0:
        print(f"Fine-tuning last {args.fine_tune} layers")
        base_model = model.layers[2]  # base_model is the 3rd layer in this structure
        base_model.trainable = True
        for layer in base_model.layers[:-args.fine_tune]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        model.fit(train_ds, validation_data=val_ds, epochs=3)

    # Save model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save(args.output)
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
