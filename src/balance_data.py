import os
import random
import shutil
from PIL import Image, ImageOps

def balance_dataset(data_dir):
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return

    # 1. Count images in each class
    class_counts = {}
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    print("Analyzing class distribution...")
    max_count = 0
    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        count = len(images)
        class_counts[class_name] = count
        if count > max_count:
            max_count = count
        print(f"  {class_name}: {count}")

    print(f"\nTarget count per class: {max_count}")

    # 2. Oversample
    print("\nBalancing dataset...")
    for class_name in classes:
        current_count = class_counts[class_name]
        if current_count < max_count:
            diff = max_count - current_count
            print(f"  Augmenting {class_name} by {diff} images...")
            
            class_path = os.path.join(data_dir, class_name)
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if not images:
                print(f"    Warning: No images found in {class_name}, skipping.")
                continue

            generated_count = 0
            while generated_count < diff:
                # Pick a random image to augment
                img_name = random.choice(images)
                img_path = os.path.join(class_path, img_name)
                
                try:
                    with Image.open(img_path) as img:
                        # Apply simple augmentations
                        aug_type = random.choice(['flip', 'mirror', 'rotate'])
                        
                        if aug_type == 'flip':
                            aug_img = ImageOps.flip(img)
                        elif aug_type == 'mirror':
                            aug_img = ImageOps.mirror(img)
                        else:
                            angle = random.choice([90, 180, 270])
                            aug_img = img.rotate(angle)
                        
                        # Save new image
                        new_name = f"aug_{generated_count}_{img_name}"
                        new_path = os.path.join(class_path, new_name)
                        
                        # Convert to RGB if needed (e.g. if RGBA) before saving as JPEG
                        if aug_img.mode in ('RGBA', 'P'):
                            aug_img = aug_img.convert('RGB')
                            
                        aug_img.save(new_path)
                        generated_count += 1
                        
                except Exception as e:
                    print(f"    Error processing {img_name}: {e}")

    print("\nBalancing complete!")

if __name__ == "__main__":
    data_dir = r"d:\BKK\bkk_crop_disease_detect_ml\dataset\plantCity\train"
    balance_dataset(data_dir)
