import os
import matplotlib.pyplot as plt

data_dir = r"d:\BKK\bkk_crop_disease_detect_ml\dataset\plantCity\train"
class_counts = {}

if os.path.exists(data_dir):
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            class_counts[class_name] = count
            print(f"{class_name}: {count}")
else:
    print(f"Directory not found: {data_dir}")

# Plotting (optional, but good for visualization if run locally, here just printing is enough)
# plt.bar(class_counts.keys(), class_counts.values())
# plt.show()
