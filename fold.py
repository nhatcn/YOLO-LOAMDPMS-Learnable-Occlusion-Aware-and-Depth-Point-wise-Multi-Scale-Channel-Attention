import os
import shutil
import random
import yaml

# Thư mục gốc
image_folder = "Rain/Rain"
label_folder = "Rain/Rain_YOLO_darknet"

# Classes bạn muốn dùng
selected_classes = [3, 6, 8]
class_names = ['car', 'bus', 'truck']

# Thư mục output
output_folder = "Rain/folds"
os.makedirs(output_folder, exist_ok=True)

# Lấy danh sách tất cả ảnh
images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Lọc ảnh có nhãn phù hợp
filtered_images = []
for img in images:
    label_file = os.path.join(label_folder, os.path.splitext(img)[0] + ".txt")
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            lines = f.readlines()
        keep = any(int(line.split()[0]) in selected_classes for line in lines)
        if keep:
            filtered_images.append(img)

# Shuffle
random.shuffle(filtered_images)

# Chia 5 fold
k = 5
fold_size = len(filtered_images) // k
folds = [filtered_images[i*fold_size:(i+1)*fold_size] for i in range(k-1)]
folds.append(filtered_images[(k-1)*fold_size:])  # fold cuối

# Tạo fold folder + copy file
for i, fold_images in enumerate(folds):
    fold_dir = os.path.join(output_folder, f"fold{i+1}")
    images_dir = os.path.join(fold_dir, "images")
    labels_dir = os.path.join(fold_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    for img in fold_images:
        shutil.copy(os.path.join(image_folder, img), images_dir)
        label_file = os.path.join(label_folder, os.path.splitext(img)[0] + ".txt")
        new_lines = [line for line in open(label_file).readlines() if int(line.split()[0]) in selected_classes]
        with open(os.path.join(labels_dir, os.path.basename(label_file)), "w") as f:
            f.writelines(new_lines)

# Tạo data.yaml cho từng fold (train = 4 fold, val = 1 fold)
for i in range(k):
    val_fold = f"fold{i+1}"
    train_folds = [f"fold{j+1}" for j in range(k) if j != i]
    
    # Tạo list file paths train
    train_paths = []
    for tf in train_folds:
        fold_images_dir = os.path.join(output_folder, tf, "images")
        for img in os.listdir(fold_images_dir):
            train_paths.append(os.path.abspath(os.path.join(fold_images_dir, img)))
    
    # Val paths
    val_images_dir = os.path.join(output_folder, val_fold, "images")
    val_paths = [os.path.abspath(os.path.join(val_images_dir, img)) for img in os.listdir(val_images_dir)]
    
    # Tạo data.yaml
    data_yaml = {
        'train': train_paths,
        'val': val_paths,
        'nc': len(selected_classes),
        'names': {str(c): n for c, n in zip(selected_classes, class_names)}
    }
    
    yaml_path = os.path.join(output_folder, f"data_fold{i+1}.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

print("Hoàn tất chia 5 fold và tạo data.yaml chuẩn K-Fold!")
