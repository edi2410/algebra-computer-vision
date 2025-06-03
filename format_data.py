### Premještanje slika u nove foldere kako bi ga prilagodili za flow_from_directory
import os
import shutil


ROOT_DIR = 'data2'
NEW_TRAIN_DIR = os.path.join(ROOT_DIR, 'train')
NEW_VALID_DIR = os.path.join(ROOT_DIR, 'test')

os.makedirs(NEW_TRAIN_DIR, exist_ok=True)
os.makedirs(NEW_VALID_DIR, exist_ok=True)


# za svaki tip prijeloma
for fracture_type in os.listdir(ROOT_DIR):
    type_path = os.path.join(ROOT_DIR, fracture_type)
    if not os.path.isdir(type_path):
        continue
    
    # Premjesti train slike
    src_train = os.path.join(type_path, 'train')
    dst_train = os.path.join(NEW_TRAIN_DIR, fracture_type)
    if os.path.exists(src_train):
        os.makedirs(dst_train, exist_ok=True)
        for fname in os.listdir(src_train):
            shutil.move(os.path.join(src_train, fname), os.path.join(dst_train, fname))
    # Premjesti validation slike
    src_val = os.path.join(type_path, 'test')
    dst_val = os.path.join(NEW_VALID_DIR, fracture_type)
    if os.path.exists(src_val):
        os.makedirs(dst_val, exist_ok=True)
        for fname in os.listdir(src_val):
            shutil.move(os.path.join(src_val, fname), os.path.join(dst_val, fname))




