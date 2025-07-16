import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

csv_path = '../data/cassava-leaf-disease-classification/train.csv'
image_path = '../data/cassava-leaf-disease-classification/train_images'
output_path = '../data/cassava-leaf-disease-classification/departed_images'
test_rate = 0.2

df = pd.read_csv(csv_path)
classes = df['label'].unique()

os.makedirs(output_path, exist_ok=True)
train_dir = os.path.join(output_path, 'train')
val_dir = os.path.join(output_path, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)



for c in classes:
    os.makedirs(os.path.join(train_dir, str(c)), exist_ok=True)
    os.makedirs(os.path.join(val_dir, str(c)), exist_ok=True)

for c in classes:
    c_df = df[df['label'] == c]
    tra_file, val_file = train_test_split(
        c_df['image_id'].tolist(), test_size=test_rate, random_state=114
    )

    for f in tra_file:
        src_path = os.path.join(image_path, f)
        dst_path = os.path.join(train_dir, str(c), f)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

    for f in val_file:
        src_path = os.path.join(image_path, f)
        dst_path = os.path.join(val_dir, str(c), f)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)

print('Done')
