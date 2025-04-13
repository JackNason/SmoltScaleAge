
import os
import tensorflow as tf
import pandas as pd
from pathlib import Path

# Set root directory
root_dir = Path("images")

# Class name to index
class_map = {name: idx for idx, name in enumerate(sorted(os.listdir(root_dir)))}

print(class_map)

# Collect paths and labels
data = []
for class_name in class_map:
    class_folder = root_dir / class_name
    for img_file in class_folder.glob("*.tif"):
        data.append({
            "filepath": str(img_file),
            "class": class_map[class_name]
        })

df = pd.DataFrame(data)

print(df)

coords_df = pd.read_csv('labels.csv')

# Strip filename only (no folders) for the join
df['filename'] = df['filepath'].apply(lambda x: os.path.basename(x))

# Merge classification + regression data
merged_df = pd.merge(df, coords_df, on='filename')

"""
df = pd.read_csv('labels.csv')
filenames = df['filename'].values
labels = df[['x1', 'y1', 'x2', 'y2']].values.astype('float32')

image_paths = ['images/' + fname for fname in filenames]

# Create tf.data.Dataset
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [128, 128])  # or whatever size you're using
    image = image / 255.0  # normalize
    return image, label

# tf.data pipeline
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = dataset.map(load_image).batch(32).shuffle(500)

"""