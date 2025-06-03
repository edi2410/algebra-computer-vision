import os
from datetime import datetime
import matplotlib.pyplot as plt
from keras import mixed_precision, layers, optimizers, losses, metrics, models, applications
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.utils import image_dataset_from_directory
from tensorflow.data import AUTOTUNE
import numpy as np
from collections import Counter
import tensorflow as tf

# -------------------------------
# 1. Mixed precision policy (if GPU available)
# -------------------------------
try:
    import tensorflow as tf
    if tf.config.list_physical_devices('GPU'):
        mixed_precision.set_global_policy('mixed_float16')
        print("--INFO-- Using mixed precision")
    else:
        print("--INFO-- GPU not found, running on float32.")
except Exception as e:
    print(e)
    
# -------------------------------
# 2. Config
# -------------------------------

TRAIN_DIR = 'data2/train/'  # Putanja do foldera sa slikama
VAL_DIR = 'data2/test/'    # Putanja do foldera sa slikama
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SPLIT = 0.15      # 15% za validaciju kao što je common
SEED = 42

# -------------------------------
# 3. Učitavanje datasetova
# -------------------------------
train_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=SPLIT,
    subset='training',
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    color_mode="rgb"
)

val_dataset = image_dataset_from_directory(
    VAL_DIR,
    validation_split=SPLIT,
    subset='validation',
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    color_mode="rgb"
)

class_names = train_dataset.class_names
NUM_CLASSES = len(class_names)
print(f"✅ Detected {NUM_CLASSES} classes: {class_names}")

# -------------------------------
# 4. Uravnoteženje klasa (bolje, brže)
# -------------------------------
def get_class_counts(ds):
    labels = []
    for _, label in ds.unbatch():
        labels.append(int(label))
    return dict(Counter(labels))

label_counts = get_class_counts(train_dataset)
total_samples = sum(label_counts.values())
class_weights = {i: total_samples/(NUM_CLASSES*label_counts[i]) for i in range(NUM_CLASSES)}
print("Class weights:", class_weights)

# -------------------------------
# 5. Data Augmentation & Normalization
# -------------------------------

normalization = layers.Rescaling(1./127.5, offset=-1)

def preprocess_train(x, y):
    x = normalization(x)
    return x, y

def preprocess_val(x, y):
    return normalization(x), y

train_dataset = train_dataset.map(preprocess_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_dataset   = val_dataset.map(preprocess_val,   num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# -------------------------------
# 6. Model: EfficientNet B0 + klasifikator
# -------------------------------
base_model = applications.EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=IMG_SIZE+(3,)
)
base_model.trainable = True

inputs = layers.Input(shape=IMG_SIZE+(3,))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax', dtype='float32')(x)
model = models.Model(inputs, outputs)


# -------------------------------
# 7. Kompajliranje modela
# -------------------------------
model.compile(
    optimizer=optimizers.AdamW(learning_rate=0.0001),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=[metrics.SparseCategoricalAccuracy()]
)

# -------------------------------
# 8. Callbacks
# -------------------------------
log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
callbacks = [
    ModelCheckpoint('best_model.keras', save_best_only=True, verbose=1),
    EarlyStopping(patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.2, patience=3, verbose=1),
    TensorBoard(log_dir=log_dir)
]

# -------------------------------
# 9. Treniranje modela
# -------------------------------
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2
)

# -------------------------------
# 10. Fine-tuning
# -------------------------------
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.00001),
    loss=losses.SparseCategoricalCrossentropy(),
    metrics=[metrics.SparseCategoricalAccuracy()]
)

history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2
)
model.save('bone_fracture_classifier.keras')

print(" Gotovo, model spremljen kao 'bone_fracture_classifier.keras'")