import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

BASE_DIR = r'C:\Users\A HAJIBU\OneDrive\Desktop\online fake logo detection'
train_data_dir = os.path.join(BASE_DIR, 'Dataset', 'train')
test_data_dir  = os.path.join(BASE_DIR, 'Dataset', 'test')

for path in [train_data_dir, test_data_dir]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Folder not found: {path}")
    else:
        print(f"✅ Found: {path}")

img_width, img_height = 224, 224
batch_size = 32
epochs = 10

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

print(f"Training samples: {train_generator.samples}")
print(f"Testing samples : {test_generator.samples}")

if train_generator.samples == 0:
    raise ValueError("❌ No training images found! Check Dataset/train folder.")

base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    steps_per_epoch=max(1, train_generator.samples // batch_size),
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=max(1, test_generator.samples // batch_size)
)

save_path = os.path.join(BASE_DIR, 'logo_classification_model.h5')
model.save(save_path)
print(f"✅ Model saved to: {save_path}")