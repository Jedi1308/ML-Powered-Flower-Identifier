
import os
import numpy as np
import glob
import shutil

import tensorflow as tf

import matplotlib.pyplot as plt

### TODO: Import TensorFlow and Keras Layers

#import packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Loading
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos')

"""The dataset we downloaded contains images of 5 types of flowers:

1. Rose
2. Daisy
3. Dandelion
4. Sunflowers
5. Tulips

Create the labels for these 5 classes:
"""

classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

for cl in classes:
  img_path = os.path.join(base_dir, cl)
  images = glob.glob(img_path + '/*.jpg')
  print("{}: {} Images".format(cl, len(images)))
  train, val = images[:round(len(images)*0.8)], images[round(len(images)*0.8):]

  for t in train:
    if not os.path.exists(os.path.join(base_dir, 'train', cl)):
      os.makedirs(os.path.join(base_dir, 'train', cl))
    shutil.move(t, os.path.join(base_dir, 'train', cl))

  for v in val:
    if not os.path.exists(os.path.join(base_dir, 'val', cl)):
      os.makedirs(os.path.join(base_dir, 'val', cl))
    shutil.move(v, os.path.join(base_dir, 'val', cl))

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

# Data Augmentation

batch_size = 100
IMG_SHAPE = 150

### TODO: Apply Random Horizontal Flip

### TODO: Apply Random Rotation

### TODO: Apply Random Zoom

### TODO: Put It All Together

image_gen = ImageDataGenerator(rescale=1./255,
                               rotation_range=45,
                               width_shift_range=0.15,
                               height_shift_range=0.15,
                               zoom_range=0.5,
                               horizontal_flip=True)

train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(IMG_SHAPE,IMG_SHAPE),
                                               class_mode='sparse')


augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

image_gen_val = ImageDataGenerator(rescale=1./255)
val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                               directory=val_dir,
                                               target_size=(IMG_SHAPE,IMG_SHAPE),
                                               class_mode='sparse')

# TODO: Create the CNN

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3, 3), activation='relu',input_shape=(150, 150, 3)),
                                   tf.keras.layers.MaxPooling2D(2, 2),

                                   tf.keras.layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)),
                                   tf.keras.layers.MaxPooling2D(2, 2),

                                   tf.keras.layers.Conv2D(64, (3, 3), activation='relu',input_shape=(150, 150, 3)),
                                   tf.keras.layers.MaxPooling2D(2, 2),

                                   tf.keras.layers.Dropout(0.2),
                                   tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(512, activation='relu'),
                                    # should have added another dropout layer here
                                   tf.keras.layers.Dense(5)])

# TODO: Compile the Model

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# TODO: Train the Model

epochs = 80

history = model.fit_generator(train_data_gen,
                              steps_per_epoch=int(np.ceil(2935 / float(batch_size))),
                              epochs=epochs,
                              validation_data=val_data_gen,
                              validation_steps=int(np.ceil(735 / float(batch_size))))

# TODO: Plot Training and Validation Graphs.


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


