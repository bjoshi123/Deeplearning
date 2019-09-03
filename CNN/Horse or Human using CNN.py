#https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip
#https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip
#download training and validation data from the above link and place the zip in same dir with this file

import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#code to unzip the zip file and store
training data
localzip = 'horse-or-human.zip'
zip_ref = zipfile.ZipFile(localzip,'r')
zip_ref.extractall('horse-or-human')
zip_ref.close()

validation data
localzip = 'validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(localzip,'r')
zip_ref.extractall('validation-horse-or-human')
zip_ref.close()

train_horse_dir = os.path.join('horse-or-human/horses')
train_human_dir = os.path.join('horse-or-human/humans')

train_horses_name = os.listdir(train_horse_dir)
print(train_horses_name[:10])

train_human_name = os.listdir(train_human_dir)
print(train_human_name[:10])


#length of training
print(len(os.listdir(train_horse_dir))) #
print(len(os.listdir(train_human_dir))) #

#plot some image
# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname)
                for fname in train_horses_name[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname)
                for fname in train_human_name[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix+next_human_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()


#defining model structure
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3), activation='relu', input_shape=(300,300,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#model summary
model.summary()

#model compilation
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['acc'])


# data preprocessing
train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    'horse-or-human',
    target_size=(300,300),
    batch_size=128,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    'validation-horse-or-human',
    target_size=(300,300),
    batch_size=32,
    class_mode='binary'
)

# model training
history = model.fit_generator(
    train_generator,
    epochs=15,
    steps_per_epoch=8,
    validation_data=test_generator,
    validation_steps=8,
    verbose=1
)
