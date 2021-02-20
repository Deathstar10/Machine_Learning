import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img 
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers.core import Dropout
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D, Lambda, Dropout, InputLayer, Input



base_dir = '../input/dog-breed-identification'

# Directory with our training Dog pictures
train_dir = os.path.join(base_dir, 'train')

# Directory with our testing Dog pictures
test_dir = os.path.join(base_dir, 'test')

# Storing Dog labels in a Dataframe
dog_labels_df = pd.read_csv(base_dir + '/labels.csv')


# Creating a list of distinct dog breeds
dog_breeds = sorted(list(set(dog_labels_df['breed'])))
num_classes = len(dog_breeds)
print(dog_breeds[:10])

# Associating each breed with an integer
class_to_num = dict(zip(dog_breeds, range(num_classes)))

def img_to_array(data_dir, labels_df, image_size=(300, 300, 3)):

    filenames = labels_df['id']
    labels = labels_df['breed']

    # Number of images
    data_size = len(filenames)

    # Initializing Image Numpy array
    X = np.zeros([data_size, image_size[0], image_size[1], image_size[2]], dtype=np.uint8)
    Y = np.zeros([data_size, 1], dtype=np.uint8)

    for i in range(data_size):
        file = filenames[i]
        img_path = os.path.join(data_dir, file + '.jpg')

        # converting Images to 4-D Numpy array
        img_pixels = load_img(img_path, target_size=image_size)
        X[i] = img_pixels

        # Converting Labels to 2-D numpy array
        image_breed = labels[i]
        Y[i] = class_to_num[image_breed]

    # One-hot encoding
    Y = to_categorical(Y)

    #shuffle the images and label arrays
    ind = np.random.permutation(data_size)
    X = X[ind]
    Y = Y[ind]
    print(Y[0])

    print("Output data shape: ", X.shape)
    print("Output labels shape: ", Y.shape)
    return X, Y

img_size = (224, 224, 3)
train_images, train_labels = img_to_array(train_dir, dog_labels_df, img_size)

# Scaling values between 0-1
train_images = train_images / 255.0

from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential,Model,load_model

img_input = layers.Input(shape=(224, 224, 3))

base_model = ResNet50(weights='imagenet')


#Prepare DNN model
model = tf.keras.models.Sequential()

model.add(base_model)

model.add(layers.Flatten(input_shape=(224, 224)))
model.add(layers.Dense(units=32, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=120, activation='softmax')) 

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

#Train simple DNN.
h = model.fit(train_images, train_labels,
            batch_size=128,
            epochs=38,
            validation_split=0.2
             )