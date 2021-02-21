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

sample_df = pd.read_csv(base_dir + '/sample_submission.csv')


dog_list = ['beagle', 'chihuahua', 'doberman', 'french_bulldog', 'golden_retriever', 'malamute', 'pug', 'saint_bernard', 'scottish_deerhound', 'tibetan_mastiff']

dog_set = set(dog_list)
num_classes = 10
dog_dataframe = dog_labels_df.loc[dog_labels_df['breed'].isin(dog_list)]
dog_dataframe = dog_dataframe.reset_index()
print(dog_dataframe.head())
print(len(dog_dataframe))

# Creating a list of distinct dog breeds
dog_breeds = sorted(list(set(dog_dataframe['breed'])))


# Associating each breed with an integer
class_to_num = dict(zip(dog_breeds, range(num_classes)))

def img_to_array(data_dir, labels_df, image_size=(300, 300, 3)):

    filenames = labels_df['id']
    labels = labels_df['breed']

    # Number of images
    data_size = len(filenames)
    print("Data size: ", data_size)

    # Initializing Image Numpy array
    X = np.zeros([data_size, image_size[0], image_size[1], image_size[2]], dtype=np.uint8)
    Y = np.zeros([data_size, 1], dtype=np.uint8)

    for i in range(data_size):
        file = filenames[i]
        #print("This is the first file: ", file)
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
    #print(Y[0])

    print("Output data shape: ", X.shape)
    print("Output labels shape: ", Y.shape)
    return X, Y

# Shape of our Input images
img_size = (224, 224, 3)
train_images, train_labels = img_to_array(train_dir, dog_dataframe, img_size)


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
model.add(layers.Dense(units=128, activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=64, activation='relu'))
model.add(layers.Dense(units=10, activation='softmax')) 

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

#Train simple DNN.
history = model.fit(train_images, train_labels,
            batch_size=32,
            epochs=60,
            validation_split=0.2
             )

# This function is to convert Test images to Numpy arrays
def img_to_array2(data_dir, labels_df, image_size=(224, 224, 3)):

    filenames = labels_df['id']

    # Number of images
    data_size = len(filenames)

    # Initializing Image Numpy array
    X = np.zeros([data_size, image_size[0], image_size[1], image_size[2]], dtype=np.uint8)

    for i in range(data_size):
        file = filenames[i]
        img_path = os.path.join(data_dir, file + '.jpg')

        # converting Images to 4-D Numpy array
        img_pixels = load_img(img_path, target_size=image_size)
        X[i] = img_pixels

    print("Output data shape: ", X.shape)
    return X

del train_images

test_images= img_to_array2(test_dir, sample_df, img_size)

test_images = test_images / 255.0

# Evaluate against the test set.
print("\n Evaluate the new model against the test set:")
y_pred = model.predict(train_images, batch_size=32)

for b in dog_list:
    sample_df[b] = y_pred[:,class_to_num[b]]
sample_df.to_csv('pred.csv', index=None)

