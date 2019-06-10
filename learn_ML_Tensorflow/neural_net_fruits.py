# Data for this code is derived from https://www.kaggle.com/moltean/fruits
# The folder data/ conatains only training images for three fruits to limit required effort. 
from tensorflow import keras
import numpy as np


my_generator = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.3,
    shear_range=0.3
)

generated_data = my_generator.flow_from_directory('data', target_size=(100,100))
# At this point the training data is ready.

# Explanation of layer design: https://code.tutsplus.com/courses/learn-machine-learning-with-google-tensorflow/lessons/creating-a-convolutional-network-for-image-classification
my_network = keras.Sequential()

my_network.add(keras.layers.Conv2D(32,3,3,
    input_shape=(100,100,3),
    activation='relu'))

my_network.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

my_network.add(keras.layers.Conv2D(64,3,3, activation='relu'))

my_network.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

my_network.add(keras.layers.Flatten())

my_network.add(keras.layers.Dense(128, activation='relu'))

my_network.add(keras.layers.Dropout(0.5))

my_network.add(keras.layers.Dense(3,activation='softmax'))

my_network.compile(
    optimizer=keras.optimizers.RMSprop(lr=0.0004),
    loss='categorical_crossentropy'
)

## Training and saving network. Run once if you don't already have fruits.h5 file.
# my_network.fit_generator(generated_data, epochs=5)
# my_network.save_weights('fruits.h5')


my_network.load_weights('fruits.h5')

# Test/validation data
# test_img = keras.preprocessing.image.load_img('data/banana_1_297_100.jpg') # recognized as banana [~1,~0,~0] 🙂
# Random images from Google, resized to 100 * 100 pixels
# test_img = keras.preprocessing.image.load_img('data/banana_2.jpg') # recognized as banana [1,0,0] 🙂
# test_img = keras.preprocessing.image.load_img('data/banana_2.jpg') # recognized as banana [1,0,0] 🙂
# test_img = keras.preprocessing.image.load_img('data/cucumber.jpg') # recognized as banana [1,0,~0] 🤔
# test_img = keras.preprocessing.image.load_img('data/onion.jpg') # recognized as banana [1,~0,~0] 😢
# test_img = keras.preprocessing.image.load_img('data/orange-with-leaf.jpg') # recognized as banana [~1,~0,~0] 😫
# test_img = keras.preprocessing.image.load_img('data/pineapple-with-green.jpg') # recognized as pineapple [~0,~0,~1] 🙂
# test_img = keras.preprocessing.image.load_img('data/pineapple-green-cut-away.jpg') # recognized as pineapple [~0,~0,~1] 🙂
test_img = keras.preprocessing.image.load_img('data/orange-without-strunk.jpg') # recognized as orange [~0,~1,~0] 🙂

# Training data only contained banana, orange and pineapple – obviously not enough data.


test_img_arr = keras.preprocessing.image.img_to_array(
    test_img,
    data_format='channels_last' # color channel information is last in our [100,100,3] image array format
)
# wrap array into another array [[...]]
test_img_arr = np.array([test_img_arr])
# print indices so we know which of the columns corresponds to 'banana'
print(generated_data.class_indices)

print(my_network.predict(test_img_arr))
