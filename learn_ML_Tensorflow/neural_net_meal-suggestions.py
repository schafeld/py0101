# Based on: https://code.tutsplus.com/courses/learn-machine-learning-with-google-tensorflow/lessons/creating-your-first-neural-network

from tensorflow import keras
import numpy as np

# 'Sequential' class -> straight sequence of layers (input, n hidden layers, output), Multi-layer Perceptron (MLP)
meals_classification_network = keras.Sequential()

# first hidden layer
meals_classification_network.add(keras.layers.Dense(
    12,
    input_dim=6,
    activation='relu'
))
#second hidden layer
meals_classification_network.add(keras.layers.Dense(
    8,
    activation='relu'
))
# output layer
meals_classification_network.add(keras.layers.Dense(
    4,
    activation='softmax' # usually used to pick one from three or more items (two -> sigmoid)
))

# Compile neural network for back-propagation
meals_classification_network.compile(
    optimizer='adam',
    loss='mse' # mean square error
)

# Fictional training data
# Inputs would be excel rows with individual observations and columns resembling specific observed values...
inputs=np.array([
    [1,0,0,1,0,0],
    [1,0,0,0,1,0],
    [1,0,0,0,0,1],
    [0,1,0,1,0,0],
    [0,1,0,0,1,0],
    [0,1,0,0,0,1],
    [0,0,1,1,0,0],
    [0,0,1,0,1,0],
    [0,0,1,0,0,1]
])
# ...outputs would be corresponding result rows with columns resembling result matrix (yes/no).
outputs=np.array([
    [0,1,0,0],
    [0,0,1,0],
    [0,1,0,0],
    [1,0,0,0],
    [0,0,1,0],
    [0,0,0,1],
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1]
])

# Train the neural network to produce an hdf5 file
# meals_classification_network.fit(inputs, outputs, epochs=5000)
# meals_classification_network.save_weights('meals.h5')

# Initialize network with weighted training data
meals_classification_network.load_weights('meals.h5')

# Expected result: Array with 4 values, second close to 1 others close to 0.
print (meals_classification_network.predict(np.array([
    [1,0,0,0,0,1]
])))

# Release system resources
keras.backend.clear_session()
