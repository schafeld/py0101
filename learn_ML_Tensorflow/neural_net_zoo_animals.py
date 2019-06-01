# Based on: https://code.tutsplus.com/courses/learn-machine-learning-with-google-tensorflow/lessons/using-real-world-data-with-a-neural-network

from tensorflow import keras
import numpy as np
import pandas as pd

training_data = pd.read_csv('./zoo-animal-classification/zoo.csv')

# 7 types of animals require 7 input neurons.
# But we're mapping 1-7 directly to nimal class and leave index 0 unused, thus 8 neurons.
output_data = np.array(
    training_data['class_type'].values
)

# 'one-hot tensors, arrays with only one entry True/1
# create arrays filled with 8 zeros (corresponding to 8 neurons)
outputs = np.zeros((output_data.size, 8))
outputs[np.arange(output_data.size), output_data] = 1

inputs = np.array(
    training_data[training_data.columns.values[1:17]]
)

animal_classification_network = keras.Sequential()

animal_classification_network.add(keras.layers.Dense(
    30,
    input_dim=16, # see training_data.columns
    activation='relu'
))

animal_classification_network.add(keras.layers.Dense(
    20,
    activation='relu'
))

animal_classification_network.add(keras.layers.Dense(
    8,
    activation='softmax' # because there's only 1 relevant output item
))

animal_classification_network.compile(
    optimizer='sgd',
    loss='mse'
)

# Train the neural net. Run once to generate training data file.
# animal_classification_network.fit(inputs, outputs, epochs=7500)
# animal_classification_network.save_weights('animals.h5')

animal_classification_network.load_weights('animals.h5')

# First label is 'None' as index 0 isn't used, see above.
labels = [None, 'mammal (Säugetier)', 'bird (Vogel)', 'reptile (Reptil)', 'fish (Fisch)', 'amphibian (Amphibie)', 'bug (Käfer)', 'invertebrate (Wirbellos)' ]

# Pick a tiger as test animal, described by its characteristics.
# (animal_name,) hair, feathers, eggs, milk, airborne, aquatic, predator, toothed, backbone, breathes,
# venomous, fins, legs, tail, domestic, catsize (, class_type), i.e. 16 classification items
#
# Look out: This has to be an array within an array, [[]]! Otherwise you get a strange error like this:
# ValueError: Error when checking input: expected dense_input to have shape (16,) but got array with shape (1,) 
#  Tiger, Tiger, silly cat
#  Don't throw errors
#  That'll drive me mad!
test_data = np.array([[1,0,0,1,0,0,1,1,1,1,0,0,4,1,0,1]])

predictions = animal_classification_network.predict(test_data)

highest_neuron = np.argmax(predictions[0])
print('Animal is %s' % labels[highest_neuron])

keras.backend.clear_session()
