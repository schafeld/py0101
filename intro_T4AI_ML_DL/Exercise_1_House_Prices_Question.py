#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'intro_T4AI_ML_DL'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# In this exercise you'll try to build a neural network that predicts the price of a house according to a simple formula.
# 
# So, imagine if house pricing was as easy as a house costs 50k + 50k per bedroom, so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.
# 
# How would you create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k etc.
# 
# Hint: Your network might work better if you scale the house price down. You don't have to give the answer 400...it might be better to create something that predicts the number 4, and then your answer is in the 'hundreds of thousands' etc.
# 
# My example has 4 houses (xs) with prices in thousands of currency units (ys). 500 epochs get an estimated price of about 409 currrency units. Which is okay.

#%%
import tensorflow as tf
import numpy as np
from tensorflow import keras
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([1, 2, 3, 4], dtype=float)
ys = np.array([100, 150, 200, 250], dtype=float)
model.fit(xs, ys, epochs=500)
print(model.predict([7.0]))


#%%



