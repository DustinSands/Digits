# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:15:36 2020

@author: Racehorse
"""


import keras as K
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import os

# Where you want figures to be saved (if they are to be saved)
fig_dir = r'C:\Users\Racehorse\Google Drive\Programming\ML\figs'

# Target area
low = 1
high = 500

# For automated parameter search
search_parameters = {'universal':[[], [10], [10,10]], 
                     'ones':[[10]*4, [10]*8, [10]*12], 
                     'tens':[[10]*2, [10]*4, [10]*6],
                     'hundreds':[[], [10], [10,10]],
                     'epochs':[1, 10, 100],
                     }

def data_gen(batch_size):
  """ A generator that generates the feature int input along with
  the correct ones, tens, and hundreds digits targets.  Takes batch_size 
  as input.
  
  Returns a tuple of (feature, [targets]) where target is a one hot 
  encoding of [hundreds, tens, ones].
  """
  global low
  global high
  while True:
    int_in = []
    tens_out = []
    ones_out = []
    hundreds_out = []
    for batch_index in range(batch_size):
      number = random.randrange(low, high)
      int_in.append(number)
      hundreds_out.append(math.floor(number/100))
      tens_out.append(math.floor(number%100/10))
      ones_out.append(number%10)
    int_in = np.array(int_in)
    hundreds_out = tf.one_hot(hundreds_out, 10)
    tens_out = tf.one_hot(tens_out, 10)
    ones_out = tf.one_hot(ones_out, 10)
    yield (int_in, [hundreds_out, tens_out, ones_out])

def gen_data(start, stop, digit = None):
  """ Creates a training set with one data point at each integer in the range
  [start, stop].  The batch size is therefor start-stop+1.
  
  A digit of "None" creates the target set for all digits.
  1 is hundreds
  2 is tens
  3 is ones
  """
  int_in = []
  tens_out = []
  ones_out = []
  hundreds_out = []
  for index in range(stop-start):
    number = index
    int_in.append(number)
    hundreds_out.append(math.floor(number/100))
    tens_out.append(math.floor(number%100/10))
    ones_out.append(number%10)
  int_in = np.array(int_in)
  hundreds_out = tf.one_hot(hundreds_out, 10)
  tens_out = tf.one_hot(tens_out, 10)
  ones_out = tf.one_hot(ones_out, 10)
  if digit == 1:
    data_out = hundreds_out
  elif digit == 2:
    data_out = tens_out
  elif digit == 3:
    data_out = ones_out
  else: data_out = [hundreds_out, tens_out, ones_out]
  return (int_in, data_out)

def build_model(ones, tens, hundreds, universal):
  """ Builds a model with the number of hidden layers specified by ones, tens,
  hundreds, and universal (integers).  Ones, tens, and hundreds are completely
  separate.
  
  Returns a compiled model using Adadelta.
  """
  data_in = K.Input(shape=(1,))
  universal_layers = [data_in]
  for layer in universal:
    universal_layers.append(K.layers.Dense(layer, activation = 'relu')(universal_layers[-1]))
  tens_layers = [universal_layers[-1]]
  ones_layers = [universal_layers[-1]]
  hundreds_layers = [universal_layers[-1]]
  for layer in hundreds:
    hundreds_layers.append(K.layers.Dense(layer, activation = 'relu')(hundreds_layers[-1]))
  for layer in tens:
    tens_layers.append(K.layers.Dense(layer, activation = 'relu')(tens_layers[-1]))
  for layer in ones:
    ones_layers.append(K.layers.Dense(layer, activation = 'relu')(ones_layers[-1]))

  tens_out = K.layers.Dense(10, name='tens', activation = K.activations.softmax)(tens_layers[-1])
  hundreds_out = K.layers.Dense(10, name='hundreds', activation = K.activations.softmax)(hundreds_layers[-1])
  ones_out = K.layers.Dense(10, name ='ones', activation = K.activations.softmax)(ones_layers[-1])
  model = K.Model(data_in,[hundreds_out, tens_out, ones_out])
  model.compile(K.optimizers.Adadelta(),loss = K.losses.CategoricalCrossentropy())
  return model

def build_relational_model(ones, tens, hundreds):
  """ Builds a model with the number of hidden layers specified by ones, tens,
  hundreds, and universal (integers).  Uses the softmax output of of each digit
  above as input to next digit.
  
  i.e. the softmax of hundreds feeds into tens, along with the input again.
  
  Returns the compiled models using Adadelta, starting with full, ones, tens, 
  hundreds, etc.
  
  Separate models can be used to train individidually and share weights! (full 
  model uses the weights of ones, tens, and hundreds).
  """
  data_in = K.Input(shape=(1,))
  hundreds_layers = [data_in]
  for layer in hundreds:
    hundreds_layers.append(K.layers.Dense(layer, activation = 'relu')(hundreds_layers[-1]))
  
  hundreds_out = K.layers.Dense(10, name='hundreds', activation = K.activations.softmax)(hundreds_layers[-1])
  tens_layers = [K.layers.concatenate([data_in, hundreds_out])]
  for layer in tens:
    tens_layers.append(K.layers.Dense(layer, activation = 'relu')(tens_layers[-1]))
  tens_out = K.layers.Dense(10, name='tens', activation = K.activations.softmax)(tens_layers[-1])
  ones_layers = [K.layers.concatenate([data_in, tens_out])]
  for layer in ones:
    ones_layers.append(K.layers.Dense(layer, activation = 'relu')(ones_layers[-1]))
  ones_out = K.layers.Dense(10, name='ones', activation = K.activations.softmax)(ones_layers[-1])
  ones = K.Model(data_in, ones_out)
  tens = K.Model(data_in, tens_out)
  hundreds = K.Model(data_in, hundreds_out)
  full_model = K.Model(data_in,[hundreds_out, tens_out, ones_out])
  for model in [ones, tens, hundreds, full_model]:
    model.compile(K.optimizers.Adadelta(),loss = K.losses.CategoricalCrossentropy())
  return full_model, ones, tens, hundreds

def train_model(model, epochs, digit):
  # Digit is which digit output you're looking at
  # 0 is all, 1 is hundreds, 2 is tens, 3 is ones
  loss = model.fit(*gen_data(1, 1000, digit), steps_per_epoch = 1000, epochs = epochs)
  plt.plot(loss.history['loss'])

def my_loss_fn(y_true, y_pred):     
  # A failed initial attempt at a loss function.
  # Doesn't work because there is no slope of an argmax
  return tf.math.abs(tf.math.add(tf.cast(tf.math.argmax(y_pred), tf.dtypes.float32), -1.*y_true))
    
def plot_fineres(model, low_bound, high_bound, digit, title = None, filename = None):
  """Plots the output from low bound to high bound for the given digit.
  "fineres" indicates that it is plotting fine points (e.g. 33.5) even though
  the model was only trained on integers.
  
  Optionally takes Title, filename to save to disk in figure directory.
  """
  x_axis = np.arange(low_bound, high_bound, (high_bound-low_bound)/1000)
  predictions = model.predict(x_axis)
  fig, ax1 = plt.subplots()
  ax1.set_xlabel('Input')
  ax1.set_ylabel('Probability', color = 'red')
  # Each index represents one of the possible outputs (base 10)
  for index in range(10):
    ax1.plot(x_axis, predictions[digit][:,index], label = str(index))
  fig.dpi = 400
  ax1.set_title(title)
  plt.show()
  if filename != None:
    path = os.path.join(fig_dir, filename)
    fig.savefig(path)
  plt.close()
    
def plot_all_digits(passed_model = None):
  #Plots all 3 digits with one command.
  if passed_model == None:
    passed_model = model
  for n in range(3):
    plot_fineres(passed_model, low, high, n)
         
def param_search():
  """Note: this parameter search was used for the non-relational model, where
  the ones, tens, and hundreds layers did not affect each other.  This should 
  be revised to do a parameter search on the relational model.
  """
  ID = 0
  for uni in search_parameters['universal']:
    for index in range(len(search_parameters['ones'])):
      ones = search_parameters['ones'][index]
      tens = search_parameters['tens'][index]
      hundreds = search_parameters['hundreds'][index]
      model = build_model(ones, tens, hundreds, uni)
      epoch_n = 0
      for epochs in search_parameters['epochs']:
        train_model(model,epochs-epoch_n)
        epoch_n = epochs
        for digit, name in zip(range(3), ('Ones', 'Tens', 'Hundreds')):
          filename = f'ON-{ID}-{epochs}-{name}'
          title = f'{name}, Epoch {epoch_n}'
          plot_fineres(model, 1, 500, digit, title, filename)
      ID += 1
            


# Plots the model structure using keras
plot_model = lambda model: tf.keras.utils.plot_model(
    model,
    to_file="diagram.png",
    show_shapes=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=200,
)

if __name__ == '__main__':
  #Example parrallel model with two universal, one hundreds, five tens,
  # and 10 ones hidden layers.
  model = build_model([20]*10, [10]*5, [10], [20, 20])
  train_model(model, 10, 0)
  plot_all_digits()
  plot_model(model)
  
  # Example relational model, including training for full and ones
  full_model, ones_model, tens_model, hundreds_model = \
    build_relational_model([10]*2, [10]*2, [10]*2)
  train_model(full_model, 10, 0)
  plot_all_digits(full_model)
  plot_model(full_model)
  
  train_model(ones_model, 10, 3)
  plot_all_digits(full_model)