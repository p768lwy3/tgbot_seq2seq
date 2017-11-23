"""
  Testing to use word to vector model and 
  recurrent LSTM neural network to generate chinese sentence.

  RMK: 
    Should I use embedding layer instead of numbering words by myself?
"""

# Import:
# common library:
import json, os
import numpy as np
import pandas as pd

# keras
import keras
from keras.layers import Dense, Dropout, Embedding, LSTM, RepeatVector, TimeDistributed
from keras.models import Sequential
from keras.optimizers import SGD

# my lib
import data_utils

# input_length=X_train[1] <- length of sentence
# Embedding variable = size of vocab
# trainX and trainy are matrix shape with (num_example, time steps=sentence length, vocabsize)
def create_sequentail_model(input_length, max_output_length, vocab_size=2005,
  num_hidden_units=256, num_recurrent_units=2):
  # read-in by LSTM/TimeDistributed/Conv1D?
  model = Sequential()
  # model.add(Embedding(vocab_size, num_hidden_units, input_length=input_size))
  # model.add(Dropout(0.1))
  model.add(LSTM(num_hidden_units, input_length=(input_length, vocab_size),
    activation='relu', recurrent_dropout=0.1))
  model.add(RepeatVector(max_output_length))
  for curr_unit in range(num_recurrent_units):
    model.add(LSTM(num_hidden_units, activation='relu',
      recurrent_dropout=0.1, return_sequences=True))
  model.add(TimeDistributed(Dense(num_hidden_units, activation='relu')))
  model.add(Dropout(0.1))
  model.add(TimeDistributed(Dense(num_hidden_units, activation='relu')))
  model.add(Dropout(0.1))
  model.add(TimeDistributed(Dense(vocab_size,
    activation='softmax')))

  #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
  print(model.summary())
  
  return model
  # this model should be ok, for 30 steps, accuracy is around 0.6
  # how much need the model be actually? # not ok, since 0.5 will only capture all zeros.
  # Rmk 2. using Conv1D at first with num_hidden_units=512, kernel_size=5 and  num_hidden_units=512, kernel_size=3 will only predict 0.5 

"""
from keras.models import Model
from keras.layers import Input
from keras.utils import plot_model

def create_api_model(input_length, max_output_length, num_hidden_units=16, num_recurrent_units=2, PLOT=False):
  inputs = Input(shape=(input_legth, ), name='input')
  embseq = embedding_layers(inputs)
  encoded = Bidirectional(LSTM(num_hidden_units, return_sequences=True), merge_mode='sum', name='encode_lstm)(embseq)
  decoded = RepeatVector((max_output_length, embedding_dim), name='repeater')(encoded)
  decoded = Bidirectional(LSTM(num_hidden_units), merge_mode='sum', name='decode_lstm)(encoded)
  autoencoder = Model(inputs, decoded)
  autoencoder.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

  if PLOT == True:
    dtime = datetime.now()
    dstring = ''.join([dtime.year, dtime.month, dtime.day, dtime.hour, dtime.minute, dtime.second])
    plot_model(autoencoder, to_file='./pic/multilayer_perceptron_graph_{0}.png'.format(dstring))

  print(autoencoder.summary())
  return autoencoder
"""

dstring = datetime.now()
config_modelfile = './model/'
config_modelfielname = 'model.json'
config_modelweight = 'model_weight.hdf5'

def save_model(model):
  jstring = model.to_json()
  open(os.path.join(config_modelfile, config_modelfielname), 'w').write(jstring)
  model.save_weights(os.path.join(config_modelfile, config_modelweight))
  return

def main()
  
  model = create_model(...)
  print('    Model is Created......')
  print('#'*96)
  
  print('  Read the dataset......')
  enc, dec = data_utils.read_training_set() # type of dataset is list, elements are array

  # build training set and cross valid test set for the learninig

  print('#'*96)

  print('  Training the model......')
  # X_train and y_train shape should be (sample_num, sent_len, vocab_num).
　　model.fit(X_train, y_train, batch_size=512, epochs=100, validation_split=0.3)
  print('#'*96)
  
  print('  Saving the model......')
  print('#'*96)

  print('  Testing the model......')
  print('#'*96)
  
  print('  ??????')
  print('#'*96)
