"""
  Fn for chating
"""

# import:
import keras
import data_utils

import json, os, sys
import numpy as np
import pandas as pd

from datetime import datetime
from keras.models import model_from_json

# Read vdict
vdict = data_utils.read_vocab_dict()

# Define a Fn to seperate the sents to diff. buckets.

# Read the model
config_modelfile = './model'
config_modelarch = 'model.json'
config_modelweig = 'model_weight.hdf5'

def read_model():
  jsfile = open(os.path.join(config_modelfile, config_modelarch), 'r')
  model_json = jsfile.read()
  jsfile.close()

  model = model_from_json(model_json)
  model.load_weights(os.path.join(config_modelfile, config_modelweig))
  
  return model

def chat(sent, model, vdict):
  ## prediction
  tksent = data_utils.sent_to_tokenizer(sent, vdict)
  tksent = list(reversed(tksent))
  tksent = np.array([int(vdict['_PAD'])]*(15-len(tksent)) + tksent) # 15 should be variable
  tksent_matrix = data_utils.seq_to_matrix(np.array([tksent]))
  pred_matrix = model.predict_on_batch(tksent_matrix) # predict in a single case

  ## Convert to sent
  pred_seq = data_utils.matrix_to_seq(pred_matrix)
  pred_sent = data_utils.tokenizer_to_sent(pred_seq[0], vdict)

  return pred_sent
