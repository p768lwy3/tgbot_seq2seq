"""
  Tokenize and Vectorize dataset for training and testing...
  Fn need to be written:
    1. Tokenize input sentence (for both training and testing to use.)
      1.1. Tokenize training set
    2. Vectorize input sentence (for both training and testing to use.)
      2.1. Vectorize training set
    3. Padding and Bucketing/ or I should use batch_size=1???
    4. ?
  Rmk:
    write down var_len in CONFIG.py
    how to read the dataset???
    is there any way to convert vectors to words in word2vec model???????
    ...
"""

# IMPORT:
# common library:
import jieba, re, os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#from gensim.models import Word2Vec # to convert words to vectors
from sklearn.manifold import TSNE # for visualize word2vec

def tokenizer(sent):
  # return a list of tokenized words
  return (' '.join(jieba.cut(sent))).split(' ')

def tokenizer_char(sent):
  # return a list of tokenized characters
  return list(sent)

def read_data(raw_data_path = './data/smsCorpus'):
  f = open(raw_data_path, 'r')
  lines = []
  for line in f:
    line = line.replace('\n', '')
    line = tokenizer_char(line)
    lines.append(line)
  return lines

def word2vec_model(sentences, vec_size=200, 
    window=5, min_count=2, workers=4, PLOT=True):
  w2v_model = Word2Vec(sentences, size=vec_size, window=window, min_count=min_count, workers=workers)
  dtime = datetime.now()
  dstring = ''.join([str(dtime.year), str(dtime.month), str(dtime.day), 
    str(dtime.hour), str(dtime.minute), str(dtime.second)])
  mpath = './model/word2vec_model_' + dstring + '.model'
  w2v_modl.save(mpath)
  word_vectors = w2v_model.wv

  if PLOT == False:
    del w2v_model
  
  if PLOT == True:
    """ ref: https://stackoverflow.com/questions/43776572/visualise-word2vec-generated-from-gensim """
    vocab = list(word_vectors.vocab)
    X = w2v_model[vocab]
    del w2v_model

    print('  The size of word vectors vocab is %d ' % len(vocab))
    print('  Ploting word vector model with TSNE......')
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    df = pd.concat([pd.DataFrame(X_tsne), pd.Series(vocab)], axis=1)
    df.colums = ['x', 'y', 'word']
    
    ## Rmk: How to plot chinese label in matplotlib?
    figpath = './pic/word2vec_plot.png'
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df['x'], df['y'])
    for i, txt in enumerate(df['word']):
      ax.annotate(txt, (df['x'].iloc[i], df['y'].iloc[i]))
    fig.savefig(figpath)
    plt.show()
    
  return word_vectors

def sent_vectorizer(sent, word_vector, var_len=200):
  vs = np.array([]) # add this to prevent UnboundLocalError
  counter = 0
  for w in sent:
    try:
      v = word_vector[w]
    except:
      v = np.zeros(var_len)
    vs = np.vstack([vs, v]) if counter > 0 else v
    counter += 1
  return vs

def dataset_vectorizer(ds, word_vector, var_len=200):
  dataset = []
  counter = 0
  for sent in ds:
    dataset.append(sent_vectorizer(sent, word_vector, var_len=var_len))
    counter += 1 
    if counter % 1000 == 0:
      print('    Now is reading the %d -th lines......' % counter)
  return dataset

def build_vocab(ds, vpath='./data/vocab', min_count=5, max_size=2000):
  vdict = {}
  for sent in ds:
    for w in sent:
      if not w in vdict:
        vdict[w] = 1
      else:
        vdict[w] += 1

  df = pd.DataFrame(vdict, index=range(0, len(list(vdict.keys()))))

  # remove values small than min_count,
  vdict = {k: v for k, v in vdict.items() if v >= min_count}

  # sort by values,
  vdict = dict(sorted(vdict.items(), key=lambda x: x[1], reverse=True)) # Now vdict is list-type..

  # remove elements if larger than max_size,
  if len(list(vdict.keys())) > max_size:
    vdict = vdict[:max_size]
  
  # save,
  fpath = './data/vocab'
  with open(fpath, 'w') as f:
    token = ['_PAD', '_GO', '_EOS', '_UNK']
    for i in range(len(token)):
      w0 = token[i] + ' ' + str(i) + '\n'
      f.write(w0)
    for i in range(len(vdict)):
      tk = int(i + len(vdict))
      w0 = vdict[i][0] + ' ' + str(tk) + '\n'
      f.write(w0)
  f.close()

  return vpath
  
def read_vocab_dict(vpath='./data/vocab'):
  f = open(vpath, 'r')
  vdict = {}
  for l in f:
    l = l.split()
    try:
      vdict[l[0]] = l[1]
    except:
      vdict[' '] = l[0] #or pass??
  return vdict
    
def basic_tokenizer(ds, vdict, tpath='./data/tokenized'):
  f = open(tpath, 'w')
  for line in ds:
    # sent = sent_to_tokenizer(line, vdict)
    sent = [str(vdict[w]) if w in vdict.keys() else str(vdict['_UNK']) for w in line]
    sent = ' '.join(sent) + '\n'
    f.write(sent)
  f.close()
  return tpath

def sent_to_tokenizer(sent, vdict):
  return np.array([int(vdict[w]) if w in vdict.keys() else int(vdict['_UNK']) for w in sent])

def value_to_key_of_dict(v, vdict):
  # check datatype of v.
  return list(vdict.keys())[list(vdict.values()).index(str(v))]

def tokenizer_to_sent(tksent, vdict):
  wsent = []
  for v in tksent:
    try:
      # Rmk, check the datatype of v should be int(?), not float, if float, convert to int first.
      key_ = value_to_key_of_dict(int(v), vdict)
      # This should be too slow, it is better not to save vocab as dict
    except:
      pass
    #if not key_ in ['_PAD', '_UNK', '_GO', '_EOS']: # this should be del when checking
    wsent.append(key_)
  return ''.join(wsent)
  
def read_tokenized_data(tpath='./data/tokenized'):
  f = open(tpath, 'r')
  lines = []
  for l in f:
    l = l.replace('\n', '')
    lines.append([int(w) for w in l.split()])
  # rmk: should i remove the blank line?
  return lines

def build_training_set(tdata, vdict,
                       enc_path='./data/data_{0}.enc', 
                       dec_path='./data/data_{0}.dec',
                       buckets_=[(15, 20), (30, 35), (45, 50), (75, 80)]): # meanlen~=15, stdlen~=13
  buckets_dict = {}
  counter = 0
  """ Rmk: encoder = ['_PAD', ..., '_PAD','.', 'boy', 'a', 'am', 'I']
       decoder = ['_GO', 'I', 'am', 'a', 'boy', '.', '_EOS', '_PAD', ..., '_PAD']
  """
  for l in tdata:
    # ENCODER:
    enc_line = list(reversed(l)) # This should be faster and less memories need compare with l[::-1]
    # should encoder side add '_EOS'??
    
    # DECODER:
    dec_line = [int(vdict['_GO'])] + l + [int(vdict['_EOS'])]
    
    for b in buckets_:
      enc_bucket_len = b[0] 
      dec_bucket_len = b[1]
      if len(enc_line) <= enc_bucket_len and len(dec_line) <= dec_bucket_len:
        enc_line = np.array([int(vdict['_PAD'])]*(enc_bucket_len-len(enc_line)) + enc_line)
        dec_line = np.array(dec_line + [int(vdict['_PAD'])]*(dec_bucket_len-len(dec_line)))
        break
        # rmk: forgot to cut the length larger than (75, 80)...!!!
    if enc_bucket_len not in buckets_dict:
      buckets_dict[enc_bucket_len] = [(enc_line, dec_line)]
    else:
      buckets_dict[enc_bucket_len].append((enc_line, dec_line))
    counter += 1
    if counter % 1000 == 0:
      print('  Now is reading the %d -th lines...' % counter)
  
  for k in buckets_dict.keys():
    buckets_dict[k] = np.array(buckets_dict[k])
    epath = enc_path.format(k)
    dpath = dec_path.format(k)
    np.save(epath, buckets_dict[k][:,0]) # or save as text file? this will be smaller..
    np.save(dpath, buckets_dict[k][:,1])
    # here can be save better to be an single array but not a array contain array...

  return enc_path, dec_path, buckets_
  
def read_training_set(enc_path='./data/data_{0}.enc.npy', 
                    dec_path='./data/data_{0}.dec.npy',
                    buckets_=[(15, 20), (30, 35), (45, 50)]):
  enc = []
  dec = []
  for b in buckets_:
    print('  Now is loading bucket %s......' % str(b))
    bsize = b[0]
    epath = enc_path.format(bsize)
    dpath = dec_path.format(bsize)
    enc_np = np.load(epath)
    dec_np = np.load(dpath)
    
    enc_arr = enc_np[0]
    for e in enc_np[1:]:
      enc_arr = np.vstack([enc_arr, e])
    dec_arr = dec_np[0]
    for e in dec_np[1:]:
      dec_arr = np.vstack([dec_arr, e])
    enc.append(enc_arr)
    dec.append(dec_arr)
  return enc, dec

def seq_to_matrix(sequence, vocab_size=2005):
  matrix = np.zeros([sequence.shape[0], sequence.shape[1], vocab_size])
  for i, sent in enumerate(sequence):
    for j, word in enumerate(sent):
      matrix[i, j, word] = 1
  return matrix

def matrix_to_seq(matrix):
  seq = np.zeros([matrix.shape[0], matrix.shape[1]])
  for i, sent in enumerate(matrix):
    # seq[i,] = np.where(sent == 1)[1] # find the position of 1
    seq[i,] = np.argmax(sent, axis=1) # find the most highest probabilty ?
  return seq



def main():
  #ds = read_data()
  #vdict = read_vocab_dict()
  #tpath = basic_tokenizer(ds, vdict)
  #tdata = read_tokenized_data()
  #epath, dpath = build_training_set(tdata, vdict)

  enc, dec = read_training_set()

  #print('Length of enc and dec list: ', len(enc), ', ', len(dec))
  #for idx in range(len(enc)):
  #  print('Shape of enc and dec elements: ', enc[idx].shape, ', ', dec[idx].shape)
  #  (15, 20) = (21509, None)
  #  (30, 35) = (7294, None)
  #  (45, 50) = (1797, None)
  
if __name__ == '__main__':
  main()
