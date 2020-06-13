# -*- coding: utf-8 -*-
"""
# Offensive Word Classification
"""

import sys, os, re, csv, codecs, string

import numpy as np
import pandas as pd

import nlpaug.augmenter.char as nac # Reference: https://nlpaug.readthedocs.io/en/latest/augmenter/char/ocr.html

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Embedding, Dropout,CuDNNLSTM, Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential, load_model #, load_weights
from keras import optimizers

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


class hp:
  """Set some basic hyper parameters:"""

  embed_size = 350 # 50 dimensional GloVe word vector  + 300 dimensional one-hot vector for denoising
  max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
  maxlen = 100 # max number of words in a comment to use

def char_level_rep(word):
  """
  Generate the character level representation vectors (300 dimensions) for an input word
  :param word: Word to get character level representation vectors generated
  """
  # find the 300 dimensions for each word in a sentence
  chars = string.printable
  # Create vectors
  v1 = np.zeros(len(chars))
  v2 = np.zeros(len(chars))
  v3 = np.zeros(len(chars))

  for w in word:
      for i, c in enumerate(chars):
          if w == c:
              # Create vector where the index of a character has the count of that character in the word
              v2[i] = v2[i] + 1

  for i, char in enumerate(chars):
      if word[0] == char:
          # Create the one-hot vector for the first character of the word
          v1[i] = 1
      if word[len(word) - 1] == char:
          # Create a one-hot vector for the last character of the word
          v3[i] = 1

  # 300 dimentions for each word in a sentence
  vector_rep = np.concatenate((v1, v2, v3))  # 300

  return vector_rep  # 300



def augment_text_ocr(comment):
  """
  OCRAug adds noise to a comment by replacing the target characters with predefined mapping table 
  """

  aug = nac.OcrAug(aug_char_p=0.3, 
                   aug_word_p=0.4,
                   aug_word_min=len(comment))
  try: 
    augmented_texts = aug.augment(comment,n=1)
  except: 
    augmented_texts = None
  return augmented_texts



def tokenize_text(num_words=hp.max_features,maxlen=hp.maxlen):
  """
  Standard keras preprocessing, to turn each comment into a list of 
  word indexes of equal length (with truncation or padding as needed).
  """

  tokenizer = Tokenizer(num_words=num_words)
  tokenizer.fit_on_texts(list(list_sentences_train))
  list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
  list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
  X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
  X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
  return tokenizer, X_t, X_te


def get_coefs(word,*arr): 
  return word, np.asarray(arr, dtype='float32')



def create_embedding_matrix(tokenizer, emb_mean, emb_std, embed_size=hp.embed_size, num_words=hp.max_features):
  """
  Creates an embedding matrix by concatenating GloVe's 50-dim pre-trained word embedding 
  with the 300-dim one-hot vector (from char_level_rep function) 
  """
  word_index = tokenizer.word_index
  nb_words = min(num_words, len(word_index))
  embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
  for word, i in word_index.items():
      if i >= num_words: continue
      embedding_vector = embeddings_index.get(word)
      
      if embedding_vector is not None:
        word_with_noise = augment_text_ocr(word)
        if word_with_noise is None: word_with_noise = word # some word such as 'a' turn into NoneType after augmentation!
        char_embedding_vectors = char_level_rep(word_with_noise)
        embedding_matrix[i] = np.concatenate((embedding_vector , char_embedding_vectors), axis=None)
    
  return embedding_matrix



def BIDIRECTIONAL_LSTM_MODEL(weights, X_t, labels):
  """ Bidirectional LSTM Model:
  """

  model = Sequential()

  model.add(Embedding(hp.max_features, hp.embed_size, weights=[weights],input_shape=(hp.maxlen,)))
  # model.add(Bidirectional(LSTM(50, return_sequences=True))) # For running on CPU
  model.add(Bidirectional(CuDNNLSTM(50, return_sequences=True))) # For running on GPU
  model.add(GlobalMaxPool1D())
  model.add(Dropout(0.15))
  model.add(Dense(50, activation="relu"))
  model.add(Dropout(0.15))
  model.add(Dense(6, activation="sigmoid"))

  sgd_opt = optimizers.SGD(lr=0.01,momentum=0.01,nesterov=True)

  model.compile(loss='binary_crossentropy', optimizer=sgd_opt, metrics=['accuracy'])
  model.summary()
  
  """Now we're ready to fit our model!"""
  model.fit(X_t, labels, batch_size=32, epochs=2, validation_split=0.1,verbose=1);

  """Save the model:"""
  # model.save("model_final.h5")
  return model


def train_model(labels):

  tokenizer, X_t, X_te = tokenize_text(num_words=hp.max_features,maxlen=hp.maxlen) 

  embedding_matrix = create_embedding_matrix(tokenizer=tokenizer,  emb_mean=emb_mean, emb_std=emb_std, embed_size=hp.embed_size, num_words=hp.max_features)

  model = BIDIRECTIONAL_LSTM_MODEL(weights=embedding_matrix, X_t=X_t, labels=labels )

  return model, X_te



if __name__ == '__main__':

  """Set the paths to the input/ouput files:"""
  path = './input/'
  comp = 'jigsaw-toxic-comment-classification-challenge/'
  output_dir = "./output/"
  EMBEDDING_FILE=f'{path}glove.6B/glove.6B.50d.txt'
  TRAIN_DATA_FILE=f'{path}{comp}train.csv'
  TEST_DATA_FILE=f'{path}{comp}test_with_augmentation.csv'
  MODEL_FILE=f'{output_dir}model_final.h5'



  """Read in our data and replace missing values:"""

  train = pd.read_csv(TRAIN_DATA_FILE)
  test = pd.read_csv(TEST_DATA_FILE)

  list_sentences_train = train["comment_text"].fillna("_na_").values
  list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
  y = train[list_classes].values
  list_sentences_test = test["augmented_comment"].fillna("_na_").values
    

  """Read the glove word vectors (space delimited strings) into a dictionary from word->vector."""
  embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE))

  """Use these vectors to create our embedding matrix, with random initialization for words that aren't in GloVe. We'll use the same mean and stdev of embeddings the GloVe has when generating the random init."""

  all_embs = np.stack(embeddings_index.values())
  emb_mean,emb_std = all_embs.mean(), all_embs.std()

  if(os.path.exists(MODEL_FILE)):
    model = load_model(MODEL_FILE)
    _, _, X_te = tokenize_text(num_words=hp.max_features,maxlen=hp.maxlen) 
  else:
    model, X_te = train_model(labels=y)
    


  """And finally, get predictions for the test set and prepare a submission CSV:"""

  y_test = model.predict([X_te], batch_size=1024, verbose=1) 

  sample_submission = pd.read_csv(f'{path}{comp}sample_submission.csv')
  sample_submission[list_classes] = y_test
  sample_submission.to_csv(f'{output_dir}submission_TEST.csv', index=False)
