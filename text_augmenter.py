# -*- coding: utf-8 -*-
"""text_augmenter.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vZUhqAPFi-WvDod17nt7x62sN2yfe7jP
"""

import os
import pandas as pd
import nlpaug.augmenter.char as nac


TEST_DATA_FILE = './input/jigsaw-toxic-comment-classification-challenge/test.csv'

test = pd.read_csv(TEST_DATA_FILE)

def augment_text_ocr(comment):
  aug = nac.OcrAug(aug_char_p=0.3, 
                   aug_word_p=0.4,
                   aug_word_min=len(comment))
  try: 
    augmented_texts = aug.augment(comment,n=1)
  except: 
    augmented_texts = None
  return augmented_texts


test['augmented_comment'] = test['comment_text'].apply(augment_text_ocr)



test.to_csv('./input/jigsaw-toxic-comment-classification-challenge/test_with_augmentation.csv', index=False)

