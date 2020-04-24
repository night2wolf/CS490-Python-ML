import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from keras.models import load_model
text = "A lot of good things are happening. We are respected again throughout the world, and that's a great thing"
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
X = tokenizer.texts_to_sequences(text)
# 28 for sentiment
#152 for spam
X = pad_sequences(X,maxlen=152)
X = X[[0],:]
# load model
model = load_model('model_spam.h5')
result = model.predict(X)
print(result)