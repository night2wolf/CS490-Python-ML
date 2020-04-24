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
tk = Tokenizer()
tk.fit_on_texts(text)
index_list = tk.texts_to_sequences(text)
x_train = pad_sequences(index_list, maxlen=28)
# load model
model = load_model('model.h5')
result = model.predict(x_train)
print(result)