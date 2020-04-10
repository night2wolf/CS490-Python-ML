from keras.models import Sequential
from keras.utils.np_utils import to_categorical
import numpy as np
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
#Q3 use 20newsgroups instead
from sklearn.datasets import fetch_20newsgroups
cats = ['alt.atheism', 'talk.religion.misc','comp.graphics', 'sci.space']
newgroups_train = fetch_20newsgroups(subset='train', shuffle=True, categories=cats)

texts = newgroups_train.data
labels = newgroups_train.target
tokenizer = Tokenizer(nb_words=2000)
tokenizer.fit_on_texts(texts)
sentences = tokenizer.texts_to_sequences(texts)
vocab = tokenizer.word_index

data = pad_sequences(sentences,maxlen=1000)
labels = to_categorical(np.asarray(labels))
indices = np.arange(data.shape[0]) 
np.random.shuffle(indices) 
data = data[indices] 
labels = labels[indices]
nb_validation_samples = int(0.2 * data.shape[0])
x_train = data[:-nb_validation_samples] 
y_train = labels[:-nb_validation_samples] 
x_test = data[-nb_validation_samples:] 
y_test = labels[-nb_validation_samples:]



model = Sequential()
model.add(layers.Embedding(len(vocab)+1,100,input_length=1000))
model.add(layers.Flatten())
model.add(layers.Dense(300,activation='relu'))
model.add(layers.Dense(300, activation='sigmoid'))
model.add(layers.Dense(5, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(x_train,y_train, epochs=5, verbose=True, validation_data=(x_test,y_test), batch_size=256)
import matplotlib.pyplot as plt
[test_loss, test_acc] = model.evaluate(x_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()