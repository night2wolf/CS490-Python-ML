# Simple CNN model for CIFAR-10
import numpy as np
import os
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
K.set_image_dim_ordering('th')
# load model
model = load_model('./model.h5')
# summarize model.
model.summary()
# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
#Q2 load 4 images and predictions
import numpy as np 
from keras.preprocessing import image
# images to test
test_image1 =image.load_img('airplane.jpeg',target_size =(32,32))
test_image2 =image.load_img('cat.jpeg',target_size =(32,32))
test_image3 =image.load_img('frog.jpeg',target_size =(32,32))
test_image4 =image.load_img('horse.jpeg',target_size =(32,32))
def result(image1):
  test_image =image.img_to_array(image1) 
  test_image =np.expand_dims(test_image, axis =0) 
  result = model.predict(test_image) 
  print(result) 
  if result[0][0]==1: 
    print("Aeroplane") 
  elif result[0][1]==1: 
    print('Automobile') 
  elif result[0][2]==1: 
    print('Bird') 
  elif result[0][3]==1: 
    print('Cat') 
  elif result[0][4]==1: 
    print('Deer') 
  elif result[0][5]==1: 
    print('Dog') 
  elif result[0][6]==1: 
    print('Frog') 
  elif result[0][7]==1: 
    print('Horse') 
  elif result[0][8]==1: 
    print('Ship') 
  elif result[0][9]==1: 
    print('Truck') 
  else: 
    print('Error')
# %matplotlib inline
  import matplotlib.pyplot as plt 
  plt.imshow(image1)
result(test_image1)  
result(test_image2)  
result(test_image3)  
result(test_image4)  