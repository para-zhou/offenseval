from keras import layers
from keras.models import Sequential
import numpy as np
import pickle as pkl
from keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout, Flatten, MaxPooling1D, Input, Concatenate
from keras.utils import np_utils
from keras.optimizers import RMSprop


train_vec = np.load('./datasets/train_bert.npy')
train_label = np.load('./datasets/train_label.npy')
test_vec = np.load('./datasets/test_bert.npy')
test_label = np.load('./datasets/test_label.npy')
train_label = np_utils.to_categorical(train_label,num_classes=2)

test_label = np_utils.to_categorical(test_label,num_classes=2)

model = Sequential()
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(2,activation='softmax'))
rmpprop = RMSprop(lr = 0.0001)

model.compile(optimizer='adam',\
	loss='categorical_crossentropy',\
	metrics=['accuracy'])

model.fit(train_vec, train_label, epochs=6, batch_size=16,validation_split=0.11)

loss, accuracy = model.evaluate(test_vec, test_label)
result = model.predict(test_vec)
np.save( 'test_3layer_bert.npy',result)
print('test loss:', loss)
print('test accuracy:', accuracy)
