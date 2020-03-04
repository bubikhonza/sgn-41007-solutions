from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical


model = Sequential()
w, h = 3, 3  # Conv. window size
model.add(Conv2D(32, (w, h),
                 input_shape=(64, 64, 1),
                 activation= 'relu',
                 padding= 'same'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(64, (w, h),
                 activation= 'relu',
                 padding= 'same'))
model.add(MaxPooling2D((4, 4)))
model.add(Flatten())    
model.add(Dense(128, activation= 'relu'))
model.add(Dense(2, activation= 'sigmoid'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])


import traffic_signs as ts
X, y = ts.load_data('./')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = X_train[..., np.newaxis] / 255.0
X_test = X_test[..., np.newaxis] / 255.0
y_train  = to_categorical(y_train)
y_test = to_categorical(y_test)

model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=[X_test, y_test])
