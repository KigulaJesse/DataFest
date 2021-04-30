import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dropout
from keras import regularizers
import matplotlib.pyplot as plt


def yesNo(x):
    if x=="USEmbassy":
        return 1.0
    elif x=="Nakawa":
        return 2.0
    else:
        return 3.0


df = pd.read_csv('Train.csv')
df['site'] = df['site'].apply(lambda x : yesNo(x))
df = df.dropna()

dataset = df.values
X = dataset[:,2:17]
Y = dataset[:,17]
X, Y = X/255.0, Y/255.0

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=10, test_size=0.3)
model = Sequential([
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(15,)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),
])

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

X_train=np.asarray(X_train).astype(np.int)
Y_train=np.asarray(Y_train).astype(np.int)
X_test=np.asarray(X_test).astype(np.int)
Y_test=np.asarray(Y_test).astype(np.int)

hist = model.fit(X_train, Y_train, batch_size=32,  validation_data=(X_test, Y_test), epochs=10)
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
print('\nTest accuracy:',test_acc * 100)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
#model.summary()

