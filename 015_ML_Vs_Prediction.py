import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Dense

# TRAIN DATA
dataset = np.loadtxt('Log 1b.las', skiprows=37)
x_train = dataset[:, [1, 2, 3, 7]]
y_train = 1/dataset[:, 6]*(10**6)
depth_train = dataset[:, 0]

# TEST DATA
dataset2 = np.loadtxt('Log 1b.las', skiprows=37)
x_test = dataset2[:, [1, 2, 3, 7]]
y_test = 1/dataset2[:, 6]*(10**6)
depth_test = dataset2[:, 0]

# Normalizing the data
nor = StandardScaler()
x_train = nor.fit_transform(x_train)
x_test = nor.transform(x_test)

# Creating ANN Architecture
lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)
model = Sequential()  # intializer
model.add(Dense(12, input_dim=x_test.shape[1], kernel_initializer='glorot_uniform',bias_initializer='glorot_uniform',
                activation=lrelu, kernel_regularizer=l1_l2(l1=0.01, l2=0.01), bias_regularizer=l1_l2(l1=0.01, l2=0.01)))  # input layer
model.add(Dense(6, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', activation=lrelu,
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01), bias_regularizer=l1_l2(l1=0.01, l2=0.01)))  # 1st hidden layer
model.add(Dense(3, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', activation=lrelu,
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01), bias_regularizer=l1_l2(l1=0.01, l2=0.01)))  # 2nd hidden layer
model.add(Dense(1, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', activation='linear',
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01), bias_regularizer=l1_l2(l1=0.01, l2=0.01)))  # output layer
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])  # Compiling the ANN
model.summary()

# Training Model
model.fit(x_train, y_train, batch_size=10, epochs=500, verbose=1, validation_split=0.2)

vs_pred = model.predict(x_test)

plt.subplot(1, 2, 1)
plt.plot(y_test, depth_test)
plt.ylim(np.max(depth_test), np.min(depth_test))
plt.xlim(800, 1600)
plt.subplot(1, 2, 2)
plt.plot(vs_pred,  depth_test)
plt.ylim(np.max(depth_test), np.min(depth_test))
plt.xlim(800, 1600)
plt.show()