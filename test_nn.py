from activation_functions import sigmoid
from activation_functions import relu
import numpy as np
import neural_network

# Test self implemented functions
X = np.array([[200, 17], [100, 1]])
W = np.array([[1, -3, 5],[-2, 4, 6]])
B = np.array([[-1, 1, 2]])
out = neural_network.dense(X, W, B, relu)
print(f'relu_out = {out}')
out = neural_network.dense(X, W, B, sigmoid)
print(f'sigmoid_out = {out}')

# Tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

tf.random.set_seed(1234)
# multiclass classification
model = Sequential(
    [
        tf.keras.Input(shape=(400,)),
        Dense(units=25, activation='relu', name='L1'),
        Dense(units=15, activation='relu', name='L2'),
        Dense(units=10, activation='linear', name='L3')
    ], name = 'model'
)
model.summary()
[layer1, layer2, layer3] = model.layers
W1, b1 = layer1.get_weights()
W2, b2 = layer2.get_weights()
W3, b3 = layer3.get_weights()
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # softmax, more numerically accurate implementation
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # Adaptive Moment estimation
)
# binary classification
model = Sequential(
    [
        tf.keras.Input(shape=(400,)),
        Dense(units=25, activation='relu', name='L1'),
        Dense(units=15, activation='relu', name='L2'),
        Dense(units=1, activation='linear', name='L3')
    ], name = 'model'
)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # sigmoid, more numerically accurate implementation
    # loss=tf.keras.losses.MeanSquaredError(), # regression
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)
# regression, to predict numerical values
model = Sequential(
    [
        tf.keras.Input(shape=(400,)),
        Dense(units=25, activation='relu', name='L1', kernel_regularizer=tf.keras.regularizers.L2(0.01)), # syntax of adding regularization
        Dense(units=15, activation='relu', name='L2', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
        Dense(units=1, activation='linear', name='L3', kernel_regularizer=tf.keras.regularizers.L2(0.01))
    ], name = 'model'
)
model.compile(
    loss=tf.keras.losses.MeanSquaredError(), # regression
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)
# history = model.fit(X, y, epochs=10) # train the model
# prediction = model.predict(test_data) # prediction
# f_x = tf.nn.softmax(logits) # for multiclass classification
# f_x = tf.nn.sigmoid(logit) # for binary classification

# Multi-label Classification
# Multiple outputs constructed by binary classification


