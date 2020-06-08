import os
import sys
import subprocess
import numpy as np

# suppress tf init prints
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.models import *
from tensorflow.keras.initializers import *
from tensorflow.keras.optimizers import *

import matplotlib.pyplot as plt

path_to_converter = os.path.abspath("../lwtnn/converters/keras2json.py")

if len(sys.argv) > 1:
    path_to_converter = sys.argv[1]
       

if not os.path.exists(path_to_converter):
    print("could not found keras2json.py. Try to give it as cmd parameter")
    exit(1)

def f(x):
    return np.cos(x[0] + x[1]) # * np.sin(x[0] - x[1])**2

train_size = 200
test_size = 50

x_train = np.random.uniform(-3.0, 3.0, (train_size,2))
y_train = f(x_train.T)

x_test = np.random.uniform(-3.0, 3.0, (test_size,2))
y_test = f(x_test.T)

# floating precision & shape
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)
y_train = y_train.reshape(-1,1)

x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test  = y_test.reshape(-1,1)

print("SHAPES:")
print("x_train.shape:", x_train.shape)
print("x_test.shape:", x_test.shape)    
print("y_train.shape:",y_train.shape)
print("")

def r_squared(y_true, y_pred):
    numerator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_pred)))
    y_true_mean = tf.math.reduce_mean(y_true)
    denominator = tf.math.reduce_sum(tf.math.square(tf.math.subtract(y_true, y_true_mean)))
    r2 = tf.math.subtract(1.0, tf.math.divide(numerator, denominator))
    r2_clipped = tf.clip_by_value(r2, clip_value_min=0.0, clip_value_max=1.0)
    return r2_clipped


# hyper params
num_features = x_train.shape[1]
num_targets = y_train.shape[1]

init_w = TruncatedNormal(mean=0.0, stddev=0.1)
init_b = Constant(value=0.0)
epochs = 250
lr = 0.01
batch_size = 16
optimizer = SGD(lr)

# DNN
model = Sequential()

model.add( Dense(units=6, kernel_initializer=init_w, bias_initializer=init_b, input_shape=(num_features, )) )
model.add( Activation("relu") )
model.add( Dense(units=6, kernel_initializer=init_w, bias_initializer=init_b ) )
model.add( Activation("relu") )
model.add( Dense(units=num_targets, kernel_initializer=init_w, bias_initializer=init_b) )
model.summary()

model.compile(
    loss="mse",
    optimizer=optimizer,
    metrics=[r_squared]
)

architecture = model.to_json()
# save the architecture string to a file somehow, the below will work
with open('model_arch.json', 'w') as architecture_file:
    architecture_file.write(architecture)
    
model.save_weights('model_weights.h5')

out = subprocess.check_output([path_to_converter, "model_arch.json", "var_spec.json", "model_weights.h5"]).decode('UTF-8')

with open('simple_lwtnn_network.json', 'w') as lwtnn_file:
    lwtnn_file.write(out)

history = model.fit(
    x_train,
    y_train,
    batch_size,
    epochs,
    verbose=2,
    validation_data=(x_test, y_test)
)

plt.plot(history.history["r_squared"])
plt.plot(history.history["val_r_squared"])
plt.legend(["train R2", "validation R2"])
plt.show()

for x, y in zip(x_test, y_test):
    x = x.reshape(-1,1)
    y_true = y
    y_pred = model.predict(x.T)
    
    print("pred:", y_pred, "\ttrue:", y_true)
