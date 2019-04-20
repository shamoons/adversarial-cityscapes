import keras

from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint
from cifar_model import Model

batch_size = 32
num_classes = 10
epochs = 100

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = Model(input_shape=x_train.shape[1:], num_classes=num_classes).model

callbacks = [ModelCheckpoint(
    'models/cifar10.h5', save_best_only=True, save_weights_only=True)]


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks=callbacks,
          shuffle=True)
