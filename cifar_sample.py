import keras
import numpy as np

from keras.datasets import cifar10
from cifar_model import Model

num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

model = Model(input_shape=x_test.shape[1:], num_classes=num_classes).model

y_test_hat = model.evaluate(model, x=x_test, y=y_test)

# y_test_hat = np.argmax([y_test_hat])

print(y_test_hat)
# print(y_test)

# for i in range(len(x_test)):
#     mo
#     print(x_test[i])
#     print(y_test[i])
