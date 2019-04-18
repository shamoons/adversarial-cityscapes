from cifar_model import Model
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend as K
from keras.datasets import cifar10
import numpy as np
import os
import keras
import imageio
import tensorflow as tf

num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_test_tensor = K.variable(x_test)

model = Model(input_shape=x_train.shape[1:], num_classes=num_classes).model
wrap = KerasModelWrapper(model)

target = keras.utils.to_categorical([0], num_classes)
target = np.repeat(target, 10000, axis=0)

fgsm_params = {
    'eps': 0.05,
    'clip_min': 0.,
    'clip_max': 1.,
    'y_target': target
}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    fgsm = FastGradientMethod(wrap, sess=sess)
    adv = fgsm.generate(x_test_tensor, **fgsm_params)

    print('Done creating examples')

    i = 0
    for adv_x in tf.unstack(adv):
        print('Saving: ', i)
        asnumpy = sess.run(adv_x)

        asnumpy *= 255
        asnumpy = asnumpy.astype('uint8')

        original_image = x_test[i] * 255
        original_image = original_image.astype('uint8')

        imageio.imwrite("adversarial_examples/" + str(i) + ".adv.png", asnumpy)
        imageio.imwrite("adversarial_examples/" +
                        str(i) + ".png", original_image)

        i += 1
