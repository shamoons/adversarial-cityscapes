from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper

from keras.datasets import cifar10
from cifar_model import Model

from keras import backend as K
import tensorflow as tf
import png
import numpy as np

num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_test_tensor = K.variable(x_test)

model = Model(input_shape=x_train.shape[1:], num_classes=num_classes).model
wrap = KerasModelWrapper(model)

fgsm_params = {'eps': 0.3,
               'clip_min': 0.,
               'clip_max': 1.
               #    'y_target': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
               }
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    fgsm = FastGradientMethod(wrap, sess=sess)
    adv = fgsm.generate(x_test_tensor, **fgsm_params)
    print(adv)
    print(sess.run(adv))
    quit()

# sess = tf.InteractiveSession()
# print(adv.eval())
# quit()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    asnumpy = sess.run(adv)

    print(asnumpy)
    # for i, image in enumerate(asnumpy):
    #     png.from_array(image).save("adversarial_examples/" + i + ".png")

# # print(adv)
# i = 0
# for adv_x in tf.unstack(adv):

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     asnumpy = sess.run(tf.unstack(adv_x))
#     print(adv_x)
#     asnumpy = tf.unstack(adv_x)
#     print(asnumpy)
#     png.from_array(asnumpy).save("adversarial_examples/" + i + ".png")
#     i += 1
