from inet import Model
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend as K
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
import keras
import imageio
import tensorflow as tf

num_classes = 1000

X = []
images = ['images/dog1.jpg', 'images/image_0001.jpg']
for image_path in images:
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    X.append(x)

model = Model()
wrap = KerasModelWrapper(model)

target = [np.zeros((1000,))]
target[0][0] = 1
target = np.repeat(target, len(X), axis=0)

fgsm_params = {
    'eps': 0.05,
    # 'clip_min': 0.,
    # 'clip_max': 1.,
    'y_target': target
}

x_tensor = K.variable(X)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    fgsm = FastGradientMethod(wrap, sess=sess)
    adv = fgsm.generate(x_tensor, **fgsm_params)

    print('Done creating examples')
    quit()

    i = 0
    for adv_x in tf.unstack(adv):
        print('Saving: ', i)
        asnumpy = sess.run(adv_x)

        asnumpy *= 255
        asnumpy = asnumpy.astype('uint8')

        original_image = x_test[i] * 255
        original_image = original_image.astype('uint8')

        imageio.imwrite("../adversarial_examples/" +
                        str(i) + ".adv.png", asnumpy)
        imageio.imwrite("../adversarial_examples/" +
                        str(i) + ".png", original_image)

        i += 1


quit()


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

        imageio.imwrite("../adversarial_examples/" +
                        str(i) + ".adv.png", asnumpy)
        imageio.imwrite("../adversarial_examples/" +
                        str(i) + ".png", original_image)

        i += 1
