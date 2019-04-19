from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from keras import backend as K
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
import numpy as np
import os
import keras
import imageio
import tensorflow as tf
import glob


num_classes = 1000

X = []
file_list = glob.glob("images/*.jpg")
for image_path in file_list:
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x.astype('float32')
    x /= 255

    X.append(x)

model = ResNet50(weights='imagenet')
wrap = KerasModelWrapper(model)

target = [np.zeros((1000,))]
target[0][0] = 1
target = np.repeat(target, len(X), axis=0)

fgsm_params = {
    'eps': 0.3,
    'clip_min': 0.,
    'clip_max': 1.,
    'y_target': target
}

X = np.array(X)

x_tensor = K.variable(X)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    fgsm = FastGradientMethod(wrap, sess=sess)
    adv = fgsm.generate(x_tensor, **fgsm_params)

    i = 0
    for adv_x in tf.unstack(adv):
        print('Saving: ', i)
        asnumpy = sess.run(adv_x)

        asnumpy *= 255
        asnumpy = asnumpy.astype('uint8')

        imageio.imwrite("adversarial_examples/" +
                        str(i) + ".adv.png", asnumpy)

        i += 1
