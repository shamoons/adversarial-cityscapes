from keras.preprocessing import image
from segmentation_models import Unet
import numpy as np
import imageio

model = Unet(backbone_name='resnet101',
             encoder_weights='imagenet', freeze_encoder=True)
model.compile('Adam', 'categorical_crossentropy', ['categorical_accuracy'])

img = image.load_img('seg_images/image1.png')
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

mask = model.predict(x)
mask = mask.astype('uint8')
imageio.imwrite("result.png", mask[0])