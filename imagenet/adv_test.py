from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from inet import Model
import numpy as np

images = ['images/dog1.jpg', 'images/image_0001.jpg']
model = Model()

for image_path in images:
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.model.predict(x)

    print(preds)
    print('Predicted:', decode_predictions(preds, top=3)[0])
