from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
import glob

file_list = glob.glob("adversarial_examples/*.jpg")
file_list = glob.glob("images/*.jpg")

model = ResNet50(weights='imagenet')

for image_path in file_list:
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)

    print(f"Predicted ({image_path}) :", decode_predictions(preds, top=3)[0])
