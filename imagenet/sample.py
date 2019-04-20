from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import glob
import numpy as np
import imageio

model = ResNet50(weights='imagenet')
# file_list = glob.glob("adversarial_examples/*.jpg")
file_list = ["adversarial_examples/5.jpg"]

IMAGE_SIZE = 224

boundings = []
for j in range(50):
    start_x = np.random.randint(0, IMAGE_SIZE / 2)
    end_x = np.random.randint(start_x + IMAGE_SIZE / 2, IMAGE_SIZE)
    start_y = np.random.randint(0, IMAGE_SIZE / 2)
    end_y = np.random.randint(start_y + IMAGE_SIZE / 2, IMAGE_SIZE)

    boundings.append([start_x, start_y, end_x, end_y])


for image_path in file_list:
    img = image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img = image.img_to_array(img)
    img = img.astype('float32')
    img /= 255

    i = 0
    for bounding in boundings:
        cropped_x = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3))

        new_end_x = bounding[2] - bounding[0]
        new_end_y = bounding[3] - bounding[1]

        old_start_x = bounding[0]
        old_start_y = bounding[1]

        old_end_x = bounding[2]
        old_end_y = bounding[3]

        cropped_x[0:new_end_x,
                  0:new_end_y] = img[old_start_x:old_end_x, old_start_y:old_end_y]

        cropped_x *= 255
        cropped_x = cropped_x.astype('uint8')

        imageio.imwrite("shifted" + str(i) + ".jpg", cropped_x)

        cropped_x = np.expand_dims(cropped_x, axis=0)

        preds = model.predict(cropped_x)

        print(f"Predicted ({image_path}) :",
              decode_predictions(preds, top=3)[0])
        i += 1
