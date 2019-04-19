from keras.applications.resnet50 import ResNet50
import keras


class Model:
    def __init__(self):
        self.model = ResNet50(weights='imagenet')
