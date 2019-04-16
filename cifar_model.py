import keras
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense


class Model:
    def __init__(self, input_shape, num_classes):

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same',
                              input_shape=input_shape))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.25))
        self.model.add(BatchNormalization())

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.25))
        self.model.add(BatchNormalization())

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(BatchNormalization())

        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))

        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        self.model.load_weights('models/cifar10.h5')

        # Let's train the model using RMSprop
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['accuracy'])
