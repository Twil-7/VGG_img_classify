import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import load_model
import cv2


def create_network():
    inputs = keras.layers.Input((224, 224, 1))

    # First convolution
    conv1 = keras.layers.Conv2D(96, kernel_size=11, strides=4, padding='same')(inputs)  # 56, 56, 96
    relu1 = keras.layers.LeakyReLU()(conv1)                                             # 56, 56, 96
    pool1 = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(relu1)    # 28, 28, 96
    bn1 = keras.layers.BatchNormalization()(pool1)                                      # 28, 28, 96

    # Second convolution
    conv2 = keras.layers.Conv2D(256, kernel_size=5, strides=1, padding='same')(bn1)     # 28, 28, 256
    relu2 = keras.layers.LeakyReLU()(conv2)                                             # 28, 28, 256
    pool2 = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(relu2)    # 14, 14, 256
    bn2 = keras.layers.BatchNormalization()(pool2)                                      # 14, 14, 256

    # Third convolution
    conv3 = keras.layers.Conv2D(384, kernel_size=3, strides=1, padding='same')(bn2)     # 14, 14, 384
    relu3 = keras.layers.LeakyReLU()(conv3)                                             # 14, 14, 384

    # Fourth convolution
    conv4 = keras.layers.Conv2D(384, kernel_size=3, strides=1, padding='same')(relu3)   # 14, 14, 384
    relu4 = keras.layers.LeakyReLU()(conv4)                                             # 14, 14, 384

    # Fifth convolution
    conv5 = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(relu4)   # 7, 7, 256
    relu5 = keras.layers.LeakyReLU()(conv5)                                             # 7, 7, 256
    pool3 = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(relu5)    # 7, 7, 256

    fl = keras.layers.Flatten()(pool3)          # 12544

    # First fully connected
    dense1 = keras.layers.Dense(4096)(fl)       # 4096
    relu6 = keras.layers.LeakyReLU()(dense1)    # 4096
    drop1 = keras.layers.Dropout(0.5)(relu6)    # 4096

    # Second fully connected
    dense2 = keras.layers.Dense(4096)(drop1)    # 4096
    relu7 = keras.layers.LeakyReLU()(dense2)    # 4096
    drop2 = keras.layers.Dropout(0.5)(relu7)    # 4096

    # Third fully connected
    dense3 = keras.layers.Dense(2)(drop2)       # 2

    outputs = keras.layers.Activation('softmax')(dense3)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


# batch generator: reduce the consumption of computer memory
def generator(train_x, train_y, batch_size):

    while 1:
        row = np.random.randint(0, len(train_x), size=batch_size)
        x = train_x[row]
        y = train_y[row]
        yield x, y


# create model and train and save
def train_network(train_x, train_y, test_x, test_y, epoch, batch_size):
    train_x = train_x[:, :, :, np.newaxis]
    test_x = test_x[:, :, :, np.newaxis]

    model = create_network()
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    history = model.fit_generator(generator(train_x, train_y, batch_size), epochs=epoch,
                        steps_per_epoch=len(train_x) // batch_size)

    score = model.evaluate(test_x, test_y, verbose=0)
    print('first_model test accuracy:', score[1])

    model.save('first_model.h5')
    show_plot(history)


# Load the partially trained model and continue training and save
def load_network_then_train(train_x, train_y, test_x, test_y, epoch, batch_size, input_name, output_name):
    train_x = train_x[:, :, :, np.newaxis]
    test_x = test_x[:, :, :, np.newaxis]

    model = load_model(input_name)
    history = model.fit_generator(generator(train_x, train_y, batch_size),
                                  epochs=epoch, steps_per_epoch=len(train_x) // batch_size)

    score = model.evaluate(test_x, test_y, verbose=0)
    print(output_name, 'test accuracy:', score[1])

    model.save(output_name)
    show_plot(history)




