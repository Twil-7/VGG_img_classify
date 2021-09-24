import cv2
import numpy as np
import random


def read_data():
    data_x = np.zeros((2400, 224, 224))
    data_y = np.zeros((2400, 2))

    for i in range(1200):
        cat_image = cv2.imread("/home/archer/CODE/PF/data/train/cat." + str(i + 1) + ".jpg")
        cat_gray_image = cv2.cvtColor(cat_image, cv2.COLOR_BGR2GRAY)
        cat_resize_image = cv2.resize(cat_gray_image, (224, 224), interpolation=cv2.INTER_AREA)

        data_x[i, :, :] = cat_resize_image / 255
        data_y[i, :] = np.array([1, 0])

    print('the cat images have been download !')

    for i in range(1200, 2400):
        dog_image = cv2.imread("/home/archer/CODE/PF/data/train/dog." + str(i - 1199) + ".jpg")
        dog_gray_image = cv2.cvtColor(dog_image, cv2.COLOR_BGR2GRAY)
        dog_resize_image = cv2.resize(dog_gray_image, (224, 224), interpolation=cv2.INTER_AREA)

        data_x[i, :, :] = dog_resize_image / 255
        data_y[i, :] = np.array([0, 1])

    print('the dog images have been download !')

    return data_x, data_y


def make_network_data():
    data_x, data_y = read_data()
    random_index = np.arange(0, 2400, 1)
    random.shuffle(random_index)

    train_x = np.zeros((2000, 224, 224))
    train_y = np.zeros((2000, 2))
    test_x = np.zeros((400, 224, 224))
    test_y = np.zeros((400, 2))

    for i in range(2000):
        index = random_index[i]
        train_x[i, :, :] = data_x[index, :, :]
        train_y[i, :] = data_y[index, :]

    for i in range(400):
        index = random_index[2000 + i]
        test_x[i, :, :] = data_x[index, :, :]
        test_y[i, :] = data_y[index, :]

    return train_x, train_y, test_x, test_y
