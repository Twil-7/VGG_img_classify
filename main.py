import getdata as gt
import network as nt
import cv2


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = gt.make_network_data()
    nt.train_network(train_x, train_y, test_x, test_y, epoch=50, batch_size=16)
    nt.load_network_then_train(train_x, train_y, test_x, test_y, epoch=100, batch_size=16,
                               input_name='first_model.h5', output_name='second_model.h5')
