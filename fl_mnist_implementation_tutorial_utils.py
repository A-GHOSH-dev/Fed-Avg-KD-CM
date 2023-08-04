

import numpy as np
import random
import cv2
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
tf.config.run_functions_eagerly(True)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K


def load_data(img_path, verbose=-1):
    data = []
    labels = []
    subdirectories = [subdir for subdir in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, subdir))]

    for label, subdir in enumerate(subdirectories):
        subdirectory_path = os.path.join(img_path, subdir)
        image_paths = list(paths.list_images(subdirectory_path))

        for (i, image_path) in enumerate(image_paths):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (28, 28))
            image = np.expand_dims(image, axis=-1)

            data.append(image)
            labels.append(label)

            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i + 1, len(image_paths)))

    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)
    data = data.reshape(-1, 784)

    return data, labels


def create_clients(features, labels, num_clients=12, initial='client'):
    features_shuffled, labels_shuffled = shuffle(features, labels)

    features_split = np.array_split(features_shuffled.reshape(-1, 784), num_clients)
    labels_split = np.array_split(labels_shuffled, num_clients)

    clients = {}
    for i in range(num_clients):
        client_name = f'{initial}_{i}'
        clients[client_name] = (features_split[i], labels_split[i])

    return clients


def batch_data(data, BATCH_SIZE=32):
    features, labels = data
    features = np.reshape(features, (-1, 784))
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(len(features)).batch(BATCH_SIZE)
    return dataset


class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model


def weight_scaling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names]) * bs
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() * bs
    return local_count / global_count


def scale_model_weights(weight, scalar):
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    avg_grad = list()
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)

    return avg_grad


# Initialize the LabelBinarizer
lb = LabelBinarizer()


def test_model(X_test, y_test, model, loss, metrics):
    X_test = tf.reshape(X_test, (-1, 784))
    # Convert labels to one-hot encoded format
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    

    # Evaluate the model on the test data
    loss, metrics = model.evaluate(X_test, y_test, verbose=0)

    # Return the loss and metrics
    return loss, metrics