

import numpy as np
import random
import cv2
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score

import tensorflow as tf
tf.config.run_functions_eagerly(True)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from fl_mnist_implementation_tutorial_utils import *

img_path = 'C:\\Users\\anany\\Desktop\\VIT\\RESEARCH\\sem 7\\Garbage_Collective_Data - Copy'

image_data, labels = load_data(img_path, verbose=1000)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.1, random_state=42)

lb.fit(y_train)
y_train_encoded = lb.fit_transform(y_train)
y_test_encoded = lb.transform(y_test)

y_train = y_train_encoded.argmax(axis=1) 
y_test = y_test_encoded.argmax(axis=1)  


clients = create_clients(X_train, y_train, num_clients=12, initial='client')

clients_batched = {client: batch_data(data) for client, data in clients.items()}

test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))

comms_round = 1
lr = 0.01
loss = 'categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(lr=lr, decay=lr / comms_round, momentum=0.9)

global_model = SimpleMLP.build(shape=784, classes=12)
global_model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

for comm_round in range(comms_round):
    global_weights = global_model.get_weights()
    scaled_local_weight_list = []
    client_names = list(clients_batched.keys())
    random.shuffle(client_names)


    for client in client_names:
        smlp_local = SimpleMLP()
        local_model = smlp_local.build(784, 12)
        local_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        local_model.set_weights(global_weights)

        X_train_batches = []
        y_train_batches = []
        for batch in clients_batched[client]:
            X_train_batches.append(batch[0])
            y_train_batches.append(batch[1])
        
        X_train_batch = np.concatenate(X_train_batches, axis=0)
        y_train_batch = np.concatenate(y_train_batches, axis=0)

        y_train_batch = lb.transform(y_train_batch)
        
        local_model.fit(X_train_batch, y_train_batch, epochs=1, verbose=0)
        scaling_factor = weight_scaling_factor(clients_batched, client)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)
        K.clear_session()


    average_weights = sum_scaled_weights(scaled_local_weight_list)
    global_model.set_weights(average_weights)

    for X_test_batch, Y_test_batch in test_batched:
        global_model_loss, global_model_acc = test_model(X_test_batch, Y_test_batch, global_model, loss, metrics)
        



        
    print('Communication Round:', comm_round + 1)
    print('Global Accuracy:', global_model_acc)
    print('Global Loss:', global_model_loss)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

global_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

y_pred = global_model.predict(X_test_batch)
        

y_pred = np.argmax(y_pred) 
y_true = np.argmax(Y_test_batch)  

cm = confusion_matrix(y_true, y_pred)


fig, ax = plt.subplots(figsize=(8, 6))


sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')

            
class_labels = [str(i) for i in range(12)]  
ax.xaxis.set_ticklabels(class_labels)
ax.yaxis.set_ticklabels(class_labels)

            
plt.show()