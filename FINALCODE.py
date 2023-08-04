import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K

# Define the student model
def create_student_model():
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Define the teacher model
def create_teacher_model():
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(512, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Define the function for federated training with knowledge distillation
def federated_train_kd(student_model, teacher_model, train_data):
    student_model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.1),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    student_model.fit(train_data, epochs=1, verbose=0)

    teacher_model.trainable = False
    student_model.trainable = False

    distillation_model = keras.models.clone_model(student_model)
    distillation_model.set_weights(student_model.get_weights())

    distillation_model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.1),
        loss=knowledge_distillation_loss(teacher_model, temperature=10),
        metrics=['accuracy']
    )
    distillation_model.fit(train_data, epochs=1, verbose=0)

    student_model.set_weights(distillation_model.get_weights())

# Define the knowledge distillation loss function
def knowledge_distillation_loss(teacher_model, temperature):
    def loss(y_true, y_pred):
        y_true_teacher = teacher_model(y_true)
        y_pred_teacher = teacher_model(y_pred)
        loss = tf.keras.losses.KLDivergence()(tf.nn.softmax(y_true_teacher / temperature),
                                              tf.nn.softmax(y_pred_teacher / temperature))
        return loss * (temperature * temperature)
    return loss

# Define the function for federated conditional mutual learning
def federated_conditional_mutual_learning(student_models):
    for i, student_model in enumerate(student_models):
        student_model.trainable = True
        student_model.compile(
            optimizer=keras.optimizers.SGD(learning_rate=0.1),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        student_model.fit(train_data, epochs=1, verbose=0)

        for j, teacher_model in enumerate(student_models):
            if i != j:
                teacher_model.trainable = False
                student_model.trainable = False

                distillation_model = keras.models.clone_model(student_model)
                distillation_model.set_weights(student_model.get_weights())

                distillation_model.compile(
                    optimizer=keras.optimizers.SGD(learning_rate=0.1),
                    loss=knowledge_distillation_loss(teacher_model, temperature=10),
                    metrics=['accuracy']
                )
                distillation_model.fit(train_data, epochs=1, verbose=0)

                student_model.set_weights(distillation_model.get_weights())

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape the images to (batch_size, 28, 28)
train_images = train_images.reshape(train_images.shape[0], 28, 28)
test_images = test_images.reshape(test_images.shape[0], 28, 28)

# Split the dataset into multiple workers
train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(64)

# Create the student models
student_models = [create_student_model() for _ in range(3)]

# Train the teacher model
teacher_model = create_teacher_model()
teacher_model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.1),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
teacher_model.fit(train_images, train_labels, epochs=3, verbose=0)

# Perform federated learning with knowledge distillation
for epoch in range(3):
    for data, labels in train_data:
        for student_model in student_models:
            federated_train_kd(student_model, teacher_model, (data, labels))

    # Average the student models' weights using federated averaging
    averaged_weights = []
    for layer_weights in zip(*[student_model.get_weights() for student_model in student_models]):
        averaged_weights.append(tf.reduce_mean(layer_weights, axis=0))
    for student_model in student_models:
        student_model.set_weights(averaged_weights)

# Perform federated conditional mutual learning
federated_conditional_mutual_learning(student_models)

# Test the student models
test_loss = []
test_accuracy = []
for student_model in student_models:
    student_model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.1),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    loss, accuracy = student_model.evaluate(test_images, test_labels, verbose=0)
    test_loss.append(loss)
    test_accuracy.append(accuracy)

for i in range(len(student_models)):
    print(f'Student {i+1} - Test Loss: {test_loss[i]:.4f}, Test Accuracy: {test_accuracy[i]*100:.2f}%')
