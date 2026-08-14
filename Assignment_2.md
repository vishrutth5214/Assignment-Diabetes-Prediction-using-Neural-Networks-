ASSIGNMENT REPORT
Image Classification using Convolutional Neural Networks (CNN)

Fashion-MNIST Dataset | TensorFlow/Keras
1. Objective
The objective of this assignment is to build an end-to-end Convolutional Neural Network (CNN) using the Fashion-MNIST dataset to classify grayscale images of clothing into ten categories. The workflow includes data preparation, image normalization and reshaping, CNN model construction, model training, evaluation, and analysis of the CNN architecture.
2. Dataset Description
Fashion-MNIST is a dataset of grayscale images representing fashion products. It contains 70,000 images: 60,000 training images and 10,000 test images. Each image is 28 × 28 pixels and belongs to one of ten classes.
Label	Category
0	T-shirt / Top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle Boot
3. Part 1 — Data Preparation
3.1 Import Libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
3.2 Load the Fashion-MNIST Dataset
fashion_mnist = tf.keras.datasets.fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
3.3 Display Dataset Information
print("Number of training images:", X_train.shape[0])
print("Number of testing images:", X_test.shape[0])
print("Image dimensions:", X_train.shape[1:])
print("Number of classes:", len(np.unique(y_train)))
Expected dataset characteristics: 60,000 training images, 10,000 test images, 28 × 28 pixel images, and 10 classes.
3.4 Normalize Pixel Values
The original pixel values range from 0 to 255. They are divided by 255 so that the values fall between 0 and 1. This provides a more suitable numerical range for neural-network training.
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
3.5 Reshape Images for CNN
CNN Conv2D layers expect image data in the form (number of images, height, width, channels). Because Fashion-MNIST is grayscale, each image has one channel.
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
3.6 Display Sample Images
class_names = [
    "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

plt.figure(figsize=(10, 5))

for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()
Output: Insert the screenshot of the five displayed sample images here after execution.
4. Part 2 — Building the CNN Model
The CNN consists of two convolution layers, two max-pooling layers, a flatten layer, a dense hidden layer, and a ten-neuron output layer. ReLU is used in the convolution and hidden layers, while Softmax is used for the ten-class output.
4.1 CNN Architecture
Layer	Configuration	Purpose
Input	28 × 28 × 1	Grayscale Fashion-MNIST image
Conv2D	32 filters, 3 × 3	Feature extraction, ReLU
MaxPooling2D	2 × 2	Spatial reduction
Conv2D	64 filters, 3 × 3	Higher-level feature extraction, ReLU
MaxPooling2D	2 × 2	Further spatial reduction
Flatten	1D vector	Converts feature maps to vector
Dense	128 neurons	Hidden classification layer, ReLU
Output Dense	10 neurons	Softmax class probabilities
4.2 CNN Code
model = Sequential()

model.add(
    Conv2D(32, kernel_size=(3, 3),
           activation="relu",
           input_shape=(28, 28, 1))
)

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(
    Conv2D(64, kernel_size=(3, 3),
           activation="relu")
)

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dense(10, activation="softmax"))

model.summary()
4.3 Model Compilation
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
Adam is used as the optimizer, Sparse Categorical Crossentropy is used as the loss function because the target labels are integer class IDs, and accuracy is used as the evaluation metric.
5. Part 3 — Model Training
For the initial training configuration, 10 epochs and a batch size of 64 are used. A 10% validation split is taken from the training data so that training progress can be monitored without using the final test set.
epochs = 10
batch_size = 64

history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1
)
5.1 Training Accuracy
final_training_accuracy = history.history["accuracy"][-1]

print("Final Training Accuracy:", final_training_accuracy)
Actual result after execution: ______________________________
Number of epochs selected: 10
Batch size selected: 64
5.2 Training Curves
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()
Output: Insert the generated accuracy/loss graphs here after execution.
6. Part 4 — Model Evaluation
6.1 Evaluate the Test Dataset
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)
Actual test accuracy after execution: ______________________________
Actual test loss after execution: ______________________________
6.2 Generate Predictions
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
6.3 Display Predictions for 10 Test Images
plt.figure(figsize=(15, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap="gray")

    predicted_label = class_names[y_pred[i]]
    actual_label = class_names[y_test[i]]

    plt.title(
        f"Pred: {predicted_label}\nActual: {actual_label}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()
Output: Insert the screenshot showing the ten predictions here after execution.
6.4 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(10, 8))
plt.imshow(cm)
plt.colorbar()

plt.xticks(range(10), class_names, rotation=45, ha="right")
plt.yticks(range(10), class_names)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()
Output: Insert the generated confusion matrix here after execution.
6.5 Classification Report
print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names
    )
)
Output: Insert the generated classification report here after execution.
7. Part 5 — Model Analysis
1. Why are CNNs more suitable than Artificial Neural Networks (ANNs) for image classification?
CNNs are designed to work with spatial information in images. Convolution filters can learn local visual patterns such as edges, lines, curves, textures, and shapes. CNNs also use shared weights, which greatly reduces the number of parameters compared with a fully connected ANN operating directly on all image pixels. Therefore, CNNs can preserve spatial relationships and learn useful image features more efficiently.
2. What is the purpose of a Convolution layer?
A convolution layer extracts useful features from an image by applying learnable filters across local regions of the image. Early convolution layers generally learn simple patterns such as edges and lines, while deeper layers can combine these patterns into more complex shapes and visual features.
3. Why is Max Pooling used in CNNs?
Max Pooling reduces the spatial dimensions of feature maps while retaining strong or important activations. This reduces computation and the number of values that later layers need to process. It also makes the representation less sensitive to small changes in the exact position of a feature.
4. Why is the Flatten layer required before the Dense layers?
Convolution and pooling layers produce multi-dimensional feature maps. Dense layers require a one-dimensional input vector. The Flatten layer converts the multi-dimensional feature maps into a one-dimensional vector so that the features can be passed to the Dense layers for classification.
5. Why is the Softmax activation function used in the output layer?
Fashion-MNIST has ten mutually exclusive classes. Softmax converts the output values into a probability distribution across the ten classes. The probabilities sum to approximately 1, and the class with the highest probability is selected as the predicted class.
6. How does the Adam optimizer help in training the CNN?
Adam updates the model's weights using gradients obtained during backpropagation. It adapts the step size for different parameters using information from previous gradients, helping the model reduce its loss efficiently. Adam is widely used because it generally provides fast and effective training with relatively little manual tuning.
8. Conclusion
A Convolutional Neural Network was designed to classify Fashion-MNIST images into ten clothing categories. The workflow included loading and exploring the dataset, normalizing and reshaping the images, building a CNN with convolution, pooling, flattening and dense layers, and compiling the model using Adam, Sparse Categorical Crossentropy and accuracy. The trained model was then evaluated on unseen test data using test accuracy, predictions, a confusion matrix and a classification report. The exercise demonstrates how CNNs automatically learn visual features and use them for image classification.

