import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

(x_train_10, y_train_10), (x_test_10, y_test_10) = cifar10.load_data()
x_train_10, x_test_10 = x_train_10 / 255.0, x_test_10 / 255.0
y_train_10 = to_categorical(y_train_10, 10)
y_test_10 = to_categorical(y_test_10, 10)

(x_train_100, y_train_100), (x_test_100, y_test_100) = cifar100.load_data()
x_train_100, x_test_100 = x_train_100 / 255.0, x_test_100 / 255.0
y_train_100 = to_categorical(y_train_100, 100)
y_test_100 = to_categorical(y_test_100, 100)

def build_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = x_train_10.shape[1:]
num_classes_10 = 10

model_10 = build_cnn_model(input_shape, num_classes_10)
model_10.summary()
history_10 = model_10.fit(x_train_10, y_train_10, epochs=20, batch_size=64, validation_data=(x_test_10, y_test_10))
test_loss_10, test_acc_10 = model_10.evaluate(x_test_10, y_test_10, verbose=2)
print(f"CIFAR-10 Test accuracy: {test_acc_10:.4f}")

num_classes_100 = 100
model_100 = build_cnn_model(input_shape, num_classes_100)
model_100.summary()
history_100 = model_100.fit(x_train_100, y_train_100, epochs=20, batch_size=64, validation_data=(x_test_100, y_test_100))
test_loss_100, test_acc_100 = model_100.evaluate(x_test_100, y_test_100, verbose=2)
print(f"CIFAR-100 Test accuracy: {test_acc_100:.4f}")

plt.plot(history_10.history['accuracy'], label='CIFAR-10 Training Accuracy')
plt.plot(history_10.history['val_accuracy'], label='CIFAR-10 Validation Accuracy')
plt.title('CIFAR-10 Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history_100.history['accuracy'], label='CIFAR-100 Training Accuracy')
plt.plot(history_100.history['val_accuracy'], label='CIFAR-100 Validation Accuracy')
plt.title('CIFAR-100 Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
