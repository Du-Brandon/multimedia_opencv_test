import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
print(f"train_images shape: {train_images.shape}, train_labels shape: {train_labels.shape}")

plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(train_images[i], cmap='gray')
    plt.title(f"True Label: {train_labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# model = tf.keras.models.Sequential([
#     tf.keras.layers. Reshape((28, 28, 1), input_shape=(28, 28)),
#     tf.keras. layers. Conv2D(32, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras. layers. Conv2D(64,(3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D((2, 2)),
#     tf.keras. layers. Flatten(),
#     tf.keras. layers. Dense(128, activation='relu'),
#     tf.keras. layers. Dense (10, activation='softmax' )
# ])
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Reshape((28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_images, train_labels, epochs=3, validation_split=0.1) #, batch_size=64

# 畫出訓練過程中的準確率和損失值
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
# plt.title('Epoch vs Accuracy')
plt.xlabel('Epoch')
plt.ylabel('ACC')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(12, 6))
for i in range(10):
    idx = np.random.randint(0, len(test_images))
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[idx], cmap='gray')
    color = 'green' if predicted_labels[idx] == test_labels[idx] else 'red'
    plt.title(f"Predicted: {predicted_labels[idx]} \n Real: {test_labels[idx]}", color=color)
    plt.axis('off')
plt.tight_layout()
plt.show()

# output_dir = "output"
# os.makedirs(output_dir, exist_ok=True)

# output_path = os.path.join(output_dir, "predictions.png")
# plt.savefig(output_path)
# print(f"圖像已儲存至 {output_path}")
# plt.close()