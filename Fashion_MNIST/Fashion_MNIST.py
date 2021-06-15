import tensorflow as tf

# Class for Stopping at 95% accuracy


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True


# Loading TensorFlow Dataset
callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist

# Accessing Data
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalizing
training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    # Input layer
    tf.keras.layers.Flatten(input_shape=(28, 28)),

    # Hidden Layer
    tf.keras.layers.Dense(128, activation=(tf.nn.relu)),

    # Output layer
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])

# Loss function and Optimizer
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)

print(classifications[0])
print(test_labels[0])
