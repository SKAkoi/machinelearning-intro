import tensorflow as tf 
fmnist = tf.keras.datasets.fashion_mnist
#call back to exit training when accuracy crosses threshold
class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.1):
            print("Reached desired accuracy. Exiting training...")
            self.model.stop_training = True
callbacks = myCallBack
#load data
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()
#normalize data
training_images, test_images = training_images/255.0, test_images/255.0
#build the model
model = tf.keras.models.Sequential([
    #generate 64 filters, each with 3x3 dimension
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)), 
    #create a 2x2 pooling layer
    tf.keras.layers.MaxPooling2D(2,2), 
    #add another convolution and pooling layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
#compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#inspect layers of the model
model.summary()
#train the model
model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
test_loss = model.evaluate(test_images, test_labels)


