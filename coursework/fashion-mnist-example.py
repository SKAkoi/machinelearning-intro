import tensorflow as tf 
print(tf.__version__)
fmnist = tf.keras.datasets.fashion_mnist
#create a call back to exit training when model reaches desired accuracy
class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.1):
            print("\Reached 90% accuracy. Exiting training phase")
            self.model.stop_training = True

callbacks = myCallBack()
#load data
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()
#print an example of the data
import matplotlib.pyplot as plt 
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])
#normalize data
training_images, test_images = training_images/255.0, test_images/255.0
#build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
    #tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
#compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#train model
model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])
#test the model
model.evaluate(test_images, test_labels)
#get classification for an item
classifications = model.predict(test_images)
print(classifications[0])
#print first label
print(test_labels[0])