import tensorflow as tf 
fmnist = tf.keras.datasets.fashion_mnist
#call back to exit training when accuracy crosses threshold
class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.1):
            print("Reached desired accuracy. Exiting training...")
            self.model.stop_training = True
callbacks = myCallBack()
#load data
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()
#reshape training images
training_images = training_images.reshape(60000, 28, 28, 1)
#normalize training images
training_images = training_images/255.0
#reshape test images
test_images = test_images.reshape(10000, 28, 28, 1)
#normalize test images
test_images = test_images/255.0
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

#visualize convolutions
import matplotlib.pyplot as plt 
f, axarr = plt.subplot(3,4)
FIRST_IMAGE = 0
SECOND_IMAGE = 7
THIRD_IMAGE = 26
CONVOLUTION_NUMBER = 1
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
    f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[0,x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[0,x].grid(False)
    f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[0,x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[0,x].grid(False)
    f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
    axarr[0,x].imshow(f3[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    axarr[0,x].grid(False)


