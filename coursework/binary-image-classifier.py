#Retrieve training data
'''
wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip \
    -O /tmp/horse-or-human.zip
'''
#Retrieve validation data
'''
wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip \
    -O /tmp/validation-horse-or-human.zip
'''


import os
#Directory with the training horse images
train_horse_dir = os.path.join('/tmp/horse-or-human/horses')

#Directory with the training human images
train_human_dir = os.path.join('/tmp/horse-or-human/humans')

#Validation directory with training horse images
validation_horse_dir = os.path.join('/tmp/validation-horse-or-human/horses')

#Validation directory with training human images
validation_human_dir = os.path.join('/tmp/validation-horse-or-human/humans')

#print file names
train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])
train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])
validation_horse_names = os.listdir(validation_horse_dir)
print(validation_horse_names[:10])
validation_human_names = os.listdir(validation_human_dir)
print(validation_human_names[:10])

#print total number of images in each dir
print('total training horse images', len(os.listdir(train_horse_dir)))
print('total training human images', len(os.listdir(train_human_dir)))
print('total validation horse images', len(os.listdir(validation_horse_dir)))
print('total validation human images', len(os.listdir(validation_human_dir)))

#visualize a sample of the pictures
#matplotlib inline
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
#parameters for graph to output 4x4 images
nrows = 4
ncols = 4
#initialize index for iterating over images
pic_index = 0
#display batch of 8 horse and human pictures
fig = plt.gcf()
fig.set_size_inches(ncols % 4, nrows * 4)
pic_index += 8
next_horse_pic = [os.path.join(train_horse_dir, fname) for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pic = [os.path.join(train_human_dir, fname) for fname in train_human_names[pic_index-8:pic_index]]
for i, img_path in enumerate(next_horse_pic + next_human_pic):
    #set up subplot
    sp = plt.subplot(nrows, ncols, i+1)
    sp.axis('off')
    img = mpimg.imread(img_path)
    plt.imshow(img)
plt.show()

#build model
import tensorflow as tf
#add convolutional layers and flatten to feed into dense layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),
    tf.keras.layers.MaxPooling2D(2,2), 
    #2nd convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    #3rd convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    #4th convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    #5th convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2), 
    #Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(), 
    #Create a 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'), 
    #Output layer with only 1 neuron - output is a binary classification of 0 or 1
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#print a summery of the model
model.summary()

#compile the model using binary crossentropy for loss and RMSProp for optimization
from tensorflow.keras.optimizers import RMSprop
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['acc'])

#data generators for sourcing the images, converting into float32 tensors and feeding the network
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#rescale images to normalize them
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)
#flow training images in batches of 128 using training data gen
train_generator = train_datagen.flow_from_directory(
    '/tmp/horse-or-human',
    target_size=(300,300), 
    batch_size=128, 
    class_mode='binary')
#flow validation images in batches of 128 using validation data gen
validation_generator = validation_datagen.flow_from_directory(
    '/tmp/validation-horse-or-human', 
    target_size=(300,300), 
    batch_size=32, 
    class_mode='binary'
)

#train the model
#use model.fit_generator() because of the generators for generating flow
'''
history = model.fit_generator(
    train_generator, 
    steps_per_epoch=16, 
    epochs=15,
    verbose=1, 
    validation_data=validation_generator, 
    validation_steps=8
)
'''
model.fit(train_generator, steps_per_epoch=16, epochs=15, verbose=1, validation_data=validation_generator, validation_steps=8)
#clean up
import signal
os.kill(os.getpid(), signal.SIGKILL)


