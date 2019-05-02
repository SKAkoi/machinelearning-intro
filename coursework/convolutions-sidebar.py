# A basic convolution on a 2D Greyscale image
# load the image
import cv2
import numpy as np 
from scipy import misc
i = misc.ascent()

# draw the image with pyplt
import matplotlib.pyplot as plt
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show()

#copy transformed image and get dimensions
i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

#create a filter as a 3x3 array
#this filter detects edges nicely
#it creates a convolution that only passes through sharp edges and straight lines
#filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0] ]
#a couple more to try for fun
filter = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
#filter = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
#if the digits in a filter don't add up to 0 or 1, you should weight them to get them to do so
#if the weights are 1,1,1, 1,2,1, 1,1,1
#Multiply by a weight of 0.1 to get them all to add up to ten
weight = 1

#create a convolution
for x in range(1, size_x-1):
    for y in range(1, size_y-1):
        convolution = 0.0
        convolution = convolution + (i[x-1, y-1]*filter[0][0])
        convolution = convolution + (i[x, y-1]*filter[0][1])
        convolution = convolution + (i[x+1, y-1]*filter[0][2])
        convolution = convolution + (i[x-1, y]*filter[1][0])
        convolution = convolution + (i[x, y]*filter[1][1])
        convolution = convolution + (i[x+1, y]*filter[1][2])
        convolution = convolution + (i[x-1, y+1]*filter[2][0])
        convolution = convolution + (i[x, y+1]*filter[2][1])
        convolution = convolution + (i[x+1, y+1]*filter[2][2])
        convolution = convolution * weight
        if(convolution < 0):
            convolution = 0
        if(convolution > 255):
            convolution = 255
        i_transformed[x,y] = convolution

#plot the image
plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
plt.show()

#create a 2x2 pooling
new_x = int(size_x/2)
new_y = int(size_y/2)
new_image = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):
        pixels = []
        pixels.append(i_transformed[x,y])
        pixels.append(i_transformed[x+1, y])
        pixels.append(i_transformed[x, y+1])
        pixels.append(i_transformed[x+1, y+1])
        new_image[int(x/2),int(y/2)] = max(pixels)
#plot the image
plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
plt.show()

