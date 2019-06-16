import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

#load data
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
print("y = " + str(train_set_y[:, index]) + ", it's a " + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + " '.picture ")

#number of training examples
m_train = len(train_set_x_orig)
#number of test examples
m_test = len(test_set_x_orig)
#dimension of image
num_px = train_set_x_orig.shape[1]
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

#reshape the training and test examples
#using X_flatten = X.reshape(X.shape[0], -1).T will flatten a matrix of shape(a,b,c,d) to a matrix of shape(b*c*d, a)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T 
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

#standardize the data
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

#sigmoid function for activation
def sigmoid(z): 
    denom = 1 + np.exp(-1 * z)
    s = 1 / denom
    return s
#sigmoid example
print("sigmoid([0,2]) = " + str(sigmoid(np.array([0,2]))) )

#initialize parameters
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    #test 
    assert(w.shape == (dim,1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b
#example
dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))

#forward and backward propagation
def propagate(w, b, X, Y): 
    m = X.shape[1]
    #forward propagation
    A = sigmoid(np.dot(w.T, X) + b)
    firstTerm = -1/m
    sumFirstTerm = np.multiply(Y, np.log(A))
    sumSecondTerm = np.multiply(np.subtract(1,Y), np.log(np.subtract(1,A)))
    summedParts = np.sum(sumFirstTerm + sumSecondTerm)
    cost = firstTerm * summedParts
    #backward propagration
    dw = 1/m * np.dot(X, np.subtract(A,Y).T)
    db = 1/m * np.sum(np.subtract(A,Y))
    #confirm
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw": dw, "db": db}
    return grads, cost
w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))

#optimize and update using gradient descent
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        #cost and gradient calculation
        grads, cost = propagate(w,b,X,Y)
        #retrieve derivatives
        dw = grads['dw']
        db = grads['db']
        #update weights
        w = w - learning_rate * dw
        b = b - learning_rate * db
        #record the costs
        if i % 100 == 0: 
            costs.append(cost)
        #print the cost for every hundred iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs
#example
params, grads, costs = optimize(w, b, X, Y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
print ("w = " + str(params["w"]))
print ("b = " + str(params["b"]))
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))

#make prediction
def predict(w, b, X):
    m = X.shape[i]
    Y_predictions = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    #compute vector A of predictions
    A = sigmoid(np.dot(w.T, X) + b)
    #convert probabilities to actual predictions
    for i in range(A.shape[1]):
        if(A[0, i] <= 0.5): 
            Y_prediction[0,i] = 0
        else: 
            Y_prediction[0,i] = 1
    assert(Y_prediction.shape == (1,m))
    return Y_prediction
#example
w = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(w, b, X)))

#build the model
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    #initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])
    #gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    #retrieve parameters
    w = parameters["w"]
    b = parameters["b"]
    #predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    #print errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d
#example
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

#plot learning curve with costs
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()




