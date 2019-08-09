import numpy as np
import h5py
import matplotlib.pyplot as plt

def load_data():
    '''
    Function to load Data
    '''
    train_dataset = h5py.File('train.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    test_dataset = h5py.File('test.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def processData(x_train, x_test):
    '''
    process feature data
    '''
    x_train_flatten = x_train.reshape(x_train.shape[0], -1).T
    x_test_flattem = x_test.reshape(x_test.shape[0], -1).T
    x_train = x_train_flatten / 255
    x_test = x_test_flattem / 255
    return x_train, x_test


def initialize_weight(layers_dim):
    '''
    Different initializations lead to different results
    Random initialization is used to break symmetry and make sure different hidden units can learn different things
    Don't intialize to values that are too large
    He initialization works well for networks with ReLU activations.
    '''
    L = len(layers_dim)
    parameter = {}
    for i in range(1,L):
        # parameter['W' + str(i)] = np.random.randn(layers_dim[i], layers_dim[i - 1]) * 0.01 # random initialization
        parameter['W'+str(i)] = np.random.randn(layers_dim[i], layers_dim[i-1]) * np.sqrt((2/layer_dims[i - 1])) #he initialization
        parameter['b'+str(i)] = np.zeros((layers_dim[i], 1))
    return parameter

def activation(Z, activation_type):
    '''
    activation function based on activation type
    '''
    if activation_type == 'sigmoid':
        A = 1 / (1 + np.exp(-Z))
        cache = Z
        return A, cache
    elif activation_type == 'relu':
        A = np.maximum(0, Z)
        cache = Z
        return A, cache



def linear_calculation(A_prev, W,b):
    '''
    Linear calculation part of forward propogation
    '''
    Z = np.dot(W,A_prev) + b
    cache = (A_prev,W,b)
    return  Z, cache

def forward_pass(A_prev, W,b, activation_arr):
    Z, linear_cache = linear_calculation(A_prev, W, b)
    A, activation_cache = activation(Z, activation_arr)
    cache = (linear_cache, activation_cache)
    return A, cache


def L_forward_pass(X,parameters, layer_activation):
    A = X
    L = len(parameters)//2
    caches = []

    for i in range(1,L):
        A_prev = A
        activation = layer_activation[i]
        A,cache = forward_pass(A_prev, parameters['W'+str(i)], parameters['b'+str(i)], activation)
        caches.append(cache)

    activation = layer_activation[L]
    AL, cache = forward_pass(A, parameters['W' + str(L)], parameters['b' + str(L)], activation)
    caches.append(cache)
    return AL,caches

def compute_cost(AL, Y):
	'''
	calculate cross entropy cost
	'''
    m = AL.shape[1]
    cost = (1. / m) * (-np.dot(Y, np.log(AL).T) - np.dot(1 - Y, np.log(1 - AL).T))
    cost = np.squeeze(cost)
    return cost

def activation_backward(dA, cache, action_type):
    '''
    calculate backward prop for activation function
    '''
    if action_type == 'sigmoid':
        Z = cache
        s = 1 / (1 + np.exp(-Z))
        dZ = dA * s * (1 - s)
        return dZ
    elif action_type == 'relu':
        Z = cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

def linear_backward(dZ,cache):
    '''
    backward prop for linear part
    '''
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (np.dot(dZ, A_prev.T)) / m
    db = (np.sum(dZ, axis=1, keepdims=True)) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def back_prop(dA, cache, activation_type):
    linear_cache, activation_cache = cache
    dZ = activation_backward(dA, activation_cache, activation_type)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def L_backward(AL,Y, caches, layers_activation):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1]
    activation_type = layer_activation[L]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = back_prop(dAL, current_cache, activation_type)

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        activation_type = layer_activation[l]
        dA_prev_temp, dW_temp, db_temp = back_prop(grads["dA" + str(l + 2)], current_cache,activation_type)
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters

def model(X, Y, layers_dims,layers_activation, learning_rate=0.005, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []
    parameters = initialize_weight(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = L_forward_pass(X, parameters,layers_activation)
        cost = compute_cost(AL, Y)
        grads = L_backward(AL, Y, caches, layers_activation)
        parameters = update_parameters(parameters, grads, learning_rate=learning_rate)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

def predict(x, y, parameters):
    '''
    :param x: features
    :param y: target
    :param parameters: model weights and bias
    '''
    m = x.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1, m))
    #probas, caches = L_forward_pass(x, parameters)
    probas, caches = L_forward_pass(x, parameters, ['relu', 'relu','relu','relu','sigmoid'])
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    print("Accuracy: " + str(np.sum((p == y) / m)))

#load and process data
x_train,y_train,x_test,y_test,classes=load_data()
x_train, x_test = processData(x_train,x_test)
layer_dims = [12288, 20, 7, 5, 1]
layer_activation = ['relu', 'relu','relu','relu','sigmoid']

#train model
parameters = model(x_train, y_train, layer_dims,learning_rate=0.0075, print_cost=True, layers_activation=layer_activation)

#prediction on train data
predict(x_train,y_train,parameters)

#prediction on test data
predict(x_test, y_test, parameters)






