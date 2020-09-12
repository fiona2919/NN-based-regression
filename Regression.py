# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 22:45:04 2018

@author: M
"""

import numpy as np
import matplotlib.pyplot as plt

# Network architecture
input_units = 15
hidden_layer = 2
hidden_units = np.array([10, 10])
learning_rate = 1e-5
epoch = 1000
minibatch_size = 1
train_N = 576
test_N = 192
no_activation = True # True for linear activation

def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output
def sigmoid_derivative(x):
    return x*(1.0-x)

""" ---Import the csv file ---"""
data = np.genfromtxt('Dataset\\energy_efficiency_data.csv',delimiter=',')
""" ---Initialization of weight and bias---"""
DNN_weight = []
DNN_bias = []
weight_delta = []
bias_delta = []
train_layer_output = []
test_layer_output = []
train_hidden_node = []
test_hidden_node = []
for l in range(hidden_layer+1):
    if l == 0:
        ni = input_units
    else:
        ni = hidden_units[l-1]
    if l == hidden_layer:
        no = 1
        DNN_bias.append(np.ones([1,no]))
    else:
        no = hidden_units[l]
        DNN_bias.append(np.zeros([1,no]))
    bound = np.sqrt(6/(ni+no))
    weight_delta.append(np.zeros([ni,no]))
    bias_delta.append(np.zeros([1,no]))
    DNN_weight.append(np.random.uniform(-bound,bound,[ni,no]))
    train_layer_output.append(np.zeros([train_N,no]))
    test_layer_output.append(np.zeros([test_N,no]))
    train_hidden_node.append(np.zeros([train_N,no]))
    test_hidden_node.append(np.zeros([test_N,no]))
""" ---preprocessing--- """
#np.random.shuffle(data) # shuffle data
onehot_ori = np.zeros([len(data),4])
onehot_gad = np.zeros([len(data),5])
for i in range(len(data)):
    onehot_ori[i,int(data[i,5])-2] = 1
    if data[i,7] > 0:
        onehot_gad[i,int(data[i,7])-1] = 1
X = np.zeros([train_N+test_N,input_units])
select_feature = [2,3,6,7]
#for i,j in zip(range(len(select_feature)),select_feature):
#    if j == 7:
#        X[:,i:] = onehot_gad
#    else:
#        X[:,i] = (data[:,j]-np.min(data[:,j]))/(np.max(data[:,j])-np.min(data[:,j]))
for i in range(5):
    X[:,i] = (data[:,i]-np.min(data[:,i]))/(np.max(data[:,i])-np.min(data[:,i]))
X[:,5:9] = onehot_ori
X[:,9] = (data[:,6]-np.min(data[:,6]))/(np.max(data[:,6])-np.min(data[:,6]))
X[:,10:] = onehot_gad
train_X = X[0:train_N,:]
test_X = X[-test_N:,:]
train_T = (data[0:train_N,8]-np.min(data[:,8]))/(np.max(data[:,8])-np.min(data[:,8]))
test_T = (data[-test_N:,8]-np.min(data[:,8]))/(np.max(data[:,8])-np.min(data[:,8]))
#train_T = data[0:train_N,8]
#test_T = data[-test_N:,8]
loss = []

""" ---training--- """
for _ in range(epoch):
    """ forward """
    inputs = train_X
    for l in range(hidden_layer+1):
        train_hidden_node[l] = np.dot(inputs,DNN_weight[l])+DNN_bias[l]
        if no_activation:
            train_layer_output[l] = train_hidden_node[l]
        elif l == hidden_layer:
            train_layer_output[l] = train_hidden_node[l]
        else:
            train_layer_output[l] = sigmoid(train_hidden_node[l])
        inputs = train_layer_output[l]
    # error function
    loss.append(np.sum(np.square(train_T-train_layer_output[hidden_layer])))
    """ backward """
    rand = np.random.permutation(train_N)
#    rand = range(train_N)
    batch_count = 0
    for n in rand:
        batch_count += 1
        # calculate the gradient
        error = -2*(train_T[n]-train_layer_output[hidden_layer][n])
        for l in range(hidden_layer,-1,-1):
            if no_activation:
                delta = error
            elif l == hidden_layer:
                delta = error
            else:
                delta = np.multiply(error.T,sigmoid_derivative(train_layer_output[l][n]))
            if l == 0:
                weight_delta[l] += np.multiply(train_X[n,None].T,delta)
            else:
                weight_delta[l] += np.multiply(train_layer_output[l-1][n,None].T,delta)
            bias_delta[l] += delta
            error = np.dot(DNN_weight[l],delta.T)
        if batch_count == minibatch_size:
            batch_count = 0
            for ll in range(hidden_layer+1):
                DNN_weight[ll] -= learning_rate*(weight_delta[ll]/minibatch_size)
                DNN_bias[ll] -= learning_rate*(bias_delta[ll]/minibatch_size)
                weight_delta[ll] = np.zeros(DNN_weight[ll].shape)
                bias_delta[ll] = np.zeros(DNN_bias[ll].shape)
""" ---testing--- """ 
inputs = train_X
for l in range(hidden_layer+1):
    train_hidden_node[l] = np.dot(inputs,DNN_weight[l])+DNN_bias[l]
    if no_activation:
        train_layer_output[l] = train_hidden_node[l]
    elif l == hidden_layer:
        train_layer_output[l] = train_hidden_node[l]
    else:
        train_layer_output[l] = sigmoid(train_hidden_node[l])
    inputs = train_layer_output[l]
""" rms error """
rms_train = np.sqrt(np.sum(np.square(train_T-train_layer_output[hidden_layer]))/train_N)
print('train rmse %.5f' % rms_train)

inputs = test_X
for l in range(hidden_layer+1):
    test_hidden_node[l] = np.dot(inputs,DNN_weight[l])+DNN_bias[l]
    if no_activation:
        test_layer_output[l] = test_hidden_node[l]
    elif l == hidden_layer:
        test_layer_output[l] = test_hidden_node[l]
    else:
        test_layer_output[l] = sigmoid(test_hidden_node[l])
    inputs = test_layer_output[l]
""" rms error """
rms_test = np.sqrt(np.sum(np.square(test_T-test_layer_output[hidden_layer]))/test_N)
print('test  rmse %.5f' % rms_test)

plt.plot(train_T,'b')
plt.plot(train_layer_output[hidden_layer],'r')
plt.legend(['label', 'predict'])
plt.xlabel('#th case')
plt.ylabel('heating load')
plt.title('prediction for training data')
plt.show()

plt.plot(test_T,'b')
plt.plot(test_layer_output[hidden_layer],'r')
plt.legend(['label', 'predict'])
plt.xlabel('#th case')
plt.ylabel('heating load')
plt.title('prediction for test data')
plt.show()

plt.plot(loss)
plt.title('training curve')
plt.xlabel('Epoch')
plt.show()

""" --- Feature Selection ---  """    
#Feature = np.zeros([8,1])
#F = np.sum(DNN_weight[0],axis=1)
#Feature[0:5] = F[0:5,None]
#Feature[6] = F[9]
#Feature[5] = np.mean(abs(F[5:9]))
#Feature[7] = np.mean(abs(F[-5:]))
#print(np.argsort(abs(Feature),axis=0).T)




