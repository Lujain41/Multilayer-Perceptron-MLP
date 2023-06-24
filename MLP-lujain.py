# WELCOME TO " MULTILAYAR PERCEPTRON " CODE DONE BY  LUJAIN ALI ALSHEHRI 

#import the important libraries must uses in this code

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.special import expit
import matplotlib.pyplot as plt
from tqdm import tqdm


# read CSV file into DataFrame
data_fram=pd.read_csv(r'wdbc.data',header=None)
data_fram=data_fram.drop(0,axis=1)

# classify the data by category label column & Shuffle the balanced data
data_index = data_fram.groupby(data_fram.iloc[:, 0])
data_size = data_index.size().min()
data_balanced = pd.concat([group.sample(data_size) for _, group in data_index])
data_balanced = data_balanced.sample(frac=1).reset_index(drop=True)
counts = data_balanced.iloc[:, 0].value_counts()
print(counts)

# Convert data to numpy arrays and Encoding the class label
Y = data_balanced[[1]]
X = data_balanced.drop(1,axis=1)
Y=Y.replace({'B':1,'M':0})
Y=Y.astype('int')
X=np.array(X)
Y=np.array(Y)
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2,random_state=147)

# Check the balanced of data
train_y_counts = pd.Series(y_train.flatten()).value_counts()
test_y_counts = pd.Series(y_test.flatten()).value_counts()

print("The set of training counts:", train_y_counts)
print('\n')
print("The set of testing  counts:\n", test_y_counts)
print('\n')

# build Multilayer Perceptron class 
class multilayer_perceptron:
    def __init__(self, num_of_inputs, num_of_hidden, num_of_outputs, learning_rate=0.1,random_seed=None):

        np.random.seed(random_seed)
        self.num_of_inputs = num_of_inputs
        self.num_of_hidden = num_of_hidden
        self.num_of_outputs = num_of_outputs
        self.learning_rate = learning_rate

       # Initialize weights and biases
        self.weights1 = np.random.randn(num_of_inputs, num_of_hidden)
        self.bias1 = np.zeros(num_of_hidden)
        self.weights2 = np.random.randn(num_of_hidden, num_of_outputs)
        self.bias2 = np.zeros(num_of_outputs)

    # compute sigmoid as activation function 
    def sigmoid_AF(self, x):
        return expit(x)

    def sigmoid_AF_derivative(self, x):
        return x * (1 - x)

    # calculate forward pass
    def forward_pass(self, x):
        hidden_layer = self.sigmoid_AF(np.dot(x, self.weights1) + self.bias1)
        output_layer = self.sigmoid_AF(np.dot(hidden_layer, self.weights2) + self.bias2)
        return output_layer, hidden_layer

    # calculate binary cross-entropy loss BCE
    def cross_entropy_loss(self, y_predict, y_true):
        eps = 1e-8
        cross_entropy_loss = -np.mean(y_true * np.log(y_predict + eps) + (1 - y_true) * np.log(1 - y_predict + eps))
        return cross_entropy_loss

    # Backward pass on train data & calculate gradients
    def back_propagation(self, x, y_true):
        y_predict, hidden_layer = self.forward_pass(x)

        output_error = y_predict - y_true
        output_delta = output_error * self.sigmoid_AF_derivative(y_predict).reshape(1, 1)

        hidden_error = np.dot(output_delta, self.weights2.T)
        hidden_delta = hidden_error * self.sigmoid_AF_derivative(hidden_layer)

        weights2_train = np.dot(hidden_layer.T, output_delta)
        bias2_train = np.sum(output_delta, axis=0)
        weights1_train = np.dot(x.T, hidden_delta)
        bias1_train = np.sum(hidden_delta, axis=0)

        # update weights and biases
        self.weights2 -= self.learning_rate * weights2_train
        self.bias2 -= self.learning_rate * bias2_train
        self.weights1 -= self.learning_rate * weights1_train
        self.bias1 -= self.learning_rate * bias1_train

    def train_mlp(self, X_train, y_train, X_test, y_test, num_of_epochs=100,plot=''):
        train_losses_history = []
        test_losses_history = []
        train_accs_history = []
        test_accs_history = []
     
        # train the epochs
        for epoch in tqdm(range(num_of_epochs), desc='training'):
            for x, y in zip(X_train, y_train):
                self.back_propagation(x.reshape(1, -1), y.reshape(1, -1))

            # calculate training & test loss and append to lists
            train_loss_history = self.cross_entropy_loss(self.forward_pass(X_train)[0], y_train)
            test_loss_history = self.cross_entropy_loss(self.forward_pass(X_test)[0], y_test)
            train_acc_history = np.mean(y_train == np.round(self.forward_pass(X_train)[0]))
            test_acc_history = np.mean(y_test == np.round(self.forward_pass(X_test)[0]))

            train_losses_history.append(train_loss_history)
            test_losses_history.append(test_loss_history)
            train_accs_history.append(train_acc_history)
            test_accs_history.append(test_acc_history)


        # Plot loss history
        if plot=='losses':
            plt.plot(train_losses_history,'red', label='Train loss')
            plt.plot(test_losses_history,'yellow',label='Test loss')
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
        # Plot accuracy history
        elif plot == 'accuracy':
            plt.plot(train_accs_history,'purple', label='Train accuracy')
            plt.plot(test_accs_history, 'green',label='Test accuracy')
            plt.legend()
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.show()
        return train_acc_history

    def prediction(self, X):
        # make predict y
        y_predict = np.round(self.forward(X)[0])
        return y_predict

#Plot a figure for train/test classification accuracy for the training epochs.
##model_num_of_epochs=[80, 100, 160 ]
model_num_of_epochs =  multilayer_perceptron(num_of_inputs=30,num_of_hidden=30,num_of_outputs=1,learning_rate =0.005)
accuracy1 = model_num_of_epochs.train_mlp(X_train, y_train,X_test, y_test, num_of_epochs=80,plot='accuracy')

model_num_of_epochs =  multilayer_perceptron(num_of_inputs=30,num_of_hidden=30,num_of_outputs=1,learning_rate =0.005)
accuracy2 = model_num_of_epochs.train_mlp(X_train, y_train, X_test, y_test, num_of_epochs=100,plot='accuracy')

model_num_of_epochs = multilayer_perceptron(num_of_inputs=30,num_of_hidden=30,num_of_outputs=1,learning_rate =0.005)
accuracy3 = model_num_of_epochs.train_mlp(X_train, y_train, X_test, y_test, num_of_epochs=160,plot='accuracy')

df = pd.DataFrame({
  "Epochs_Num_": [80, 100, 160 ],
  "Acuracy_": [accuracy1, accuracy2, accuracy3 ]
})

print(df)

model_num_of_epochs =  multilayer_perceptron(num_of_inputs=30,num_of_hidden=30,num_of_outputs=1,learning_rate =0.005)
accuracy1 = model_num_of_epochs.train_mlp(X_train, y_train, X_test, y_test, num_of_epochs=80,plot='losses')

model_num_of_epochs =  multilayer_perceptron(num_of_inputs=30,num_of_hidden=30,num_of_outputs=1,learning_rate =0.005)
accuracy2 = model_num_of_epochs.train_mlp(X_train, y_train, X_test, y_test, num_of_epochs=100,plot='losses')

model_num_of_epochs = multilayer_perceptron(num_of_inputs=30,num_of_hidden=30,num_of_outputs=1,learning_rate =0.005)
accuracy3 = model_num_of_epochs.train_mlp(X_train, y_train, X_test, y_test, num_of_epochs=160,plot='losses')

df = pd.DataFrame({
  "Epochs_Num_": [80, 100, 160 ],
  "Loss_": [accuracy1, accuracy2, accuracy3 ]
})

print(df)

#rain the MLP with one hidden layer with different learning rates [1.0, 0.5, 0.1, 0.01] 
## in the Acuracy axsis
model_learning_rate =  multilayer_perceptron(num_of_inputs=30,num_of_hidden=30,num_of_outputs=1,learning_rate =1.0)
accuracy1 = model_learning_rate.train_mlp(X_train, y_train, X_test, y_test, num_of_epochs=160,plot='accuracy')

model_learning_rate =  multilayer_perceptron(num_of_inputs=30,num_of_hidden=30,num_of_outputs=1,learning_rate =0.5)
accuracy2 = model_learning_rate.train_mlp(X_train, y_train, X_test, y_test, num_of_epochs=160,plot='accuracy')

model_learning_rate = multilayer_perceptron(num_of_inputs=30,num_of_hidden=30,num_of_outputs=1,learning_rate =0.1)
accuracy3 = model_learning_rate.train_mlp(X_train, y_train, X_test, y_test, num_of_epochs=160,plot='accuracy')

model_learning_rate = multilayer_perceptron(num_of_inputs=30,num_of_hidden=30,num_of_outputs=1,learning_rate =0.01)
accuracy4 = model_learning_rate.train_mlp(X_train, y_train, X_test, y_test, num_of_epochs=160,plot='accuracy')

df = pd.DataFrame({
  "Learning_Rate_": [1.0, 0.5, 0.1 , 0.01 ],
  "Accuracy_": [accuracy1, accuracy2, accuracy3, accuracy4 ]
})
print(df)

## in the Loss axsis
model_learning_rate =  multilayer_perceptron(num_of_inputs=30,num_of_hidden=30,num_of_outputs=1,learning_rate =1.0)
accuracy1 = model_learning_rate.train_mlp(X_train, y_train, X_test, y_test, num_of_epochs=160,plot='losses')

model_learning_rate =  multilayer_perceptron(num_of_inputs=30,num_of_hidden=30,num_of_outputs=1,learning_rate =0.5)
accuracy2 = model_learning_rate.train_mlp(X_train, y_train, X_test, y_test, num_of_epochs=160,plot='losses')

model_learning_rate = multilayer_perceptron(num_of_inputs=30,num_of_hidden=30,num_of_outputs=1,learning_rate =0.1)
accuracy3 = model_learning_rate.train_mlp(X_train, y_train, X_test, y_test, num_of_epochs=160,plot='losses')

model_learning_rate = multilayer_perceptron(num_of_inputs=30,num_of_hidden=30,num_of_outputs=1,learning_rate =0.01)
accuracy4 = model_learning_rate.train_mlp(X_train, y_train, X_test, y_test, num_of_epochs=160,plot='losses')

df = pd.DataFrame({
  "Learning_Rate_": [1.0, 0.5, 0.1 , 0.01 ],
  "Loss_": [accuracy1, accuracy2, accuracy3, accuracy4 ]
})
print(df)

#the number of nodes k in the hidden layer with [5, 10, 15, 20, 25, 30] and evaluate the performance of your MLP classifier.
tr_acc=[]
num_of_nodes = [5,10,15,20,25,30]
for m in range(len(num_of_nodes)):
    print(num_of_nodes[m])
    mlp = multilayer_perceptron(num_of_inputs= 30, num_of_hidden = num_of_nodes[m], num_of_outputs = 1, learning_rate = 0.5)
    accuracy=mlp.train_mlp(X_train, y_train, X_test, y_test, num_of_epochs=160)
    tr_acc.append(accuracy)
plt.plot(num_of_nodes,tr_acc)

plt.xlabel("number of nodes k in the hidden layer")
plt.ylabel("Accuracy")
plt.title("the performance of your MLP classifier")

plt.show()
