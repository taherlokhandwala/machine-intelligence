'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark

-> Number of Layers = 3

-> 1st Layer is the input layer with 15 units. The dataset has 9 features, but since a few of them are categorical and a few of them are numerical, we have adopted one hot encoding for the categorical values and hence we have 15 units.Second layer i.e the hidden layer has 12 units.We found this optimal for our network as more units were overfitting and less units were underfitting.The final layer i.e the output layer has 1 unit.

->Learning rate = 0.01.
Low learning rate slows down the learning process but converges smoothly. 

->Activation Function:
        For the hidden layer we use the ReLu activation function due to its simplicity.
        For the the output function we use sigmoid function since this function is ideal for binary classification.

->Number of epochs = 50

'''

# importing necessary libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# defining neural network class


class NN():

    ''' X and Y are dataframes '''

    def __init__(self):
        ''' Initialises all the necessary parameters for the network '''
        self.features = 15
        self.hidden_units = 12
        self.output_units = 1
        self.learning_rate = 0.01
        self.epochs = 50
        np.random.seed(1)
        self.w1 = np.random.randn(
            self.features, self.hidden_units)
        self.w2 = np.random.randn(
            self.hidden_units, self.output_units)
        self.b1 = np.random.randn(self.hidden_units,)
        self.b2 = np.random.randn(self.output_units,)
        self.loss = []
        self.X = None
        self.Y = None
        self.z1 = None
        self.a1 = None

    def activation_relu(self, value):
        # ReLu function
        return np.maximum(0, value)

    def activation_relu_der(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def activation_sigmoid(self, value):
        # Sigmoid function
        return 1.0 / (np.exp(-value) + 1.0)

    def calculate_loss(self, y, yhat):
        ''' Calculates loss using cross entropy function '''
        no_of_samples = len(y)
        loss = -1/no_of_samples * \
            (np.sum(np.multiply((1 - y), np.log(1 - yhat)) + np.multiply(np.log(yhat), y)))
        return loss

    def feed_forward(self):
        '''  Performs forward propagation '''

        z1 = self.X.dot(
            self.w1) + self.b1
        a1 = self.activation_relu(z1)
        z2 = a1.dot(self.w2
                    ) + self.b2
        yhat = self.activation_sigmoid(z2)
        loss = self.calculate_loss(self.Y, yhat)
        self.z1 = z1
        self.a1 = a1

        return (yhat, loss)

    def back_propagation(self, yhat):
        ''' Calculates the derivatives and update weights and bias '''

        diff_wrt_yhat = -(np.divide(self.Y, yhat) -
                          np.divide((1 - self.Y), (1-yhat)))
        diff_wrt_sigmoid = yhat * (1-yhat)
        diff_wrt_z2 = diff_wrt_yhat * diff_wrt_sigmoid

        diff_wrt_a1 = diff_wrt_z2.dot(self.w2.T)
        diff_wrt_w2 = self.a1.T.dot(diff_wrt_z2)
        diff_wrt_b2 = np.sum(diff_wrt_z2, axis=0)

        diff_wrt_z1 = diff_wrt_a1 * \
            self.activation_relu_der(self.z1)
        diff_wrt_w1 = self.X.T.dot(diff_wrt_z1)
        diff_wrt_b1 = np.sum(diff_wrt_z1, axis=0)

        self.w1 = self.w1 - \
            self.learning_rate * diff_wrt_w1
        self.w2 = self.w2 - \
            self.learning_rate * diff_wrt_w2
        self.b1 = self.b1 - \
            self.learning_rate * diff_wrt_b1
        self.b2 = self.b2 - \
            self.learning_rate * diff_wrt_b2

    def fit(self, X, Y):
        '''
        Function that trains the neural network by taking x_train and y_train samples as input
        '''
        self.X = X
        self.Y = Y

        for _ in range(self.epochs):
            yhat, loss = self.feed_forward()
            self.back_propagation(yhat)
            self.loss.append(loss)
            print("Epoch:%d   Loss: %f" % (_, loss))

    def predict(self, X):
        """
        The predict function performs a simple feed forward of weights
        and outputs yhat values 

        yhat is a list of the predicted value for df X
        """
        z1 = X.dot(self.w1) + self.b1
        a1 = self.activation_relu(z1)
        z2 = a1.dot(self.w2
                    ) + self.b2
        yhat = np.round(self.activation_sigmoid(z2))
        return yhat

    def CM(self, y_test, y_test_obs):
        '''
        Prints confusion matrix 
        y_test is list of y values in the test dataset
        y_test_obs is list of y values predicted by the model

        '''

        for i in range(len(y_test_obs)):
            if(y_test_obs[i] > 0.6):
                y_test_obs[i] = 1
            else:
                y_test_obs[i] = 0

        cm = [[0, 0], [0, 0]]
        fp = 0
        fn = 0
        tp = 0
        tn = 0

        for i in range(len(y_test)):
            if(y_test[i] == 1 and y_test_obs[i] == 1):
                tp = tp+1
            if(y_test[i] == 0 and y_test_obs[i] == 0):
                tn = tn+1
            if(y_test[i] == 1 and y_test_obs[i] == 0):
                fp = fp+1
            if(y_test[i] == 0 and y_test_obs[i] == 1):
                fn = fn+1
        cm[0][0] = tn
        cm[0][1] = fp
        cm[1][0] = fn
        cm[1][1] = tp

        if tp+fp <= 0:
            print("Change seed value to randomise test data.Also change the hyperparameters to avoid vanishing gradient")
            return

        p = tp/(tp+fp)
        r = tp/(tp+fn)
        f1 = (2*p*r)/(p+r)

        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")

    def calculate_accuracy(self, y, yhat):
        ''' Calculates accuracy of the network '''

        accuracy = float(sum(y == yhat) / len(y) * 100)
        return accuracy


# Read Dataset
df = pd.read_csv(r"LBW_Dataset_Cleaned.csv")

# Seperating features and target
x = df.drop(columns=['Result'])
y = df['Result']
y = y.values.reshape(x.shape[0], 1)

# Splitting dataset in train and test data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1, shuffle=True)

# Standardising data
ssc = StandardScaler()
ssc.fit(x_train)
x_test = ssc.transform(x_test)
x_train = ssc.transform(x_train)

# Passing train data through the network
neural_network = NN()
neural_network.fit(x_train, y_train)

# Predcting values
y_train_pred = neural_network.predict(x_train)
y_test_pred = neural_network.predict(x_test)

# Finding accuracy of the network
print("\nTraining accuracy is : ",
      neural_network.calculate_accuracy(y_train, y_train_pred))
print("Test accuracy is : ", neural_network.calculate_accuracy(y_test, y_test_pred))

neural_network.CM(y_test, y_test_pred)
