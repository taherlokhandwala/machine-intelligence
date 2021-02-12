Team : PESU-MI_0274_1292_1485

Implementation: 

We created a neural network that predicts if a patient is a case of LBW or a case of non-LBW.
 
-> Data Preporcessing:
For fields such as 'Delivery phase', 'Education', 'Residence' we have replace NA with mode.
For 'Age','Weight' we have replaced NA with median due to skew.
For 'HB' and 'BP', NA values have been replaced by mean grouped by age.
Due to the machine learning model not knowing what categorical data ( "Community", "Delivery phase", "IFA", "Residence" ) 
is we cannot directly use it. We therefore convert to one hot encoding to reprsent the data.

-> Layers:
We have implemented a neural network with 3 layers.First being the input layer with 15 nodes.The dataset has 9 features, 
but since a few of them are categorical and a few of them are numerical, we have adopted one hot encoding for the categorical 
values and hence we have 15 units.Second layer being the hidden layer with 12 nodes and the final layer i.e the output layer 
with 1 node.

-> Activation Function:
We have used the ReLu activation function as the activation function in the hidden layer.For the output function we have used the 
sigmoid function since we are trying to perform a binary prediction.

-> Loss Function:
Since it is a binary prediction problem we went with the cross-entropy loss function.

The fit function takes 2 parameters: X(input dataset) and Y (labels). First, it saves the train and target to the class variable 
and then initializes the weights and biases by calling the parameters_initialization function. Then, it loops through the 
specified number of iterations, performs forward and backpropagation and saves the loss.

The predict function passes the test data through the forward propagation layer and computes the prediction using the saved 
weights and biases that we had obtained from the training phase.

Hyperparameters:
-> Number of Layer = 3
-> 1st Layer is the input layer with 15 units. The dataset has 9 features, but since a few of them are categorical and a few of 
them are numerical, we have adopted one hot encoding for the categorical values and hence we have 15 units.Second layer i.e the 
hidden layer has 12 units.We found this optimal for our network as more units were overfitting and less units were underfitting.
The final layer i.e the output layer has 1 unit.
->Learning rate = 0.01.Low learning rate slows down the learning process but converges smoothly. 
->Activation Function:
        For the hidden layer we use the ReLu activation function due to its simplicity.
        For the the output function we use sigmoid function since this function is ideal for binary prediction.
->Number of epochs = 50

Steps to Run file:
-> Run the following command :
    python3 PESU-MI_0274_1292_1485.py