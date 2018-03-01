"""
==============================================================================>
Major master research project
Marco Trueba van den Boom
-------------------------------------------------------------------------------
This script trains a fully connected neural network using Keras with a
Tensorflow backend. The goal of the network is to find hypervelocity stars
in Gaia.
==============================================================================>
"""

# ========================== Import Packages ================================ #

# Standard packages
import numpy as np
from optparse import OptionParser
import time

# Keras (with TensorFlow backend)
from keras import backend as K # TensorFlow
from keras.callbacks import Callback # For loss_history class
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers, initializers, regularizers
from keras.utils.generic_utils import get_custom_objects # For custom tanh activation


# =============================== Core ====================================== #

t_init = time.time() # to measure the time the whole process takes

np.random.seed(2) # For reproducibility




# ======================== Functions Definitions ============================ #

def shuffle(a,b):
    # Shuffles arrays a and b in unison
    
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]



def custom_tanh(z):
    # Activation function: hyperbolic tangent (see Y. LeCun, 1988)

    a = 1.7159
    b = 2./3.

    return a * K.tanh(b * z)

    
    
def map_hypothesis(nn, h):
    
    '''
    DESCRIPTION ---------------------------------------------------------------
    
    Maps the hypothesis into the [0,1] range for a probabilistic interpretation
    of its output.
    
    ATTRIBUTES ----------------------------------------------------------------
    nn --> neural network model
    h  --> hypothesis array
    
    OUTPUT --------------------------------------------------------------------
    
    Mapped hypothesis.
    
    '''    

    def map_tanh(h):
        return 1./2. * (h + 1)
        
    def map_custom_tanh(h):
        return 1./(1.7159 * 2.) * (h + 1.7159)
        
    # The mapping depends on the activation function in the last layer
    if nn.activation_last_layer == 'tanh':
        return map_tanh(h)
    
    elif nn.activation_last_layer == 'custom_tanh':
        return map_custom_tanh(h)
    
    else:
        return h



def change_target_values(activation_last_layer, Y_train, Y_val, Y_test):

    '''
    DESCRIPTION ---------------------------------------------------------------
    
    The target values depend on the activation function. Placed at the point of
    the maximum second derivative.
                
    ATTRIBUTES ----------------------------------------------------------------
    
    activation_last_layer  --> activation function of the output layer
    Y_train, Y_val, Y_test --> target values of the training, cv and test set
    
    OUTPUT --------------------------------------------------------------------
    
    Adapted target values
    
    '''
    
    for data in (Y_train, Y_val, Y_test):
    
        if activation_last_layer == 'tanh':
            data[data == 0] = -0.577
            data[data == 1] = 0.577
        
        elif activation_last_layer == 'custom_tanh': # maximum already at 1
            data[data == 0] = -1
        
        else: # sigmoid logistic function
            data[data == 0] = 0.211
            data[data == 1] = 0.789

    return Y_train, Y_val, Y_test


        
        
    
def load_data(path, percentage):
    
    '''
    DESCRIPTION ---------------------------------------------------------------
    
    Reading the data set. Already feature scaled!
                
    ATTRIBUTES ----------------------------------------------------------------
    
    path       --> directory to load from
    percentage --> percentage of the data to load
    
    OUTPUT --------------------------------------------------------------------
    
    X_train --> features of the training set
    Y_train --> values of the training set (1 for HVS and 0 for normal star)
    X_val --> features of the cross-validation set
    Y_val --> values of the cross-validation set (default: 1 for HVS and 0 otherwise)
    X_test  --> features of the test set
    Y_test  --> values of the test set (default: 1 for HVS and 0 otherwise)
    
    '''

    print '\nReading in data...'
    
    # Training set: labeled data on which the NN learns the classification rule
    X_train = np.load(path + "Training_Set.npy")
    Y_train = np.load(path + "Training_Set_output.npy")
    
    
    # Validation set: labeled data to choose the best values for the hyper-parameters
    X_val = np.load(path + "CrossVal_Set.npy")
    Y_val = np.load(path + "CrossVal_Set_output.npy")

    # Test set: labeled data to evaluate the performance of the algorithm
    X_test = np.load(path + "Test_Set.npy")
    Y_test = np.load(path + "Test_Set_output.npy")
    
    
    # Shuffle randomly the data sets in case only a percentage of the data is used
    X_train, Y_train = shuffle(X_train, Y_train) 
    X_val, Y_val = shuffle(X_val, Y_val)
    X_test, Y_test = shuffle(X_test, Y_test)

    return X_train[:len(X_train)*percentage/100], Y_train[:len(Y_train)*percentage/100], \
           X_val[:len(X_val)*percentage/100], Y_val[:len(Y_val)*percentage/100], \
           X_test[:len(X_test)*percentage/100], Y_test[:len(Y_test)*percentage/100]



    
    
           
# ================================ Classes ================================== #


class loss_history(Callback):
    
    '''
    DESCRIPTION -----------------------------------------------------------
    
    Keras callback to get loss value after every n batches.
                
    ATTRIBUTES ------------------------------------------------------------
    
    n --> number of batches to wait before checking loss
    
    '''
        
    def __init__(self, n):
        self.seen = 0
        self.n = n

    def on_train_begin(self, logs={}):
        self.train = []
        self.test = []

    def on_batch_end(self, batch, logs={}):        
        self.seen += logs.get('size', 0)
        if self.seen % self.n == 0:
            self.train.append(logs.get('loss'))          
            
    def on_epoch_end(self, batch, logs={}):        
            self.test.append(logs.get('val_loss'))             
            
            

# Neural netwrok class
class neural_network(object):
   
    def __init__(self, percentage, neurons, cost_function, activation,
                 activation_last_layer, optimizer, lr, decay, initializer,
                 dropout, regularization, n_epochs, n_batchsize, verbose):
        
        '''        
        DESCRIPTION -----------------------------------------------------------
        
        Initialize variables.
                    
        ATTRIBUTES ------------------------------------------------------------
        
        percentage            --> percentage of the data used for the NN
        neurons               --> array of number of neurons per hidden layer
        cost_function         --> name of the cost (or loss) function
        activation            --> used activation function
        activation_last_layer --> activation function of the last layer
        optimizer             --> name of the optimization method
        lr                    --> learning rate of the optimizer
        decay                 --> learning rate decay over each update
        initializer           --> random initializer of the weights
        dropout               --> dropout rate of hidden layers
        regularization        --> regularization method (None, l1, l2, l1_l2)
        n_epochs              --> number of epochs to train
        n_batchsize           --> number of star per batch
        verbose               --> verbosity level
        
        '''
        
        # Variable from the data
        self.percentage = percentage
        
        # Variables given by the NN (parameters)   
        self.neurons = neurons
        self.cost_function = cost_function
        self.activation = activation
        self.activation_last_layer = activation_last_layer
        self.optimizer = optimizer
        self.lr = lr
        self.decay = decay
        self.initializer = initializer
        self.dropout = dropout
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.n_batchsize = n_batchsize
        
        # Variables created by the class
        self.model = None
        self.P = None
        self.R = None
        self.F1 = None
        self.MCC = None
        
        # Create neural network model in Keras
        self.verbose = verbose
        self.create_model()
       
        
     

    def create_model(self):
        
        '''        
        DESCRIPTION -----------------------------------------------------------
        
        Create the NN model in Keras.
        
        '''
        
        # Prepare activation function (if custom)
        if self.activation == 'custom_tanh' or self.activation_last_layer == 'custom_tanh':
            get_custom_objects().update({'custom_tanh': Activation(custom_tanh)})
        
        
        # Define initializer: LeCun => limit = sqrt(3 * scale / n)
        # Keras default (scale=1.0, mode='fan_in', distribution='normal')
        if self.initializer == 'LeCun':
            initializer = initializers.VarianceScaling(scale=1./3., mode='fan_in', distribution='uniform')    
        else:
            initializer = self.initializer
            
        
        # Select regularization method
        lam = 0.01 # Regularization parameter lambda
        if self.regularization == 'l1':
            regulator = regularizers.l1(lam)
        elif self.regularization == 'l2':
            regulator = regularizers.l2(lam)            
        elif self.regularization == 'l1_l2':
            regulator = regularizers.l1_l2(lam)
        else: # default (no regularization)
            self.regularization = 'None'
            regulator = None
            
        
        # Initialize model
        self.model = Sequential()
                  
        # Add input layer
        self.model.add( Dense(self.neurons[0], kernel_initializer = initializer,
                              input_shape = (5,), activation = self.activation,
                              kernel_regularizer=regulator) )
        
        self.model.add(Dropout(self.dropout))
        
        # Add fully connected (dense) hidden layers
        for layer in range(len(self.neurons)-1):
            self.model.add( Dense(self.neurons[layer], kernel_initializer = initializer,
                                  activation = self.activation,
                                  kernel_regularizer=regulator) )
            
            self.model.add(Dropout(self.dropout))
            
        # Add output layer
        self.model.add( Dense(1, kernel_initializer = initializer,
                         activation = self.activation_last_layer,
                         kernel_regularizer=regulator) )
        
        
        # Define optimization method: Keras default (lr=0.01, decay=0.0, epsilon(if needed)=1e-7)
        if self.optimizer == 'Adagrad':
            optimizer = optimizers.Adagrad(lr=self.lr, decay=self.decay)
        elif self.optimizer == 'SGD':
            optimizer = optimizers.SGD(lr=self.lr, decay=self.decay)
        elif self.optimizer == 'Adam':
            optimizer = optimizers.Adam(lr=self.lr, decay=self.decay, epsilon=1e-8)
        else:
            optimizer = self.optimizer


        # Compile model
        self.model.compile(loss = self.cost_function, optimizer = optimizer)
        

        
        
    def train(self, X_train, Y_train, X_test, Y_test):
        
        '''        
        DESCRIPTION -----------------------------------------------------------
        
        Train the NN.
                    
        ATTRIBUTES ------------------------------------------------------------
        
        X_train --> features of the training set
        Y_train --> values of the training set (1 for HVS and 0 for normal star)
        X_test  --> features of the test set
        Y_test  --> values of the test set (1 for HVS and 0 for normal star)
        
        '''
        
        if self.verbose > 0: print '\nTraining neural network...' 
        
        # Fit the model
        loss = loss_history(n=1000) # check loss every n batches
        self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                            shuffle=True, epochs=self.n_epochs, batch_size=self.n_batchsize,
                            callbacks=[loss], verbose=self.verbose)
        
        return loss
        
    
    def performance_evaluation(self, X_test, Y_test, threshold=0.5):
        
        '''        
        DESCRIPTION -----------------------------------------------------------
        
        Evaluate training with different evaluation methods.
                    
        ATTRIBUTES ------------------------------------------------------------
        
        model  --> Keras model of the NN
        X_test --> features of the test set
        Y_test --> values of the test set (1 for HVS and 0 for normal star)
        eps    --> threshold to accept a hypervelocity star (0 for tanh and 1 for sigmoid)
        
        OUTPUT ----------------------------------------------------------------
        
        TP  --> True positives
        FP  --> False positives
        FN  --> False negatives
        TN  --> True negatives
        P   --> precision
        R   --> recall
        F1  --> F1-score or F-measurement
        MCC --> Matthews Correlation Coefficient
        
        '''
        
        # Calculate confusion matrix
        def confusion_matrix(h, Y):
            # Times 1. to make floats
            TP = 1. * np.size( np.where( (h > threshold) & (Y == max(Y))))  # True positives
            FP = 1. * np.size( np.where( (h > threshold) & (Y == min(Y))))  # False positives
            FN = 1. * np.size( np.where( (h < threshold) & (Y == max(Y))))  # False negatives
            TN = 1. * np.size( np.where( (h < threshold) & (Y == min(Y))))  # True negatives
            
            return TP, FP, FN, TN
        
         
        # Calculate precision
        def precision(TP, FP):
            if TP + FP <= 0:
                return 0
            else:
                return TP / (TP + FP)
        
                
        # Calculate recall
        def recall(TP, FN):
            if TP + FN <= 0:
                return 0
            else:
                return TP / (TP + FN)
            
                
        # Calculate F1-score, [0,1]
        def F1_score(P, R):
            if P + R <= 0:
                return 0
            else:
                return 2.*P*R / (P+R)

        # Calculate Matthews Correlation Coefficient, [-1,1]
        def MCC(TP, FP, FN, TN):
            if (TP+FP) * (TP+FN) * (TN+FP) * (TN+FN) <=0:
                return 0
            else:
                return ( TP*TN - FP*FN ) / np.sqrt( (TP+FP) * (TP+FN) * (TN+FP) * (TN+FN) )
           

         
        # =========== Error analysis and Performance evaluation ============= #
        
        if self.verbose > 0: print '\nRunning performance evaluation...'    
    
        # Predictions on the test set through forward propagation
        hypothesis = self.model.predict(X_test, verbose=self.verbose)
        
        # Flatten arrays for function
        hypothesis = hypothesis.flatten()
        Y_test = Y_test.flatten()
        
        # Map hypothesis to range [0,1]
        hypothesis = map_hypothesis(self, hypothesis)

        TP, FP, FN, TN = confusion_matrix(hypothesis, Y_test)

        self.P = precision(float(TP), float(FP))
        self.R = recall(float(TP), float(FN))

        self.F1  = F1_score(self.P, self.R)
        self.MCC = MCC(TP, FP, FN, TN)
        
        self.TP = int(TP)
        self.FP = int(FP)
        self.FN = int(FN)
        self.TN = int(TN)
        
    

        
        
        
    def save_output(self, run, path_out, loss, t_end):
        
        '''        
        DESCRIPTION -----------------------------------------------------------
        
        Writing parameters and final statistics of the NN to different files.
                    
        ATTRIBUTES ------------------------------------------------------------
        
        run      --> number code of the training
        path_out --> directory to save the files to
        loss     --> class containing the training loss history 
        t_end    --> total time the whole code took
        
        SAVING ----------------------------------------------------------------
        
        nn_parameters   --> parameters of the NN
        weights --> final synaptic weights after the training
        results  --> Cost and accuracy per epoch as a Python dictionary
        
        '''
    
        # Create text to be saved (containing all paramaters of the NN)
        results = "Parameters of the neural network ===================================\n\n\
                   Number of layers: %i\n\
                   Number of neurons per hidden layer: %s\n\
                   Percentage of data used: %i%%\n\
                   ------------------------------------------------\n\
                   Cost function: %s\n\
                   Activation function: %s\n\
                   Activation function in output layer: %s\n\
                   Optimizer: %s\n\
                      >>> Learning rate: %.3f\n\
                      >>> Decay: %.3f\n\
                   Kernel initializer: %s\n\
                   Dropout ratio: %.1f\n\
                   Regularization: %s\n\
                   ------------------------------------------------\n\
                   Number of epochs: %i\n\
                   Stars per batch: %i\n\
                   ------------------------------------------------\n\
                   Jmin on the training set: %f\n\
                   Jmin on the test set: %f\n\
                   TP: %i, FP: %i, FN: %i, TN: %i\n\
                   Precision: %.3f\n\
                   Recall: %.3f\n\
                   F1: %.3f\n\
                   MCC: %.3f\n\
                   ------------------------------------------------\n\
                   Time (hours): %.3f"\
                      %(len(self.neurons)+2, ' '.join(map(str,self.neurons)), self.percentage,
                        self.cost_function, self.activation, self.activation_last_layer,
                        self.optimizer, self.lr, self.decay, self.initializer, self.dropout,
                        self.regularization, self.n_epochs, self.n_batchsize, loss.train[-1], loss.test[-1],
                        self.TP, self.FP, self.FN, self.TN, self.P, self.R, self.F1, self.MCC, t_end)
    
        # Save parameters of the NN
        with open(path_out + 'nn_parameters' + str(run) + '.txt', "w") as out_file:
            out_file.write(results)
            out_file.close()
        
        # Save weights as an .hdf5 file
        self.model.save_weights(path_out + 'weights' + str(run) + '.hdf5')

        # Save loss history (lelijk maar werkt)
        temp = np.full(len(loss.train), np.NaN)
        temp[0:len(loss.test)] = loss.test
        np.savetxt(path_out + 'results' + str(run) + '.txt', zip(loss.train, temp))
        
    
    
    


def main(run, percentage, neurons, cost_function, activation, activation_last_layer,
                 optimizer, lr, decay, initializer, dropout, regularization,
                 n_epochs, n_batchsize, verbose):
    
    '''
    DESCRIPTION ---------------------------------------------------------------
    
    Core code. Runs the neural network, i.e. it trains and saves the data after
    
    
    ATTRIBUTES ----------------------------------------------------------------
    
    run                   --> number code of the training (for saving purposes)
    percentage            --> percentage of the data used for the NN
    neurons               --> array of number of neurons per hidden layer
    cost_function         --> name of the cost (or loss) function
    activation            --> used activation function
    activation_last_layer --> activation function of the last layer
    optimizer             --> name of the optimization method
    lr                    --> learning rate of the optimizer
    decay                 --> learning rate decay over each update
    initializer           --> random initializer of the weights
    dropout               --> hidden layers dropout ratio
    regularization        --> Regularization method (None, l1, l2, l1_l2)
    n_epochs              --> number of epochs to train
    n_batchsize           --> number of star per batch    
  
    '''
    
            
    # ============================= Load data =============================== #
    
    path_dat = '/disks/strw14/TvdB/Master2/Dataset/' # directory from which to load the data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(path_dat, percentage)
    
    # Remove bias unit (not needed for Keras)
    X_train = np.delete(X_train, 0, axis=1)
    X_test = np.delete(X_test, 0, axis=1)
    
       
    
    print '\nTraining with %.5g%% of the data' %percentage
    print '--------------------------------------------------------------------'
    print 'Number of stars in the training set:', len(X_train)
    print 'Number of stars in the test set:', len(X_test)
    
       
       
    
    
    # ============================= Parameters ============================== #
    
    neurons = [int(s) for s in neurons.split() if s.isdigit()] # make array from neurons
    L = len(neurons) + 2                                       # number of layers 
    
    if L == 2: # no hidden layers...
        raise ValueError('Select number of neurons per hidden layer is str format')
            

    
    
    # ========================= Run Neural Network ========================== #    
    
    # Call neural network object
    nn = neural_network(percentage, neurons, cost_function, activation,
                        activation_last_layer, optimizer, lr, decay, initializer,
                        dropout, regularization, n_epochs, n_batchsize, verbose)
    
    
    # Print hyperparamaters
    print '\n=====================  Parameters of the NN ========================'
    print 'Total numbers of layers in the neural network:', L
    print 'Number of neurons per hidden layer:', " ".join(map(str,neurons))
    print '--------------------------------------------------------------------'
    print 'Cost function:', cost_function
    print 'Activation function:', activation
    print 'Activation function in output layer:', activation_last_layer
    print 'Optimizer:', optimizer
    print '   >>> Learning rate:', lr
    print '   >>> Decay:', decay
    print 'Initializer:', initializer
    print 'Dropout ratio:', dropout
    print 'Regularization:', nn.regularization
    print '--------------------------------------------------------------------'
    print "Number of epochs: %i" %(n_epochs)
    print "Batch size: %i" %(n_batchsize)

    


    
    # Adapt target values to activation
    Y_train, Y_val, Y_test = change_target_values(nn.activation_last_layer,
                                                  Y_train, Y_val, Y_test)

    # Start training
    loss = nn.train(X_train, Y_train, X_test, Y_test)

    # Run evaluation of the performance (on the test set!)
    nn.performance_evaluation(X_test, Y_test)
    
    # End values of the Cost function
    Jmin_train = loss.train[-1]
    Jmin_test = loss.test[-1]
   
    # Calculate time needed to compute everything (in hours)
    t_end = round((time.time()-t_init)/60./60., 3)

    # Save parameters and results to files
    path_out = '/disks/strw14/TvdB/Master2/Keras/' # directory to output the files
    nn.save_output(run, path_out, loss, t_end)

   

    # Print results
    print '\n======================== Output Parameters ======================== \n'

    print "Jmin on training set: %.5f" %Jmin_train
    print "Jmin on test set: %.5f" %Jmin_test
    print "------------------------------------------------------------------- "
    print "The code took %.3f hours" %t_end

    print '\n===================== Performance Evaluation  ===================== '
    print 'TP: %i, FP: %i, FN: %i, TN: %i' %(nn.TP, nn.FP, nn.FN, nn.TN)
    print 'Precision = %.3f' %nn.P
    print 'Recall = %.3f' %nn.R
    print 'F1 = %.3f' %nn.F1
    print 'MCC = %.3f' %nn.MCC
    print '=================================================================== '
    

    
    

def new_option_parser():
    op = OptionParser()
    
    # Number code of the training to perform (for saving purposes)
    op.add_option("-r", "--run", default = 0, dest="run",
                  help="Number code of the training to perform (default: 0)", type='int')

    # Percentage of the data to use
    op.add_option("-p", "--percentage", default = 100, dest="percentage",
                  help="Percentage of the data to use (default: 100)", type='int')    

    # Repeating number of times the optimization process over the whole training set
    op.add_option("-e", "--epochs", default = 8, dest="n_epochs",
                  help="Number of epochs (default: 8)", type='int')    

    # Number of stars trained per batch
    op.add_option("-b", "--batch", default = 1, dest="n_batchsize",
                  help="Number of stars trained per batch (default: 1)", type='int')

    # Number of neurons per hidden layer
    op.add_option("-n", "--neurons", default = '119 95', dest="neurons",
                  help="Number of stars trained per batch (default: 119 95)", type='str')
    
    # Cost function of the NN
    op.add_option("-c", "--cost", default = 'binary_crossentropy', dest="cost_function",
                  help="Cost function of the NN (default: binary_crossentropy)", type='str')

    # Activation function
    op.add_option("-a", "--activation", default = 'custom_tanh', dest="activation",
                  help="Activation function (default: custom_tanh)", type='str')

    # Activation function in output layer
    op.add_option("-A", "--activation_last_layer", default = 'sigmoid', dest="activation_last_layer",
                  help="Activation function in output layer (default: sigmoid)", type='str')
    
    # Optimization alghorithm
    op.add_option("-o", "--optimizer", default = 'Adam', dest="optimizer",
                  help="Optimizatiopn alghorithm (default: Adam)", type='str')
        
    # Learning rate for optimizer
    op.add_option("-l", "--lr", default = 0.001, dest="lr",
                  help="Learning rate for optimizer (default: 0.001)", type='float')
 
    # Decay of the learning rate after every update
    op.add_option("-d", "--decay", default = 0.0, dest="decay",
                  help="Decay of the learning rate after every update (default: 0.0)", type='float')
  
    # Random initialization algorithm
    op.add_option("-i", "--initializer", default = 'glorot_normal', dest="initializer",
                  help="Random initialization algorithm (default: glorot_normal)", type='str')
    
    # Hidden layers dropout ratio
    op.add_option("-D", "--dropout", default = 0.0, dest="dropout",
                  help="Hidden layers dropout ratio (default: 0.0)", type='float')

    # Regularization method
    op.add_option("-R", "--regularization", default = 'None', dest="regularization",
                  help="Regularization method (default: None)", type='str')

    # Verbosity level
    op.add_option("-v", "--verbose", default = 1, dest="verbose",
                  help="Verbosity level (default: 1)", type='int')
    
    return op
    
    


if __name__ == "__main__":
    
    options, arguments = new_option_parser().parse_args()
    main(**options.__dict__)


