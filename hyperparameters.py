"""
==============================================================================>
Major master research project
Marco Trueba van den Boom
-------------------------------------------------------------------------------
Script to select the optimal hyperparameters of the neural network using a
Bayesian Optimization approach implemented with the GPyOpt library.
==============================================================================>
"""

# ========================== Import Packages ================================ #

# Standard packages
import numpy as np
from optparse import OptionParser
import time

# Bayesian optimization package
import GPyOpt

# Neural network class and useful functions
from nn_keras import neural_network, load_data, change_target_values



# =============================== Core ====================================== #

t_init = time.time() # to measure the time the whole process takes

np.random.seed(2) # For reproducibility



# ================================ MAIN ===================================== #

def main(run, percentage, max_iter):
                   
    # ============================= Load data =============================== #
    
    catalog = 1
    path_dat = '/disks/strw14/TvdB/Master2/Dataset/Omar/' # directory from which to load the data
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_data(path_dat, percentage)
    
    if catalog != 1:
        # Remove bias unit (not needed for Keras)
        X_train = np.delete(X_train, 0, axis=1)
        X_val = np.delete(X_val, 0, axis=1)
        X_test = np.delete(X_test, 0, axis=1)

    
    # Adapt target values to activation in last layer
    Y_train, Y_val, Y_test = change_target_values('sigmoid', Y_train, Y_val, Y_test)
       
    print '\nRun:', run
    print 'Maximum number of iterations: %i' %max_iter
    print 'Optimizing with %.5g%% of the data' %percentage
    print '--------------------------------------------------------------------'
    print 'Number of stars in the cross-validation set:', len(X_val)
    print 'Number of stars in the test set:', len(X_test)          




    # ======================= Bayesian Optimization ========================= #


    # Function to optimize
    def optimize(x):
                
        lr = float(x[:,0])
        n_layers = int(x[:,1])
        dropout = float(x[:,7])
        n_batchsize = int(x[:,8])
        n_epochs = int(x[:,9])
     
        
        # Prepare neurons per hidden layer
        neurons = []
        for n in range(n_layers):
            neurons.append(int(x[:,n+2][-1]))
   
        
        # Print training stats
        print '\n============= Hyperparameters ================='
        print 'Number of hidden layers: %i' %n_layers
        print 'Neurons per hidden layer:', neurons
        print 'Learning rate: %f' %lr
        print 'Dropout ratio: %.1f' %dropout
        print 'Batch size: %i' %n_batchsize
        print 'Number of epochs: %i' %n_epochs
        print '---------------------------------'
             
    
        # Call neural network object
        nn = neural_network(percentage=100, neurons=neurons, cost_function='binary_crossentropy',
                            activation='custom_tanh', activation_last_layer='sigmoid',
                            optimizer='Adam', lr=lr, decay=0.0,
                            initializer='glorot_uniform', dropout=dropout, regularization='None',
                            n_epochs=n_epochs, n_batchsize=n_batchsize, verbose=0)
        
        # Start training
        nn.train(X_val, Y_val, X_test, Y_test)
    
        # Run evaluation of the performance (on the test set!)
        nn.performance_evaluation(run, X_test, Y_test, make_plot=False)

        print 'MCC:', nn.MCC
                
        return nn.MCC

        
        
        
    # ============================ Implementation =========================== #
    
    
    # Bounds for hyperparameters, dict should be in order of continuous and then discrete types
    domain = [{'name': 'lr', 'type': 'continuous', 'domain': (0.00001,0.001)},
              {'name': 'layers', 'type': 'discrete', 'domain': (2,3,4)},
              {'name': 'n1', 'type': 'discrete', 'domain': (32, 64, 128, 256, 512)},
              {'name': 'n2', 'type': 'discrete', 'domain': (32, 64, 128, 256, 512)},
              {'name': 'n3', 'type': 'discrete', 'domain': (32, 64, 128, 256, 512)},
              {'name': 'n4', 'type': 'discrete', 'domain': (32, 64, 128, 256, 512)},
              {'name': 'n5', 'type': 'discrete', 'domain': (32, 64, 128, 256, 512)},
              {'name': 'dropout', 'type': 'discrete', 'domain': (0.0, 0.1)},
              {'name': 'batch_size', 'type': 'discrete', 'domain': (1, 8, 16, 32)},
              {'name': 'epochs', 'type': 'discrete', 'domain': (np.arange(8,15,1))}]
    
    '''           
    # Initial values for the trials
    init_values = np.array([[0.0001, 3, 512, 512, 128, 32, 32, 0.0, 8, 8],
                            [0.0001, 3, 512, 512, 128, 32, 32, 0.1, 8, 10],
                            [0.0001, 4, 128, 128, 128, 64, 32, 0.1, 8, 12],
                            [0.0001, 4, 256, 256, 128, 128, 32, 0.1, 8, 8],
                            [0.0001, 5, 256, 128, 128, 64, 64, 0.1, 8, 8]])
    '''           
               
               
               
    # Running optimization
    opt = GPyOpt.methods.BayesianOptimization(f=optimize, domain=domain,
                                              batch_size=1,
                                              num_cores=1,
                                              maximize=True,
                                              #initial_design_numdata = 5,
                                              #X = init_values,
                                              normalize_Y=False,
                                              verbosity=True)
    
    opt.run_optimization(max_iter=max_iter, verbosity=True, report_file='BO' + str(run) + '.txt')
    
    
    
    
    # ============================ Finish up ================================ #
        
    # Calculate time needed to compute everything (in hours)
    t_end = round((time.time()-t_init)/60./60., 3)
   
    
    # Print stats
    print '\n============= Final Statistics ================='
    print 'Optimal Learning rate: %.6f' %opt.x_opt[0]
    print 'Optimal number of hidden layers:', int(opt.x_opt[1])
    print 'Optimal number of neurons:', [int(opt.x_opt[n+2]) for n in range(int(opt.x_opt[1]))]
    print 'Optimal dropout ratio:', int(opt.x_opt[7])               
    print 'Optimal batch size:', int(opt.x_opt[8])               
    print 'Optimal number of epochs:', int(opt.x_opt[9])                                         
    print '-------------------------------------------------'
    print 'Optimal MCC: %.3f' %-opt.fx_opt # Min beacuse it automatically calculates the minimum
    print '-------------------------------------------------'
    print "The code took %.3f hours" %t_end


    

def new_option_parser():
    op = OptionParser()
    
    # Run
    op.add_option("-r", "--run", default = 0, dest="run",
                  help="Run (default: 0)", type='int')

    # Percentage of the data to use
    op.add_option("-p", "--percentage", default = 100, dest="percentage",
                  help="Percentage of the data to use (default: 100)", type='int')

    # Number code of the training to perform (for saving purposes)
    op.add_option("-i", "--iterations", default = 5, dest="max_iter",
                  help="Maximum number of iterations (default: 5)", type='int')
    
    return op
    
    


if __name__ == "__main__":
    
    options, arguments = new_option_parser().parse_args()
    main(**options.__dict__)
