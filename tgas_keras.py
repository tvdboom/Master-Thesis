"""
==============================================================================>
Major master research project
Marco Trueba van den Boom
-------------------------------------------------------------------------------
Applies the fully trained NN onto GAIA DR2 (previously on DR1). Can be parallelized.
Also performs a Monte-Carlo, drawing 1000 hypothesis of every star changing the initial
features with a normal distribution centered at their value and a std of their error,
and calculating the mean of the hypothesis. This is done to take into account errors in
the features, especially those in the parallax.
==============================================================================>
"""

# ========================== Import Packages ================================ #

# Standard packages
import numpy as np
from optparse import OptionParser
import time

# Import nn class and mapping function from training file
from nn_keras import neural_network, map_hypothesis

# For paralellization
from joblib import Parallel, delayed
import multiprocessing



# =============================== Core ====================================== #

t_init = time.time() # to measure the time the whole process takes




# ======================== Functions Definitions ============================ #

def feature_scaling(X, mean, std):
    
    '''
    DESCRIPTION ---------------------------------------------------------------
    
    Apply feature scaling.
    
    ATTRIBUTES ----------------------------------------------------------------
    X    --> unscaled features
    mean --> mean of the features in the training set
    std  --> standard deviation of the features in the training set
    
    OUTPUT --------------------------------------------------------------------
    
    Scaled features.
    
    '''
    
    return np.divide(np.subtract(X, mean), std)
    
    
    
def perform_mc(star, nn, n_mc, m_best, X_best, X_err, mean, std):
    
    '''
    DESCRIPTION ---------------------------------------------------------------
    
    Calculate various hypothesis for a single best candidate star through a
    Monte-Carlo approach. This is done to take into account the erros in the
    features, especially the large errors in parallax.
    
    ATTRIBUTES ----------------------------------------------------------------
   
    nn     --> neural network model
    star   --> star to perform MC on
    n_mc   --> number of hypothesis realizations per star
    m_best --> number of stars selected as best candidates
    X_best --> five initial features of the best candidates
    X_err  --> error in the features
    mean --> mean of the features in the training set (for scaling)
    std  --> standard deviation of the features in the training set (for scaling)
    
    OUTPUT --------------------------------------------------------------------
    
    Array of the hypothesis calculated for a single star.
    
    '''
    
    print 'Performing Monte-Carlo on candidate', star, '/', m_best, '\r',
    
    # Create new features
    e_RA = [X_best[star, 0] for i in range(n_mc)]
    e_dec = [X_best[star, 1] for i in range(n_mc)]
    e_plx = np.random.normal(X_best[star,2] , X_err[star,2], size = n_mc)
    e_mu_RA = np.random.normal(X_best[star,3] , X_err[star,3], size = n_mc)
    e_mu_dec = np.random.normal(X_best[star,4] , X_err[star,4], size = n_mc)
    
    # Reshape matrix to correct format for NN
    X_new = np.concatenate((e_RA, e_dec, e_plx, e_mu_RA, e_mu_dec)) # 5 parameters new solution
    X_new = X_new.reshape((5, n_mc))                                # (features x n_montecarlo) matrix
    X_new = X_new.T                                                 # (n_montecarlo x features) matrix
    X_new = feature_scaling(X_new, mean, std)                       # Scaling the features

    # Make and return predictions through forward propagation
    hypothesis = nn.model.predict(X_new, verbose=0).flatten()

    # Make sure hypothesis is always in range [0,1]
    hypothesis = map_hypothesis(nn, hypothesis)
    
    return hypothesis
    

def write_output(path_res, filename, indices, m_best, Gaia_best, TYC_best, h_best,
                 X_best, X_err, RA2000_best, DE2000_best, GMag_best, D_mean, D_std):
    
    '''
    DESCRIPTION ---------------------------------------------------------------
    
    Write file with parameters.
    
    ATTRIBUTES ----------------------------------------------------------------
    
    path_res    --> directory to save the files to
    filename    --> name of the file ton save
    indices     --> indices to enumerate the files
    m_best      --> number of stars selected as best candidates
    Gaia_best   --> Gaia name of the best candidates
    TYC_best    --> Tycho name of the best candidates
    h_best      --> hypothesis of the best candidates
    X_best      --> five initial features of the best candidates
    X_err       --> error in teh features
    RA2000_best --> right ascension in 2000 catalog
    DE2000_best --> declination in 2000 catalog
    GMag_best   --> G magnitude of best candidates
    D_mean      --> mean of the hypothesis
    D_std       --> standard deviation of the hypothesis
    
    '''
       
    out_file = open(path_res + filename, "w") # Output file

    out_file.write('---------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
    out_file.write('Gaia-ID   TYC-ID   h   RA_ICRS   e_RA_ICRS   DE_ICRS   e_DE_ICRS   RAJ2000   DEJ2000   Plx   e_Plx   pmRA     e_pmRA    pmDE     e_pmDE     GMag   Dmean   Dstd      \n')
    out_file.write('                       deg       deg         deg       deg         deg       deg       mas   mas     mas/yr   mas/yr    mas/yr   mas/yr     mag                      \n')
    out_file.write('---------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')        

    for i in range(m_best):
        if i in indices or np.size(indices) == m_best:
            out_file.write(str(Gaia_best[i]) + '|' + str(TYC_best[i]) + '|' + str(h_best[i]) + \
            '|' + str(X_best[i,0] ) + '|' + str(X_err[i,0]) + '|' + str(X_best[i,1]) + \
            '|' + str(X_err[i,1]) + '|' + str(RA2000_best[i]) + '|' + str(DE2000_best[i]) + \
             '|' + str(X_best[i,2]) + '|' + str(X_err[i,2]) +'|' + str(X_best[i,3]) + \
             '|' + str(X_err[i,3]) + '|' + str(X_best[i,4]) + '|' + str(X_err[i,4]) + \
             '|' + str(GMag_best[i]) + '|' + str(D_mean[i]) + '|' + str(D_std[i]) + '\n')

    out_file.close()






def main(run, apply_neural_network, parallel_mc):
    
    '''
    DESCRIPTION ---------------------------------------------------------------
    
    MAIN
    
    ATTRIBUTES ----------------------------------------------------------------
    
    run                            --> number code of file to process
    apply_neural_network (boolean) --> True: Apply the NN to get the hypothesis
                                       False: Read the hypothesis from a file
    parallel_mc (boolean)          --> True: Perform MC in parallel
                                       False: Do not perform MC in parallel                                
    
    '''
    
     
    # =========================== Importing data ============================ #

    print '\nReading in data....'

    path_dat = '/disks/strw14/TvdB/Master2/Dataset/'
    path_res = '/disks/strw14/TvdB/Master2/Keras/'

    # Gaia DR1/TGAS data
    RA2000, DE2000, RA, sigma_RA, dec, sigma_dec,  par, sigma_par, \
    mu_RA, sigma_mu_RA, mu_dec, sigma_mu_dec, GMag  = \
                                np.genfromtxt(path_dat + "TGAS.tsv",
                                              skip_header = 70,
                                              filling_values = 0,
                                              usecols = [0,1,4,5,6,7,8,9,10,11,12,13,14],
                                              unpack = True,
                                              invalid_raise = False)
    
    # Read names separated because of strings 
    TYC, Gaia = np.genfromtxt(path_dat + "TGAS.tsv", skip_header = 70,
                              filling_values= 0, usecols = [2,3], dtype=np.str, unpack=True)
    
    # Mean and standard deviation of each feature for scaling
    mean, std = np.loadtxt(path_dat + "Mean_std_nobias.txt", unpack=True)
    
    # Correct units
    RA = RA * np.pi/180. # [rad]
    dec = dec * np.pi/180. # [rad]

    sigma_RA  =  sigma_RA * 4.8481368e-9 # [rad]
    sigma_dec = sigma_dec * 4.8481368e-9 # [rad]



    
    # ================= Dataset creation: (Feature Matrix) ================== #

    m = len(RA)  # length of TGAS catalogue: 2057050
    n = 5        # number of features (excluding the bias term)
    
    # Create feature matrix
    X = np.concatenate((RA, dec, par, mu_RA, mu_dec)) # 5 parameters solution
    X = X.reshape((n, m))                             # (features x len(Gaia)) matrix
    X = X.T                                           # (len(Gaia) x features) matrix
    X_s = feature_scaling(X, mean, std)               # Apply feature scaling

   
    
    print "\nLoading and preparing the data took %i seconds = %s minutes"\
                %(time.time() - t_init, round((time.time()-t_init)/60., 3))
    
    
    # If stated specicific file to process do that, else list of files
    if run == str(9999):
        # Number codes to use for files to process
        names = ['1', '2', '3']
    else:
        names = [run]
     
   
    # Loop over all files
    for name in names:
        print '\nProcessing file:', name\
        
        
        # ========================= Load NN model =========================== # 
        
        # Load parameters of the NN
        neurons, percentage, cost_function, activation, activation_last_layer, optimizer, \
        lr, decay, initializer, dropout, regularization, n_epochs, n_batchsize = \
                                    np.genfromtxt(path_res + 'nn_parameters' + name + '.txt',
                                                  dtype = None,
                                                  skip_header = 3,
                                                  skip_footer = 8,
                                                  delimiter = ': ',
                                                  comments = '-',
                                                  unpack = True,
                                                  usecols = 1)
        

        # Set correct variable formats (default string)
        neurons = [int(s) for s in neurons.split() if s.isdigit()] # make array from neurons
        lr = lr.astype(float)
        decay = decay.astype(float)
        dropout = dropout.astype(float)
        n_epochs = n_epochs.astype(int)                           
        n_batchsize = n_batchsize.astype(int)
        
        L = len(neurons) + 2 # total number of layers
        if L == 2:           # if no hidden layers raise an error
            raise ValueError('Select number of neurons per hidden layer is str format')
                
      
        # Print parameters of the NN that is being used
        print '\n=====================  Parameters of the NN ========================'
        print 'Total numbers of layers in the neural network:', L
        print 'Number of neurons in per hidden layer:', " ".join(map(str,neurons))
        print '--------------------------------------------------------------------'
        print 'Cost function:', cost_function
        print 'Activation function:', activation
        print 'Activation function in output layer:', activation_last_layer
        print 'Optimizer:', optimizer
        print '   >>> Learning rate:', lr
        print '   >>> Decay:', decay
        print 'Initializer:', initializer
        print 'Dropout ratio:', dropout
        print 'Regularization method:', regularization
        print '--------------------------------------------------------------------'
        print "Number of epochs: %i" %n_epochs
        print "Batch size: %i" %n_batchsize
        
        
        
        # Call neural network object
        nn = neural_network(percentage, neurons, cost_function, activation, activation_last_layer,
                            optimizer, lr, decay, initializer, dropout, regularization,
                            n_epochs, n_batchsize, verbose=1)
    
        # Load synaptic weights
        nn.model.load_weights(path_res + 'weights' + name + '.hdf5', by_name=False)
        
        
        

        # ===================== Get hypothesis of Gaia ====================== #

        if apply_neural_network: # Get the hypothesis through forward propagation
            
            print '\nPropagating forward...'
             
            # Predictions on the gaia data through forward propagation
            hypothesis = nn.model.predict(X_s, verbose=1).flatten()
            
            # Make sure hypothesis is always in range [0,1]
            hypothesis = map_hypothesis(nn, hypothesis)
            
            np.save(path_res + 'h_tgas' + name + '.npy', zip(Gaia, TYC, hypothesis))
        
        else: # Get the hypothesis loading a file
            
            print '\nLoading data from: h_tgas' + name + '.npy'
            
            data = np.load(path_res + 'h_tgas' + name + '.npy')
            hypothesis = np.array(map(float, data[:,2]))



        # ============= Analyze and prepare data for Monte-Carlo ============ #

        print '\nNumber of stars with hypothesis > 0.5:',  np.size(np.where(hypothesis > 0.5))
        
        # Defining the best candidates
        idx_best = np.where( (hypothesis > 0.5) & (np.abs(sigma_par/par) < 1) )[-1]
        m_best = np.size(idx_best) # number of best candidates
        
        print 'Number of stars with hypothesis > 0.5 and (sigma_par/par) < 1:', m_best
         
        
        h_best = hypothesis[idx_best]
        X_best = X[idx_best] # Physical Units features (not scaled!!)

        # Selecting the magnitude and errors for the best candidates
        Gaia_best = Gaia[idx_best]
        TYC_best = TYC[idx_best]
        GMag_best = GMag[idx_best]
        RA2000_best = RA2000[idx_best]
        DE2000_best = DE2000[idx_best]
        sigma_RA_best = sigma_RA[idx_best]
        sigma_dec_best = sigma_dec[idx_best]
        sigma_par_best = sigma_par[idx_best]
        sigma_mu_RA_best = sigma_mu_RA[idx_best]
        sigma_mu_dec_best = sigma_mu_dec[idx_best]

        # Error feature matrix
        X_err = np.concatenate((sigma_RA_best, sigma_dec_best, sigma_par_best,
                                sigma_mu_RA_best, sigma_mu_dec_best)) # 5 parameters error solution
        X_err = X_err.reshape(n, m_best)                              # (features x len(best_candidates)) matrix
        X_err = X_err.T                                               # transpose 

        
        # ===== Monte-Carlo over the probability for the best candidates ==== #
        
        print '\nRunning Monte-Carlo error calculations...'
        
        t_mc = time.time()
        
        n_mc = 1000 # Number of realizations per star for the Monte Carlo approach

        D = [] # Assign space in memory
        
        if parallel_mc: # Run Monte-Carlo over various cores in parallel
            
            '''
            pickle.PicklingError: Can't pickle <type 'module'>: it's not found as __builtin__.module
            '''
            
            num_cores =  multiprocessing.cpu_count() - 2 # number of cores to use for parallelization
            print 'Using %i cores for parallelization' %num_cores
            
            D = Parallel(n_jobs = num_cores)(delayed(perform_mc) \
                         (star, nn, n_mc, m_best, X_best, X_err, mean, std) for star in range(m_best))
            D = np.array(D)
        
        else: # Run Monte-Carlo on a single core
           
            for star in range(m_best): # loop over the best candidates
                D.append(perform_mc(star, nn, n_mc, m_best, X_best, X_err, mean, std))
            
                
                
        D = np.array(D)
        D_mean = np.mean(D, 1) # Mean probability for each best candidate
        D_std = np.std(D, 1)   # Standard deviation for each best candidate
        
      
        X_best[:,0] = X_best[:,0]*180./np.pi  # RA, [deg]
        X_best[:,1] = X_best[:,1]*180./np.pi  # dec, [deg]


        # Save best candidates
        filename = 'Dbest' + name + '.txt'
        write_output(path_res, filename, idx_best, m_best, Gaia_best, TYC_best,
                     h_best, X_best, X_err, RA2000_best, DE2000_best, GMag_best, D_mean, D_std)

        
        print "\nPerforming Monte-Carlo took %i seconds = %s hours"\
                    %(time.time()-t_mc, round((time.time()-t_mc)/60./60. ,3))





        # ==================== Selecting TOP Candidates ===================== #


        idx_TOP = np.where(D_mean - D_std > 0.9)[-1] # Defining the top candidates
        
        print '\nNumber of stars with D_mean - D_std > 0.9:', np.size(idx_TOP)
        
        # Save top candidates
        filename = 'top_candidates' + name + '.txt'
        write_output(path_res, filename, idx_TOP, m_best, Gaia_best, TYC_best,
                     h_best, X_best, X_err, RA2000_best, DE2000_best, GMag_best, D_mean, D_std)
        
        
        
        # =================================================================== #
                
        print "\nThe code took %i seconds = %s hours"\
                %(time.time()-t_init, round((time.time()-t_init)/60./60. ,3))




                

def new_option_parser():
    op = OptionParser()

    # Code of file to process
    op.add_option("-r", "--run", default = '9999', dest="run",
                  help="Number code of file to process (default: 0)", type='string')
    
    # Get hypothesis through neural network or by loading a file
    op.add_option("-n", "--apply_neural_network", default = False, dest="apply_neural_network",
                  help="Get the hypothesis applying the neural network (default: False)", type='string')
    
    # Run Monte-Carlo parallel over various nodes
    op.add_option("-p", "--parallel_mc", default = False, dest="parallel_mc",
                  help="Run Monte-Carlo parallel over various nodes (default: False)", type='string')

    return op
    
    


if __name__ == "__main__":
    
    options, arguments = new_option_parser().parse_args()
    main(**options.__dict__)








