import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from optparse import OptionParser
import time

from tqdm import tqdm

from joblib import Parallel, delayed
import multiprocessing


#to import codes from functions folder
import sys
sys.path.append('./Codes')

#machine learning packages
import ml_functions as ml


t_init = time.time() # To measure the time the whole process takes


# ======================= Functions Definitions ======================= #

# Needed in same file beacuse of parallelization


def reshape_matrix(theta, M, n, m1, m2): # Reshapes a 1-D parameter vector in the M+1 parameter matrices for a M+2-layers neural network

    theta_1 = theta[ 0 : int((m1*(n+1))) ]
    theta_1 = theta_1.reshape([m1,n+1])

    if M==1: # just one hidden layer

	    theta_2 = theta[ int((m1*(n+1))) : ]
	    theta_2 = theta_2.reshape([1,m1+1])

	    return theta_1, theta_2

    if M==2: # two hidden layers

	    theta_2 = theta[ int((m1*(n+1))) : int((m1*(n+1))) + int((m2*(m1+1))) ]
	    theta_2 = theta_2.reshape([m2,m1+1])

	    theta_3 = theta[ int((m1*(n+1))) + int((m2*(m1+1)))  : ]
	    theta_3 = theta_3.reshape([1,m2+1])

	    return theta_1, theta_2, theta_3


def sigmoid_hyperbolic_tangent(z): # Activation function: hyperbolic tangent ( see Y. LeCun, 1988 )

    a = 1.7159
    b = 2./3.

    # These two lines in loops are useful to prevent overflows...
    if z < -45:
	    return -a
    elif z > 45:
	    return a

    else:
	    return a * np.tanh(b * z)
sigmoid_hyperbolic_tangent = np.vectorize(sigmoid_hyperbolic_tangent)


def forward_prop_hyp(theta, x, M, n, m1, m2):#, sigmoid): #  Forward propagation to get the hypothesis for a SINGLE TRAINING EXAMPLE of a neural-network

    if M==1: # just one hidden layer
	    Theta_1, Theta_2 = reshape_matrix(theta,M,n,m1,m2)

    if M==2: # two hidden layers
	    Theta_1, Theta_2, Theta_3 = reshape_matrix(theta,M,n,m1,m2)

    # Hidden layer:
    z_2 = np.dot(Theta_1, x)
    a_2 = sigmoid_hyperbolic_tangent(z_2)

    a_2 = np.insert(a_2,0,1.) # Adding the bias unit

    # Output layer (or second layer if M==2):
    z_3 = np.dot(Theta_2, a_2)
    a_3 = sigmoid_hyperbolic_tangent(z_3) # nn hypothesis

    if M == 1:

	    return z_2,a_2,z_3,a_3

    if M == 2:

	    # second hidden layer:
	    a_3 = np.insert(a_3,0,1.) # Adding the bias unit
	    
	    # Output layer:
	    z_4 = np.dot(Theta_3, a_3)
	    a_4 = sigmoid_hyperbolic_tangent(z_4)
	    
	    return z_2,a_2,z_3,a_3,z_4,a_4





def apply_algorithm(X_s, theta_opt, M, n, m1, m2):

    # =============== APPLYING THE ALGORITHM TO GAIA ===================== #

    print '\nApplying the neural network to GAIA data...'
    t_nn = time.time()

    num_cores = multiprocessing.cpu_count() - 2 # number of cores to use for parallelization
    print 'Using %i cores for parallelization' %(num_cores)


    a = Parallel( n_jobs = num_cores)( delayed( forward_prop_hyp ) ( theta_opt, Xi, M, n, m1, m2 ) for Xi in X_s) # Forward propagation to get the nn hypothesis on the TGAS catalogue
    a = np.array(a)

    hyp = a[:,5] # Hypothesis [-1, 1]
    h_temp = ml.map_hyperbolic_tangent(hyp) # mapping the hypothesis into a probability [0,1]
    
    # Set to proper format
    h = []
    for i in range(len(h_temp)):
	    h.append(h_temp[i][-1])
		    
    h = np.array(h)

    print "\nApplying the neural network took %i seconds = %s hours" %(time.time()-t_nn, round((time.time()-t_nn)/60./60. ,3))

    return h
	


def perform_mc_parallel(i, m_best, X_best, X_err, X_mean, X_std, theta_opt, M, n, m1, m2):
	
    # =============== Perform Monte-Carlo ===================== #
    
    Num_MC = 1000 # Number of realizations per star for the Monte Carlo approach
    
    D = np.empty((m_best, Num_MC) ) # D probability (1 = HVS!)
    print 'Performing Monte-Carlo on candidate', i, '/', m_best, '\r',
    
    for j in range(Num_MC): # loop over the different realizations within each candidate uncertainties
	    X_best_new = np.random.normal(X_best[i][1:] , X_err[i] ) # adding a random error within the uncertainty
	    X_best_new = np.insert(X_best_new,0,1.) # Adding the bias unit
	    X_s_best = (X_best_new - X_mean)/X_std # Scaling the feature
	    _,_,_,_,_,D[i][j] =  forward_prop_hyp(theta_opt, X_s_best, M, n, m1, m2) # forward propagation to get the hypothesis
	    D[i][j] = ml.map_hyperbolic_tangent(D[i][j]) # D probability

    return D



def write_output(path_res, filename, indices, m_best, Gaia_best, TYC_best, h_best, X_best, X_err, RA2000_best, DE2000_best, GMag_best, D_mean, D_std):
    
    # =============== Write file with parameters ===================== #
    
    # indices: the indices of the stars to write onto a file
   
    out_file = open(path_res + filename, "w") # Output file

    out_file.write('---------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')
    out_file.write('Gaia-ID   TYC-ID   h   RA_ICRS   e_RA_ICRS   DE_ICRS   e_DE_ICRS   RAJ2000   DEJ2000   Plx   e_Plx   pmRA     e_pmRA    pmDE     e_pmDE     GMag   Dmean   Dstd      \n')
    out_file.write('                       deg       deg         deg       deg         deg       deg       mas   mas     mas/yr   mas/yr    mas/yr   mas/yr     mag                      \n')
    out_file.write('---------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')		

    for i in range(m_best):
	if i in indices or np.size(indices) == m_best:
	    out_file.write(Gaia_best[i] + '|' + TYC_best[i] + '|' + str(h_best[i]) + '|' + str(X_best[i,1] ) + '|' + str(X_err[i,0]) + '|' + str(X_best[i,2]) + '|' + str(X_err[i,1]) + '|' + str(RA2000_best[i]) + '|' + str(DE2000_best[i]) + \
	     '|' + str(X_best[i,3]) + '|' + str(X_err[i,2]) +'|' + str(X_best[i,4]) + '|' + str(X_err[i,3]) + '|' + str(X_best[i,5]) + '|' + str(X_err[i,4]) + \
	     '|' + str(GMag_best[i]) + '|' + str(D_mean[i]) + '|' + str(D_std[i]) + '\n')

    out_file.close()






def main(apply_neural_network, parallel_mc):

    # ================================================ MAIN ========================================================= #


    # ====================== Input parameters to read the corresponding HVS population ============================== #


    M = 2	# Number of hidden layers in the neural network

    m1 = 119	# Number of neurons in the 1st hidden layer (not including the bias term)
    m2 = 95	# Number of neurons in the 2nd hidden layer (not including the bias term) (just if M = 2!)

    eps = 0.	# Threshold parameter for accepting a HVS  (typically 0 in the case of a odd activation function (f(v) = -f(-v)) )

    n = 5	# Number of features (excluding the bias term)
    
     
    # =========== Importing Files ============= #

    print '\nReading in data....'

    path_dat = 'Dataset/'
    path_res = 'Results/'

    # Gaia DR1 data: TGAS
    RA2000, DE2000, RA, sigma_RA, dec, sigma_dec,  par, sigma_par, mu_RA, sigma_mu_RA, mu_dec, sigma_mu_dec, GMag  = np.genfromtxt(path_dat + "TGAS.tsv", skip_header = 70, filling_values= 0,usecols = [0,1,4,5,6,7,8,9,10,11,12,13,14], unpack=True,invalid_raise = False)
    TYC = np.genfromtxt(path_dat + "TGAS.tsv", skip_header = 70, filling_values= 0, usecols = [2], dtype=np.str, unpack=True)
    Gaia = np.genfromtxt(path_dat + "TGAS.tsv", skip_header = 70, filling_values= 0, usecols = [3], dtype=np.str, unpack=True)

    X_mean, X_std = np.loadtxt(path_dat + "Mean_std.txt", unpack=True) # Mean and standard deviation of each feature (including the bias term)

    # Correct units
    RA = RA*np.pi/180. # [rad]
    dec = dec*np.pi/180. # [rad]

    sigma_RA  =  sigma_RA*4.8481368e-9 # [rad]
    sigma_dec = sigma_dec*4.8481368e-9 # [rad]

    

    # ============== Dataset creation: (Feature Matrix) ========================== #

    Num = len(RA) # Length of TGAS catalogue #2057050

    X = np.concatenate((RA, dec, par, mu_RA, mu_dec)) # 5 parameters solution
    X = X.reshape(n,Num) # (5 x len0) Matrix
    X = X.T # Feature Matrix (len0 x 5/6)

    m,n = X.shape # m: number of samples, n: number of features

    X = np.column_stack([np.ones(m), X]) # Adding the "Bias unit" x0 == 1



    # ================= Feature Scaling ===================== #

    X_s = (X - X_mean)/X_std
	
	
	
    print "\nLoading and preparing the data took %i seconds = %s hours" %(time.time()-t_init, round((time.time()-t_init)/60./60. ,3))
    
    
    names = ['7epochs', '8epochs', '15epochs', '80epochs'] # Names to use for saving and loading files	
	
    for name in names:
	print '\nProcessing file:', name
	
	# Load synaptic weights
	if name == 'tommaso' or name == 'replica' or name == 'replica2': # Read from .txt files
	    theta_opt = np.loadtxt(path_res + 'weights_' + name + '.txt', unpack = True) # Optimal synaptic weights
	else: # Read from .npy files
	    results = np.load(path_res + 'results_' + name + '.npy') # Optimal parameters of the NN
	    theta_opt = results[0] # Optimal synaptic weights



	# ================= Get Hypothesis ====================== #

	if apply_neural_network:
	    # Get the hypothesis through the neural network
	    h = apply_algorithm(X_s, theta_opt, M, n, m1, m2)
	    np.save(path_res + 'h_tgas_' + name + '.npy', zip(Gaia, TYC, h))

	else:
	    # Get the hypothesis loading a file
	    print 'Loading data from: h_tgas_' + name + '.npy'
	    load_data = np.load(path_res + 'h_tgas_' + name + '.npy')
	    h = np.array(map(float, load_data[:,2]))




	# ==================== Monte Carlo over the probability just for the Best candidates ====================== #

	print '\nPerforming Monte Carlo...'
	t_mc = time.time()
	
	print 'Number of stars with h > 0.5:',  np.size(np.where(h>0.5))
	
	idx_best = np.where( (h > 0.5) & (np.abs(sigma_par/par)<1) )[-1] # Defining the best candidates!
	m_best = np.size(idx_best) # number of best candidates
	print 'Number of stars with h > 0.5 and (sigma_par/par)<1:', m_best
	 
	
	h_best = h[idx_best]
	X_best = X[idx_best] # Physical Units features

	# Selecting the magnitude and error for the best candidates:
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

	# Error feature matrix:
	X_err = np.concatenate(( sigma_RA_best, sigma_dec_best, sigma_par_best, sigma_mu_RA_best, sigma_mu_dec_best )) # 5 parameters (error) solution
	X_err = X_err.reshape(n,m_best) # (5 x len0) Matrix
	X_err = X_err.T # Feature Matrix (len0 x 5/6)


	if parallel_mc:
	    # Run Monte-Carlo parallel over various cores
	    num_cores = multiprocessing.cpu_count() - 2 # number of cores to use for parallelization
	    print 'Using %i cores for parallelization' %(num_cores)
	    
	    D = Parallel(n_jobs = num_cores)(delayed(perform_mc_parallel) (i, m_best, X_best, X_err, X_mean, X_std, theta_opt, M, n, m1, m2) for i in range(m_best))
	    D = np.array(D)
		
	else:
	    Num_MC = 1000 # Number of realizations per star for the Monte Carlo approach

	    D = np.empty( (m_best, Num_MC) ) # D probability (1 = HVS!)

	    for i in range(m_best): # loop over the best candidates
		print 'Performing Monte-Carlo on candidate', i+1, '/', m_best, '\r',
		for j in range(Num_MC): # loop over the different realizations within each candidate uncertainties
		    X_best_new = np.random.normal(X_best[i][1:] , X_err[i] ) 			# adding a random error within the uncertainty
		    X_best_new = np.insert(X_best_new,0,1.)					# Adding the bias unit
		    X_s_best = (X_best_new - X_mean)/X_std 					# Scaling the feature
		    _,_,_,_,_,D[i][j] =  forward_prop_hyp(theta_opt, X_s_best, M, n, m1, m2) 	# forward propagation to get the hypothesis
		    D[i][j] = ml.map_hyperbolic_tangent(D[i][j])				# D probability

		
		
		
	D_mean = np.mean(D, 1)	# Mean probability for each best candidate
	D_std = np.std(D, 1)	# Standard deviation for each best candidate

	X_best[:,1] = X_best[:,1]*180./np.pi # RA, [deg]
	X_best[:,2] = X_best[:,2]*180./np.pi # dec, [deg]


	# Save best candidates
	filename = 'Dbest_' + name + '.txt'
	write_output(path_res, filename, idx_best, m_best, Gaia_best, TYC_best, h_best, X_best, X_err, RA2000_best, DE2000_best, GMag_best, D_mean, D_std)

	
	print "\nPerforming Monte Carlo took %i seconds = %s hours" %(time.time()-t_mc, round((time.time()-t_mc)/60./60. ,3))





	# =================== Selecting TOP Candidates ! ================================ #


	idx_TOP = np.where(D_mean - D_std > 0.9)[-1] # Defining the top candidates
	
	print '\nNumber of stars with D_mean - D_std > 0.9:', np.size(idx_TOP)
	
	# Save top candidates
	filename = 'top_candidates_' + name + '.txt'
	write_output(path_res, filename, idx_TOP, m_best, Gaia_best, TYC_best, h_best, X_best, X_err, RA2000_best, DE2000_best, GMag_best, D_mean, D_std)
	
	print "\nThe code took %i seconds = %s hours" %(time.time()-t_init, round((time.time()-t_init)/60./60. ,3))











def new_option_parser():
	op = OptionParser()

	# Get hypothesis through neural network or by loading a file
	op.add_option("-n", "--apply_neural_network", default = False, dest="apply_neural_network", help="Get the hypothesis applying the neural network (default: False)", type='string')
	
	# Run Monte-Carlo parallel over various nodes
	op.add_option("-p", "--parallel_mc", default = False, dest="parallel_mc", help="Run Monte-Carlo parallel over various nodes (default: False)", type='string')

	return op
	
	


if __name__ == "__main__":
	
	options, arguments = new_option_parser().parse_args()
	main(**options.__dict__)








