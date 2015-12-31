import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from scipy.optimize import minimize


"""
	This dataset consists of ratings on a scale of 1 to 5, Y is a (1682 x 943) (num_movies x num_users) matrix, and 
	R is the (1682 x 943) matrix R(i,j)=1 mean user j has rated movie i
"""
movies_data = loadmat('data/movies.mat')

Y = movies_data['Y'] #(1682,943)
R = movies_data['R'] #(1682,943)

"""
parameters  theta and X
"""
params_data = loadmat('data/movieParams.mat') 
X = params_data['X']  #(1682, 10)
Theta = params_data['Theta'] #(943, 10)

"""
	Movies names this file contain two rows first is sequence number and second the the name of the movie 
	we are here reading on the name of the movies
"""

movies_name = {}
f = open('data/movie_ids.txt')
for line in f:
    tokens = line.split(' ')
    tokens[-1] = tokens[-1][:-1]
    movies_name[int(tokens[0]) - 1] = ' '.join(tokens[1:])

#len(movies_name)

"""
	Computing the Cost Function and Gradients with added Regularization, 
	Cost Function for Collaborative Filtering Intuitively, the "Cost" is 
	the degree to which a set of Movie Rating Rredictions deviate from the true Predictions. 
"""
def comput_cost_grad(params, Y, R, num_features, learning_rate):
    Y = np.matrix(Y)  # (1682, 943)
    R = np.matrix(R)  # (1682, 943)
    num_movies = Y.shape[0] ## number of movies 
    num_users = Y.shape[1]  ## number of users
    
    # reshape the parameter array into parameter matrices
    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))  # (1682, 10)
    Theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))  # (943, 10)
    
    # initializations 
    J_theta = 0
    X_grad = np.zeros(X.shape)  # (1682, 10)
    Theta_grad = np.zeros(Theta.shape)  # (943, 10)
    
    # compute the cost
    error = np.multiply((X * Theta.T) - Y, R)  # (1682, 943)
    squared_error = np.power(error, 2)  # (1682, 943)
    J_theta = (1. / 2) * np.sum(squared_error)
    
    # add regularization
    J_theta = J_theta + ((learning_rate / 2) * np.sum(np.power(Theta, 2)))
    J_theta = J_theta + ((learning_rate / 2) * np.sum(np.power(X, 2)))
    
    # calculate the gradients with regularization
    X_grad = (error * Theta) + (learning_rate * X)
    Theta_grad = (error.T * X) + (learning_rate * Theta)
    
    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))
    
    return J_theta, grad

"""
  We're creating our own movie ratings so we can use the model to generate personalized recommendations.
"""
my_rating = np.zeros((1682, 1))

my_rating[0] = 4
my_rating[6] = 3
my_rating[11] = 5
my_rating[53] = 4
my_rating[63] = 5
my_rating[65] = 3
my_rating[68] = 5
my_rating[97] = 2
my_rating[182] = 4
my_rating[225] = 5
my_rating[354] = 5
my_rating[374] = 5
my_rating[382] = 3
my_rating[388] = 4
my_rating[393] = 2
my_rating[405] = 5
my_rating[454] = 1

print('Rated {0} with {1} stars.'.format(movies_name[0], str(int(my_rating[0]))))
print('Rated {0} with {1} stars.'.format(movies_name[6], str(int(my_rating[6]))))
print('Rated {0} with {1} stars.'.format(movies_name[11], str(int(my_rating[11]))))
print('Rated {0} with {1} stars.'.format(movies_name[53], str(int(my_rating[53]))))
print('Rated {0} with {1} stars.'.format(movies_name[63], str(int(my_rating[63]))))
print('Rated {0} with {1} stars.'.format(movies_name[65], str(int(my_rating[65]))))
print('Rated {0} with {1} stars.'.format(movies_name[68], str(int(my_rating[68]))))
print('Rated {0} with {1} stars.'.format(movies_name[97], str(int(my_rating[97]))))
print('Rated {0} with {1} stars.'.format(movies_name[182], str(int(my_rating[182]))))
print('Rated {0} with {1} stars.'.format(movies_name[225], str(int(my_rating[225]))))
print('Rated {0} with {1} stars.'.format(movies_name[354], str(int(my_rating[354]))))
print('Rated {0} with {1} stars.'.format(movies_name[374], str(int(my_rating[374]))))
print('Rated {0} with {1} stars.'.format(movies_name[382], str(int(my_rating[382]))))
print('Rated {0} with {1} stars.'.format(movies_name[388], str(int(my_rating[388]))))
print('Rated {0} with {1} stars.'.format(movies_name[393], str(int(my_rating[393]))))
print('Rated {0} with {1} stars.'.format(movies_name[405], str(int(my_rating[405]))))
print('Rated {0} with {1} stars.'.format(movies_name[454], str(int(my_rating[454]))))

R = movies_data['R']
Y = movies_data['Y']

"""
	We can add our own ratings vector to the existing data set to include in the model.
"""

Y = np.append(Y, my_rating, axis=1) 
R = np.append(R, my_rating != 0, axis=1) 


movies = Y.shape[0]  # 1682
users = Y.shape[1]  # 944
features = 10 
learning_rate = 10. # we can change the learning parameter

X = np.random.random(size=(movies, features))
Theta = np.random.random(size=(users, features))
params = np.concatenate((np.ravel(X), np.ravel(Theta)))

"""
   Mean normalization is required to fix the problem for a user who didn't rated any movie 
   and we are suppose to recommend some movie the that user with out any rating of the user
   to any movie, this is we can say use for recommending movie to new users.
"""

Y_mean = np.zeros((movies, 1))
Y_norm = np.zeros((movies, users))

for i in range(movies):
    idx = np.where(R[i,:] == 1)[0]
    Y_mean[i] = Y[i,idx].mean()
    Y_norm[i,idx] = Y[i,idx] - Y_mean[i]

Y_norm.mean()

"""
	Implementations of Optimization Algorithms CG
"""
fun_min = minimize(fun=comput_cost_grad, x0=params, 
					args=(Y_norm, R, features, learning_rate), 
					method='CG', jac=True, options={'maxiter': 100})


X = np.matrix(np.reshape(fun_min.x[:movies * features], (movies, features)))
Theta = np.matrix(np.reshape(fun_min.x[movies * features:], (users, features)))

"""
	Our trained parameters are now in X and Theta. 
	We can use these to create some recommendations for the user we added.
"""

predictions = X * Theta.T 
my_predictions = predictions[:, -1] + Y_mean

sorted_preds = np.sort(my_predictions, axis=0)[::-1]
"""
	That gives us an ordered list of the top ratings, but we lost what index those ratings are for. 
"""
sorted_preds[:10]

"""	
	We actually need to use argsort so we know what movie the predicted rating corresponds to
"""
movie_idx = np.argsort(my_predictions, axis=0)[::-1] 
### testing
# for i in range(5):
# 	print movies_name[i]
#   print movie_idx[i]

print("Top 15 predicted movies:")
for i in range(15):
    j = int(movie_idx[i])
    print('Rating for {0}  movie {1}.'.format(str(float(my_predictions[j])), movies_name[j]))
