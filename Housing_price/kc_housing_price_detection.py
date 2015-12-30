import graphlab
import numpy as np
from math import log
from math import sqrt
import matplotlib.pyplot as plt

"""
    load dataset of kings coutry
"""
sales = graphlab.SFrame('kc_house_data.gl')

##uncomment to explore dataset info
#sales.head(2) ## view first 2 rows of the data set 
"""
  Split data into training and testing
"""
train_data,test_data = sales.random_split(.8,seed=0) # set a random seed =0

"""
take some feature as a subset and train a regression model on training data
"""
example_features = ['sqft_living', 'bedrooms', 'bathrooms']

## Here is the basic model with some features like sqrt_living, bedrooms, bathrooms
example_model = graphlab.linear_regression.create(train_data, target = 'price', features = example_features, validation_set = None)
example_weight_summary = example_model.get("coefficients")
print("print summary of this basic model")
print example_weight_summary

print("make prediction")
example_predictions = example_model.predict(train_data)
print example_predictions[0] 

### Compute RSS
def get_residual_sum_of_squares(model, data, outcome):
    # First get the predictions
    predictions = model.predict(data)  # this the y heat and real output is ___outcome___
    # rmse =  graphlab.evaluation.rmse(outcome, predictions) ## root mean square error 
    diff = np.subtract(outcome,predictions)
    # square the residuals and add them up
    RSS = np.vdot(diff,diff)
    return(RSS)

### compute this on the test data
rss_example_train = get_residual_sum_of_squares(example_model, test_data, test_data['price'])
print rss_example_train

"""
Create some new features
"""
# bedrooms_squared = bedrooms*bedrooms
# bed_bath_rooms = bedrooms*bathrooms
# log_sqft_living = log(sqft_living)
# lat_plus_long = lat + long

train_data['bedrooms_squared'] = train_data['bedrooms'].apply(lambda x: x**2)
test_data['bedrooms_squared'] = test_data['bedrooms'].apply(lambda x: x**2)
train_data['bed_bath_rooms'] = train_data['bedrooms'] * train_data['bathrooms']
test_data['bed_bath_rooms'] = test_data['bedrooms'] * test_data['bathrooms']
train_data['log_sqft_living'] = train_data['sqft_living'].apply(lambda x: log(x))
test_data['log_sqft_living'] = test_data['sqft_living'].apply(lambda x: log(x))
train_data['lat_plus_long'] = train_data['lat'] + train_data['long']
test_data['lat_plus_long'] = test_data['lat'] + test_data['long']

print sum(test_data['bedrooms_squared'])/len(test_data['bedrooms_squared'])
print sum(test_data['bed_bath_rooms'])/len(test_data['bed_bath_rooms'])
print sum(test_data['log_sqft_living'])/len(test_data['log_sqft_living'])
print sum(test_data['lat_plus_long'])/len(test_data['lat_plus_long'])

# Model 1: squarefeet, # bedrooms, # bathrooms, latitude & longitude
# Model 2: add bedrooms*bathrooms
# Model 3: Add log squarefeet, bedrooms squared, and the (nonsensical) latitude + longitude

model_1_features = ['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']
model_2_features = model_1_features + ['bed_bath_rooms']
model_3_features = model_2_features + ['bedrooms_squared', 'log_sqft_living', 'lat_plus_long']

# Learn the three models: (don't forget to set validation_set = None)

model1 = graphlab.linear_regression.create(train_data, target = 'price', features = model_1_features, validation_set = None)

model2 = graphlab.linear_regression.create(train_data, target = 'price', features = model_2_features, validation_set = None)

model3 = graphlab.linear_regression.create(train_data, target = 'price', features = model_3_features, validation_set = None)

model1_coefficients = model1.get("coefficients")
print model1_coefficients

model2_coefficients = model2.get("coefficients")
print model2_coefficients

model3_coefficients = model3.get("coefficients")
print model3_coefficients
"""
Comparing multiple models"""

# Compute the RSS on TESTING data for each of the three models and record the values:
rss_model1_test = get_residual_sum_of_squares(model1, test_data, test_data['price'])
print rss_model1_test

rss_model2_test = get_residual_sum_of_squares(model2, test_data, test_data['price'])
print rss_model2_test

rss_model3_test = get_residual_sum_of_squares(model3, test_data, test_data['price'])
print rss_model3_test
"""
Now compute the RSS on on T data for each of the three models."""

# Compute the RSS on TRAINING data for each of the three models and record the values:
rss_model1_train = get_residual_sum_of_squares(model1, train_data, train_data['price'])
print rss_model1_train

rss_model2_train = get_residual_sum_of_squares(model2, train_data, train_data['price'])
print rss_model2_train

rss_model3_train = get_residual_sum_of_squares(model3, train_data, train_data['price'])
print rss_model3_train

"""
A numpy matrix whose columns are the desired features plus a constant column (this is how we create an 'intercept')
A numpy array containing the values of the output
"""
def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1 # this is how you add a constant column to an SFrame
    # add the column 'constant' to the front of the features list so that we can extract it along with the others:
    features = ['constant'] + features # this is how you combine two lists
    # select the columns of data_SFrame given by the features list into the SFrame features_sframe (now including constant):
    features_sframe = data_sframe[features]
    # the following line will convert the features_SFrame into a numpy matrix:
    feature_matrix = features_sframe.to_numpy()
    # assign the column of data_sframe associated with the output to the SArray output_sarray
    output_sarray = data_sframe['price']
    # the following will convert the SArray into a numpy array by first converting it to a list
    output_array = output_sarray.to_numpy()
    return(feature_matrix, output_array)

(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') # the [] around 'sqft_living' makes it a list
print example_features[0,:] # this accesses the first row of the data the ':' indicates 'all columns'
print example_output[0] # and the corresponding output

my_weights = np.array([1., 1.]) # the example weights
my_features = example_features[0,] # we'll use the first data point
predicted_value = np.dot(my_features, my_weights)
print predicted_value

def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix, weights)
    return(predictions)

test_predictions = predict_output(example_features, my_weights)
print test_predictions[0] 
print test_predictions[1]

"""We are now going to move to computing the derivative of the regression cost function. 
Recall that the cost function is the sum over the data points of the squared difference 
between an observed output and a predicted output."""

"""(w[0]*[CONSTANT] + w[1]*[feature_1] + ... + w[i] *[feature_i] + ... + w[k]*[feature_k] - output)^2

Where we have k features and a constant. So the derivative with respect to weight w[i] by the chain rule is:

2*(w[0]*[CONSTANT] + w[1]*[feature_1] + ... + w[i] *[feature_i] + ... + w[k]*[feature_k] - output)* [feature_i]

The term inside the paranethesis is just the error (difference between prediction and output). So we can re-write this as:

2*error*[feature_i]
"""
def feature_derivative(errors, feature):
    # Assume that errors and feature are both numpy arrays of the same length (number of data points)
    # compute twice the dot product of these vectors as 'derivative' and return the value
    derivative = 2*np.dot(errors, feature)
    return(derivative)



(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') 
my_weights = np.array([0., 0.]) # this makes all the predictions 0
test_predictions = predict_output(example_features, my_weights) 
# just like SFrames 2 numpy arrays can be elementwise subtracted with '-': 
errors = test_predictions - example_output # prediction errors in this case is just the -example_output
feature = example_features[:,0] # let's compute the derivative with respect to 'constant', the ":" indicates "all rows"
derivative = feature_derivative(errors, feature)
print derivative
print -np.sum(example_output)*2 # should be the same as derivative



def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False 
    weights = np.array(initial_weights) # make sure it's a numpy array
    gradient_magnitude = 0
    while not converged:
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = predict_output(feature_matrix, weights)
        # compute the errors as predictions - output
        errors = predictions - output
        gradient_sum_squares = 0 # initialize the gradient sum of squares
        # while we haven't reached the tolerance yet, update each feature's weight
        for i in range(len(weights)): # loop over each weight
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            drivative = feature_derivative(errors, feature_matrix[:, i])
            # add the squared value of the derivative to the gradient magnitude (for assessing convergence)
            gradient_sum_squares += drivative * drivative
            # subtract the step size times the derivative from the current weight
            weights[i] -= step_size * drivative
        # compute the square-root of the gradient sum of squares to get the gradient matnigude:
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)

"""
	Running the Gradient Descent as Simple Regression
"""

train_data,test_data = sales.random_split(.8,seed=0)
# let's test out the gradient descent
simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

simple_weights = regression_gradient_descent(simple_feature_matrix, output,initial_weights, step_size,tolerance)
print simple_weights

(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)
test_predictions = predict_output(test_simple_feature_matrix, simple_weights)
print test_predictions[0]

"""
	Running multiple Gradient Descent
"""

model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9

simple_weights = regression_gradient_descent(feature_matrix, output,initial_weights, step_size, tolerance)
print simple_weights

(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)
test_predictions = predict_output(test_simple_feature_matrix, simple_weights)
print test_predictions[0]

"""
Use some polynomial feature here is the function use to create polynomial features 
"""
def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    poly_sframe = graphlab.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1'] = feature

    # first check if degree > 1fea
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree+1): 
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # then assign poly_sframe[name] to the appropriate power of feature
            #name = name.apply(lambda name: names**power)
            poly_sframe[name] = feature #poly_sframe.apply(lambda name : name**power)
            tmp = poly_sframe[name]
            tmp_power = tmp.apply(lambda x: x**power)
            poly_sframe[name] = tmp_power
            #print tmp_power

    return poly_sframe


sales = sales.sort(['sqft_living', 'price'])
poly1_data = polynomial_sframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price'] # add price to the data since it's the target

model_poly1 = graphlab.linear_regression.create(poly1_data, target = 'price', features = ['power_1'], validation_set = None)
#let's take a look at the weights before we plot
model_poly1.get("coefficients")

## uncomment this to view the relation diagram
#plt.plot(poly1_data['power_1'],poly1_data['price'],'.',poly1_data['power_1'], model_poly1.predict(poly1_data),'-')

poly2_data = polynomial_sframe(sales['sqft_living'], 2)
my_features = poly2_data.column_names() # get the name of the features
poly2_data['price'] = sales['price'] # add price to the data since it's the target
model_poly2 = graphlab.linear_regression.create(poly2_data, target = 'price', features = my_features, validation_set = None)
model_poly2.get("coefficients")

## uncomment this to view the relation diagram
#plt.plot(poly2_data['power_1'],poly2_data['price'],'.',poly2_data['power_1'], model_poly2.predict(poly2_data),'-')

poly3_data = polynomial_sframe(sales['sqft_living'], 3)
my_features3 = poly3_data.column_names() # get the name of the features
poly3_data['price'] = sales['price'] # add price to the data since it's the target
model_poly3 = graphlab.linear_regression.create(poly3_data, target = 'price', features = my_features3, validation_set = None)
model_poly3.get("coefficients")

## uncomment this to view the relation diagram
#plt.plot(poly3_data['power_1'],poly3_data['price'],'.',poly3_data['power_1'], model_poly3.predict(poly3_data),'-')

poly15_data = polynomial_sframe(sales['sqft_living'], 15)
my_features15 = poly15_data.column_names() # get the name of the features
poly15_data['price'] = sales['price'] # add price to the data since it's the target
model_poly15 = graphlab.linear_regression.create(poly15_data, target = 'price', features = my_features15, validation_set = None)
model_poly15.get("coefficients")

## uncomment this to view the relation diagram
#plt.plot(poly15_data['power_1'],poly15_data['price'],'.',poly15_data['power_1'], model_poly15.predict(poly15_data),'-')

"""
 Changing the data and re-learning
 First split sales into 2 subsets with .random_split(0.5, seed=0).
 Next split the resulting subsets into 2 more subsets each. Use .random_split(0.5, seed=0).
"""

train_data,test_data = sales.random_split(.5,seed=0)
set_1,set_2 = train_data.random_split(0.5,seed=0)
set_3,set_4 = test_data.random_split(0.5,seed=0)

"""
set 1"""
poly15_set1 = polynomial_sframe(set_1['sqft_living'], 15)
my_features15 = poly15_set1.column_names() # get the name of the features
poly15_set1['price'] = set_1['price'] # add price to the data since it's the target
modelset1 = graphlab.linear_regression.create(poly15_set1, target = 'price', features = my_features15, validation_set = None)
modelset1.get("coefficients")

"""
set 2"""
poly15_set2 = polynomial_sframe(set_2['sqft_living'], 15)
my_features15 = poly15_set2.column_names() # get the name of the features
poly15_set2['price'] = set_2['price'] # add price to the data since it's the target
modelset2 = graphlab.linear_regression.create(poly15_set2, target = 'price', features = my_features15, validation_set = None)
modelset2.get("coefficients")

"""
set 3"""
poly15_set3 = polynomial_sframe(set_3['sqft_living'], 15)
my_features15 = poly15_set3.column_names() # get the name of the features
poly15_set3['price'] = set_3['price'] # add price to the data since it's the target
modelset3 = graphlab.linear_regression.create(poly15_set3, target = 'price', features = my_features15, validation_set = None)
modelset3.get("coefficients")

"""
set 4"""
poly15_set4 = polynomial_sframe(set_4['sqft_living'], 15)
my_features15 = poly15_set4.column_names() # get the name of the features
poly15_set4['price'] = set_4['price'] # add price to the data since it's the target
modelset4 = graphlab.linear_regression.create(poly15_set4, target = 'price', features = my_features15, validation_set = None)
modelset4.get("coefficients")

"""
    Split our sales data into 2 sets: training_and_validation and testing. Use random_split(0.9, seed=1).
    Further split our training data into two sets: training and validation. Use random_split(0.5, seed=1).
"""

training_and_validation,testing = sales.random_split(.9, seed=1)
training,validation = training_and_validation.random_split(.5, seed=1)

for degree in range(1, 15+1):
    poly_data = polynomial_sframe(training['sqft_living'], degree)
    vali_data = polynomial_sframe(validation['sqft_living'], degree)
    test_data = polynomial_sframe(testing['sqft_living'], degree)
    poly_features = poly_data.column_names()
    poly_data['price'] = training['price']
    poly_model = graphlab.linear_regression.create(poly_data, target = 'price', features = poly_features,
                                                                                 validation_set = None, verbose = False)
    
    predictions = poly_model.predict(vali_data)
    predictions_test = poly_model.predict(test_data)
    validation_errors = predictions - validation['price']
    test_errors = predictions_test - testing['price']
    RSS = sum(validation_errors * validation_errors)
    RSS_test = sum(test_errors * test_errors)
    print "degree : " + str(degree) + ", RSS : " + str(RSS) + ", RSS_test : " + str(RSS_test) + ", Training loss : " \
           + str(poly_model.get('training_loss'))

""""
Apply some regression on the features here we are applying ridge regression"""


sales = sales.sort(['sqft_living','price'])
l2_small_penalty = 1e-5  ## take some penalty

poly15_data = polynomial_sframe(sales['sqft_living'], 15)
my_features15 = poly15_data.column_names() # get the name of the features
poly15_data['price'] = sales['price'] # add price to the data since it's the target
model15 = graphlab.linear_regression.create(poly15_data, target = 'price', features = my_features15,
                                                                           l2_penalty=l2_small_penalty,
                                                                            validation_set = None)


model15.get('coefficients')

"""
    Observe overfitting"""

(semi_split1, semi_split2) = sales.random_split(.5,seed=0)
(set_1, set_2) = semi_split1.random_split(0.5, seed=0)
(set_3, set_4) = semi_split2.random_split(0.5, seed=0)

poly15_set1 = polynomial_sframe(set_1['sqft_living'], 15)
my_features15 = poly15_set1.column_names() # get the name of the features
poly15_set1['price'] = set_1['price'] # add price to the data since it's the target
modelset1 = graphlab.linear_regression.create(poly15_set1, target = 'price', features = my_features15,
                                                                              l2_penalty=l2_small_penalty,  
                                                                              validation_set = None)


modelset1.get('coefficients')
## uncomment
# plt.plot(poly15_set1['power_1'],poly15_set1['price'],'.', poly15_set1['power_1'], modelset1.predict(poly15_set1),'-')

poly15_set2 = polynomial_sframe(set_2['sqft_living'], 15)
my_features15 = poly15_set2.column_names() # get the name of the features
poly15_set2['price'] = set_2['price'] # add price to the data since it's the target
modelset2 = graphlab.linear_regression.create(poly15_set2, target = 'price', features = my_features15,
                                                                              l2_penalty=l2_small_penalty,  
                                                                              validation_set = None)


modelset2.get('coefficients')
##uncomment
# plt.plot(poly15_set2['power_1'],poly15_set2['price'],'.',poly15_set2['power_1'], modelset2.predict(poly15_set2),'-')

poly15_set3 = polynomial_sframe(set_3['sqft_living'], 15)
my_features15 = poly15_set3.column_names() # get the name of the features
poly15_set3['price'] = set_3['price'] # add price to the data since it's the target
modelset3 = graphlab.linear_regression.create(poly15_set3, target = 'price', features = my_features15,
                                                                              l2_penalty=l2_small_penalty,  
                                                                              validation_set = None)


modelset3.get('coefficients')
### uncomment
# plt.plot(poly15_set3['power_1'],poly15_set3['price'],'.',poly15_set3['power_1'], modelset3.predict(poly15_set3),'-')

poly15_set4 = polynomial_sframe(set_4['sqft_living'], 15)
my_features15 = poly15_set4.column_names() # get the name of the features
poly15_set4['price'] = set_4['price'] # add price to the data since it's the target
modelset4 = graphlab.linear_regression.create(poly15_set4, target = 'price', features = my_features15,
                                                                              l2_penalty=l2_small_penalty,  
                                                                              validation_set = None)


modelset4.get('coefficients')
### uncomment
# plt.plot(poly15_set4['power_1'],poly15_set4['price'],'.',poly15_set4['power_1'], modelset4.predict(poly15_set4),'-')

"""Ridge regression comes to rescue"""

poly15_set1 = polynomial_sframe(set_1['sqft_living'], 15)
my_features15 = poly15_set1.column_names() # get the name of the features
poly15_set1['price'] = set_1['price'] # add price to the data since it's the target
modelset1 = graphlab.linear_regression.create(poly15_set1, target = 'price', features = my_features15,
                                                                              l2_penalty=1e5,  
                                                                              validation_set = None)


modelset1.get('coefficients')
# plt.plot(poly15_set1['power_1'],poly15_set1['price'],'.',poly15_set1['power_1'], modelset1.predict(poly15_set1),'-')

poly15_set2 = polynomial_sframe(set_2['sqft_living'], 15)
my_features15 = poly15_set2.column_names() # get the name of the features
poly15_set2['price'] = set_2['price'] # add price to the data since it's the target
modelset2 = graphlab.linear_regression.create(poly15_set2, target = 'price', features = my_features15,
                                                                              l2_penalty=1e5,  
                                                                              validation_set = None)


modelset2.get('coefficients')
# plt.plot(poly15_set2['power_1'],poly15_set2['price'],'.', poly15_set2['power_1'], modelset2.predict(poly15_set2),'-')

poly15_set3 = polynomial_sframe(set_3['sqft_living'], 15)
my_features15 = poly15_set3.column_names() # get the name of the features
poly15_set3['price'] = set_3['price'] # add price to the data since it's the target
modelset3 = graphlab.linear_regression.create(poly15_set3, target = 'price', features = my_features15,
                                                                              l2_penalty=1e5,  
                                                                              validation_set = None)


modelset3.get('coefficients')
# plt.plot(poly15_set3['power_1'],poly15_set3['price'],'.', poly15_set3['power_1'], modelset3.predict(poly15_set3),'-')

poly15_set4 = polynomial_sframe(set_4['sqft_living'], 15)
my_features15 = poly15_set4.column_names() # get the name of the features
poly15_set4['price'] = set_4['price'] # add price to the data since it's the target
modelset4 = graphlab.linear_regression.create(poly15_set4, target = 'price', features = my_features15,
                                                                              l2_penalty=1e5,  
                                                                              validation_set = None)


modelset4.get('coefficients')
# plt.plot(poly15_set4['power_1'],poly15_set4['price'],'.',poly15_set4['power_1'], modelset4.predict(poly15_set4),'-')

"""Selecting an L2 penalty via cross-validation"""

(train_valid, test) = sales.random_split(.9, seed=1)
train_valid_shuffled = graphlab.toolkits.cross_validation.shuffle(train_valid, random_seed=1)

n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation

for i in xrange(k):
    start = (n*i)/k
    end = (n*(i+1))/k-1
    print i, (start, end)

n = len(train_valid_shuffled)
k = 10 # 10-fold cross-validation

validation4 =graphlab.SFrame()

for i in xrange(k):
    if i<4:
        start = (n*i)/k
        end = (n*(i+1))/k-1
        validation4 = validation4.append(train_valid_shuffled[start:end+1])
        print i, (start, end)
#validation4 = validation4[:-1]

validation4 = validation4[5818:7758]
print int(round(train_valid_shuffled[5818:7757+1]['price'].mean(), 0))

n = len(train_valid_shuffled)
first_two = train_valid_shuffled[0:2]
last_two = train_valid_shuffled[n-2:n]
#print first_two.append(last_two)

n = len(train_valid_shuffled)
first_segment = train_valid_shuffled[0:5818] ## before the segment 3
last_segment = train_valid_shuffled[7758:n] ## after the segment 3
train4 = first_segment.append(last_segment) ## train4 dataset contain all the data excluding fourth

print int(round(train4['price'].mean(), 0))

"""
    Now we are ready to implement k-fold cross-validation. Write a function that computes k validation errors by designating 
    each of the k segments as the validation set. It accepts as parameters (i) k, (ii) l2_penalty, (iii) dataframe, 
    (iv) name of output column (e.g. price) and (v) list of feature names. The function returns the average validation error 
    using k segments as validation sets.
    """

def k_fold_cross_validation(k, l2_penalty, data, output_name, features_list):
    n = len(data)
    validation_errors = []
    for i in xrange(k):
        start = (n*i)/k
        end = (n*(i+1))/k-1
        validation_set = data[start:end+1]
        training_set = data[end+1:n].append(data[0:start])
        
        ## train a linear model using training set just formend, with given l2_penalty
        model = graphlab.linear_regression.create(training_set, 
                                                  target = output_name, 
                                                  features = features_list,
                                                  l2_penalty=l2_penalty,
                                                  validation_set = None,
                                                  verbose = False) 

        # predict on validation set 
        pred = model.predict(validation_set)
       
        validation_error = pred - validation_set['price']
        RSS = sum(validation_error * validation_error)
        
        validation_errors.append(RSS)
    
    return sum(validation_errors)/len(validation_errors)


poly15_train_valid_shuffled = polynomial_sframe(train_valid_shuffled['sqft_living'], 15)
my_features15 = poly15_train_valid_shuffled.column_names() # get the name of the features
poly15_train_valid_shuffled['price'] = train_valid_shuffled['price'] # add price to the data since it's the target

l2_penalty_list = np.logspace(1, 7, num=13)

min_error = None
best_l2_penalty = None
cross_validation_errors = []

for l2_penalty in l2_penalty_values:
    avg_val_error = k_fold_cross_validation(10, l2_penalty, poly15_train_valid_shuffled, 'price', set15_features)
    print "For l2_penalty: " + str(l2_penalty) + " ---> Avg validation penalty : " + str(avg_val_error)
    cross_validation_errors.append(avg_val_error)
    if min_error is None or avg_val_error < min_error:
        min_error = avg_val_error
        best_l2_penalty = l2_penalty
print "Best l2_penalty --->: " + str(best_l2_penalty)


poly15_sales = polynomial_sframe(sales['sqft_living'], 15)
my_features15 = poly15_sales.column_names() # get the name of the features
poly15_sales['price'] = sales['price'] # add price to the data since it's the target
modelset4 = graphlab.linear_regression.create(poly15_sales, 
                                              target = 'price', 
                                              features = my_features15,
                                              l2_penalty=best_l2_penalty,  
                                              validation_set = None)


modelset4.get('coefficients')
# plt.plot(poly15_sales['power_1'],poly15_sales['price'],'.',
        # poly15_sales['power_1'], modelset4.predict(poly15_sales),'-')

"""RSS on the TEST data of the model you learn with this L2 penalty? """
test_data = polynomial_sframe(test['sqft_living'], 15)
predictions_test = modelset4.predict(test_data)
test_errors = predictions_test - test['price']
RSS_test = sum(test_errors * test_errors)
print RSS_test

"""Computing the Derivative
We are now going to move to computing the derivative of the regression cost function. 
Recall that the cost function is the sum over the data points of the squared difference between an observed output 
and a predicted output, plus the L2 penalty term.

Cost(w)= SUM[ (prediction - output)^2 ] + l2_penalty*(w[0]^2 + w[1]^2 + ... + w[k]^2).

Since the derivative of a sum is the sum of the derivatives, we can take the derivative of the first part (the RSS) 
as we did in the notebook for the unregularized case in Week 2 and add the derivative of the regularization part. 
As we saw, the derivative of the RSS with respect to w[i] can be written as:

2*SUM[ error*[feature_i] ].

The derivative of the regularization term with respect to w[i] is:

2*l2_penalty*w[i].

Summing both, we get

2*SUM[ error*[feature_i] ] + 2*l2_penalty*w[i].

That is, the derivative for the weight for feature i is the sum (over data points) of 2 times the product of the error and the 
feature itself, plus 2*l2_penalty*w[i].
"""
def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    # If feature_is_constant is True, derivative is twice the dot product of errors and feature
    derivative = 2 * np.dot(errors, feature)
    if not feature_is_constant:
        # Otherwise, derivative is twice the dot product plus 2*l2_penalty*weight
        derivative = derivative + 2 * l2_penalty * weight 
    
    return derivative
(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price') 
my_weights = np.array([1., 10.])
test_predictions = predict_output(example_features, my_weights) 
errors = test_predictions - example_output # prediction errors

# next two lines should print the same values
print feature_derivative_ridge(errors, example_features[:,1], my_weights[1], 1, False)
print np.sum(errors*example_features[:,1])*2+20.
print ''

# next two lines should print the same values
print feature_derivative_ridge(errors, example_features[:,0], my_weights[0], 1, True)
print np.sum(errors)*2

"""Gradient Descent
Now we will write a function that performs a gradient descent. The basic premise is simple. 
Given a starting point we update the current weights by moving in the negative gradient direction. 
Recall that the gradient is the direction of increase and therefore the negative gradient is the direction of decrease and 
we're trying to minimize a cost function."""

def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
    weights = np.array(initial_weights) # make sure it's a numpy array
    iter = 1
    while iter <= max_iterations:
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = predict_output(feature_matrix, weights)
        # compute the errors as predictions - output
        errors = predictions - output
        for i in xrange(len(weights)): # loop over each weight
            # Recall that feature_matrix[:,i] is the feature column associated with weights[i]
            # compute the derivative for weight[i].
            #(Remember: when i=0, you are computing the derivative of the constant!)
            isConstant = False
            if i == 0:
                isConstant = True
            derivative = feature_derivative_ridge(errors, feature_matrix[:,i], weights[i], l2_penalty, isConstant)
            # subtract the step size times the derivative from the current weight
            weights[i] = weights[i] - step_size * derivative 
        iter += 1
    return weights

"""Visualizing effect of L2 penalty"""

simple_features = ['sqft_living']
my_output = 'price'

train_data,test_data = sales.random_split(.8,seed=0)

(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)

initial_weights = np.array([0., 0.])
step_size = 1e-12
max_iterations=1000

l2_penalty = 0.0
simple_weights_0_penalty = ridge_regression_gradient_descent(simple_feature_matrix, 
                                                             output, 
                                                             initial_weights, 
                                                             step_size, 
                                                             l2_penalty, 
                                                             max_iterations)
print simple_weights_0_penalty



l2_penalty = 1e11
simple_weights_high_penalty = ridge_regression_gradient_descent(simple_feature_matrix, 
                                                                output, 
                                                                initial_weights, 
                                                                step_size, 
                                                                l2_penalty, 
                                                                max_iterations)
print simple_weights_high_penalty

"""

    The initial weights (all zeros)
    The weights learned with no regularization
    The weights learned with high regularization
"""

test_predictions = predict_output(simple_test_feature_matrix, initial_weights)
test_errors = test_predictions - test_output
RSS_initial_weights = sum(test_errors * test_errors)
print RSS_initial_weights

testNoReg_predictions = predict_output(simple_test_feature_matrix, simple_weights_0_penalty)
testNoReg_errors = testNoReg_predictions - test_output
NoReg_RSS = sum(testNoReg_errors * testNoReg_errors)
print NoReg_RSS

testReg_predictions = predict_output(simple_test_feature_matrix, simple_weights_high_penalty)
testReg_errors = testReg_predictions - test_output
Reg_RSS = sum(testReg_errors * testReg_errors)
print Reg_RSS

"""Running a multiple regression with L2 penalty"""

model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)

initial_weights = np.array([0.0,0.0,0.0])
step_size = 1e-12
max_iterations = 1000

l2_penalty = 0.0
multiple_weights_0_penalty = ridge_regression_gradient_descent(feature_matrix, 
                                                               output, 
                                                               initial_weights, 
                                                               step_size, 
                                                               l2_penalty, 
                                                               max_iterations)
print multiple_weights_0_penalty

l2_penalty = 1e11
multiple_weights_high_penalty = ridge_regression_gradient_descent(feature_matrix, 
                                                                  output, 
                                                                  initial_weights, 
                                                                  step_size, 
                                                                  l2_penalty, 
                                                                  max_iterations)

print multiple_weights_high_penalty

"""
    The initial weights (all zeros)
    The weights learned with no regularization
    The weights learned with high regularization
"""


multiple_test_pred = predict_output(test_feature_matrix, initial_weights)
multiple_test_errors = multiple_test_pred - test_output
RSS_multi_initial_weights = sum(multiple_test_errors * multiple_test_errors)
print RSS_multi_initial_weights

multiple_testNoReg_pred = predict_output(test_feature_matrix, multiple_weights_0_penalty)
multiple_testNoReg_errors = multiple_testNoReg_pred - test_output
RSS_multi_NoReg = sum(multiple_testNoReg_errors * multiple_testNoReg_errors)
print RSS_multi_NoReg