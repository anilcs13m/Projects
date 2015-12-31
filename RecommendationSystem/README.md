Rating Prediction and Recommending Movies
=========================================

Recommendation engines use item and user-based similarity measures to examine a user's historical preferences to
make recommendations for new "things" the user might be interested in.  

Here in this we are implementing a particular recommendation system algorithm called collaborative filtering and apply it to a data set 
of movie rating. In this implementation we are given a datset with user rated some movies, this data sparse dataset as some of the movies 
have not rated by the user or some of the user never rated movies. So how to deal with such type of problem and after fixing all the problem how to recommend movies to the user

##### Here are the steps that we followed to make redictions of rating and recommend movie

=====================
* First we loaded dataset here we have three different dataset first is movies and user rating matrix second user movies matrix 
  that have (0,1) entry represent 1 of user j rated movie i in this matrix and other dataset is the parameter vector that have learning parameters for user and movie

* Implementing the collaborative filtering cost function and gradient

* You can now start training your algorithm to make movie recommendations for yourself

* Predict the rating of movie i for user j

* Computes the ratings for all the movies and users and displays the movies that it recommends







