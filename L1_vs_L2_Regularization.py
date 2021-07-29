!wget https://raw.githubusercontent.com/Shreyas3108/house-price-prediction/master/kc_house_data.csv

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score 
from statistics import mean 
from sklearn import preprocessing

# Changing the working location to the location of the data 

# Loading the data into a Pandas DataFrame 
data = pd.read_csv('kc_house_data.csv')

# Dropping the numerically non-sensical variables 
dropColumns = ['id', 'date', 'zipcode'] 
data = data.drop(dropColumns, axis = 1) 
data = data[0:1000]

# Separating the dependent and independent variables 
y = data['price'] 
X = data.drop('price', axis = 1) 

#x = X.values #returns a numpy array
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x)
#X = pd.DataFrame(x_scaled)

# Dividing the data into training and testing set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

X.describe()


# Bulding and fitting the Linear Regression model 
linearModel = LinearRegression() 
linearModel.fit(X_train, y_train) 

# Evaluating the Linear Regression model 
print(linearModel.score(X_train, y_train)) 
print(linearModel.score(X_test, y_test))


# List to maintain the different cross-validation scores 
cross_val_scores_ridge = [] 

# List to maintain the different values of alpha 
alpha = [] 
# Loop to compute the different values of cross-validation scores 
for i in range(-3, 3): 
	ridgeModel = Ridge(alpha = 10**(-i)) 
	ridgeModel.fit(X_train, y_train) 
	scores = cross_val_score(ridgeModel, X, y, cv = 10) 
	avg_cross_val_score = mean(scores)*100
	cross_val_scores_ridge.append(avg_cross_val_score) 
	alpha.append(10**(-i)) 

# Loop to print the different values of cross-validation scores 
for i in range(0, len(alpha)): 
	print(str(alpha[i])+' : '+str(cross_val_scores_ridge[i]))
  
# List to maintain the different cross-validation scores 
cross_val_scores_ridge = [] 

# List to maintain the different values of alpha 
alpha = [] 

# Loop to compute the different values of cross-validation scores 
for i in range(1, 9): 
	ridgeModel = Ridge(alpha = i * 0.25) 
	ridgeModel.fit(X_train, y_train) 
	scores = cross_val_score(ridgeModel, X, y, cv = 10) 
	avg_cross_val_score = mean(scores)*100
	cross_val_scores_ridge.append(avg_cross_val_score) 
	alpha.append(i * 0.25) 

# Loop to print the different values of cross-validation scores 
for i in range(0, len(alpha)): 
	print(str(alpha[i])+' : '+str(cross_val_scores_ridge[i]))
  
# Building and fitting the Ridge Regression model 
ridgeModelChosen = Ridge(alpha = 1.5) 
ridgeModelChosen.fit(X_train, y_train) 

# Evaluating the Ridge Regression model 

print(ridgeModelChosen.score(X_train, y_train)) 
print(ridgeModelChosen.score(X_test, y_test))

# List to maintain the cross-validation scores 
cross_val_scores_lasso = [] 

# List to maintain the different values of Lambda 
Lambda = [] 

# Loop to compute the cross-validation scores 
for i in range(-3, 3): 
	lassoModel = Lasso(alpha = 10**(-i), tol = 0.0925) 
	lassoModel.fit(X_train, y_train) 
	scores = cross_val_score(lassoModel, X, y, cv = 10) 
	avg_cross_val_score = mean(scores)*100
	cross_val_scores_lasso.append(avg_cross_val_score) 
	Lambda.append(10**(-i)) 

# Loop to print the different values of cross-validation scores 
for i in range(-3,3): 
	print(str(10**(-i))+' : '+str(cross_val_scores_lasso[i]))
  
  
# List to maintain the cross-validation scores 
cross_val_scores_lasso = [] 

# List to maintain the different values of Lambda 
Lambda = [] 

# Loop to compute the cross-validation scores 
for i in range(1, 9): 
	lassoModel = Lasso(alpha = i * 0.25, tol = 0.0925) 
	lassoModel.fit(X_train, y_train) 
	scores = cross_val_score(lassoModel, X, y, cv = 10) 
	avg_cross_val_score = mean(scores)*100
	cross_val_scores_lasso.append(avg_cross_val_score) 
	Lambda.append(i * 0.25) 

# Loop to print the different values of cross-validation scores 
for i in range(0, len(alpha)): 
	print(str(i * 0.25)+' : '+str(cross_val_scores_lasso[i]))
  
  
# Building and fitting the Lasso Regression Model 
lassoModelChosen = Lasso(alpha = 10, tol = 0.0925) 
lassoModelChosen.fit(X_train, y_train) 
# Evaluating the Lasso Regression model 
print(lassoModelChosen.score(X_train, y_train)) 
print(lassoModelChosen.score(X_test, y_test))


linearModel.coef_

# Building the two lists for visualization 
models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression'] 
scores = [linearModel.score(X_test, y_test), 
		ridgeModelChosen.score(X_test, y_test), 
		lassoModelChosen.score(X_test, y_test)] 

# Building the dictionary to compare the scores 
mapping = {} 
mapping['Linear Regreesion'] = linearModel.score(X_test, y_test) 
mapping['Ridge Regreesion'] = ridgeModelChosen.score(X_test, y_test) 
mapping['Lasso Regression'] = lassoModelChosen.score(X_test, y_test) 

# Printing the scores for different models 
for key, val in mapping.items(): 
	print(str(key)+' : '+str(val))
  
# Plotting the scores 
plt.bar(models, scores) 
plt.xlabel('Regression Models') 
plt.ylabel('Score') 
plt.show()


