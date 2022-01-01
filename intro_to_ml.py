# import the pandas library
import pandas as pd

# importing the data that we will be analyzing
melbourne_file_path = '.../input/melbourne-housing-snapshot/melb_data.csv'
m_data = pd.read_csv(melbourne_file_path)
m_data.describe() # provides a description of the m_data

# count shows how many rows have non-missing values,

# getting the columns of the data
m_data.columns
m_data = m_data.dropna(axis=0) # drops missing values

y = m_data.Price # making y the prediction target (columns that we want to predict)
m_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longitude'] # features (columns that are inputted into our model)
X = m_data[m_features] # creating a variable to store the features

X.describe() # provides an overview of the features
X.head() # provides the first 5 entries

# importing scikit-learn to create a model
from sklearn.tree import DecisionTreeRegressor
m_model = DecisionTreeRegressor(random_state=1) # defining a model, random_state ensures same results each run, we don't want our machine learning model to allow randomness
m_model.fit(X, y) # fitting the model

# step to building and using a model:
# 1. Define. What type of model will it be? What parameters need to be specified?
# 2. Fit. Capture patterns from provided data.
# 3. Predict.
# 4. Evaluate. Determine how accurate the model's predictions are.

# making predictions
print(X.head()) # what we're making predictions for
print(m_model.predict(X.head())) # the predictions

error = actual - predicted # prediction error for each house

# calculating the mean absolute error for our predictions
from sklearn.metrics import mean_absolute_error
predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

# using a different set of data (validation data) to check the validity of our model (important to check because in-sample tests are probably 100% accurate)
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
m_model = DecisionTreeRegressor()
m_model.fit(train_X, train_y)
val_predictions = m_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

# overfitting - where a model matches the training data almost perfectly, but does poorly on new data (result of trees having too much depth)
# underfitting - when a model fails to capture important distinctions and patterns in the data

# utility function to compare MAE scores for different depths of the tree
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_nodes, random_state=0)
    model.fit(train_X, train_y)
    predicted = model.predict(val_X)
    mae = mean_absolute_error(val_y, predicted)
    return mae

# random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree, usually better than a single decision tree

# building a random forest model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))

# save predictions in correct format
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
