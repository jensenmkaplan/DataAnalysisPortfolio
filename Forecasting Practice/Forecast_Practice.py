## Jensen Kaplan
## Using an older kaggle data set as a playground for forecasting
## ORIGINAL SOURCE: https://www.kaggle.com/datasets/pratyushakar/rossmann-store-sales

#imports
##################################################################
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##################################################################

### 1) DATA COLLECTION
# Load the 3 datasets. 
## The store dataset contains information about each store, 
# the train dataset contains historical sales data for each 
# store, and the test dataset contains the data for the 
# prediction
##################################################################
store = pd.read_csv('store.csv')
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
##################################################################

# Exploring the data sets by checking the column names, we need to make sure that the date is in there.
# We will see that we want to merge on the 'Store' column
##################################################################
print('Store.csv data : \n##############################')
print(store.columns)
# print(store.head) #print the first few rows
print()

print('Test.csv data : \n##############################')
print(test.columns)
# print(test.head) #print the first few rows
print()

print('Train.csv data : \n##############################')
print(train.columns)
# print(train.head) #print the first few rows
print()
##################################################################


### 2) DATA PREPARATION
# Let's find the common column to merge on using the intersection function
# then we merge them on the common column 'Store'
##################################################################
store_columns = store.columns
test_columns = test.columns
train_columns = train.columns

common_columns_store_test = store_columns.intersection(test_columns)
common_columns_all = common_columns_store_test.intersection(train_columns)

print('##############################')
print('The common columns amongst all data sets are : ', common_columns_all)
print()

# Now we need to merge the data sets with the common column being 'Store'
# using inner join to preserve columns
merged = pd.merge(train, store, on='Store', how = 'inner')

# print('Check out the merged data set: \n')
# print('The columns in the merged set: \n##############################')
# print(merged.columns,'\n')
# print(merged.head)
##################################################################

### 2) FEATURE ENGINEERING
# Step 1: Convert 'Date' column to datetime. Needed for timeseries analysis (sales over time)
# Step 2: Feature Engineering. Extract day of week, month, and year from a 'Date' column
# This way we can choose to observe other time based patterns
# Then we'll visualize
##################################################################
merged['Date'] = pd.to_datetime(merged['Date'])
merged['DayOfWeek'] = merged['Date'].dt.dayofweek
merged['Month'] = merged['Date'].dt.month
merged['Year'] = merged['Date'].dt.year

# Display the first few rows to verify the feature engineering worked
print(merged[['Date', 'DayOfWeek', 'Month', 'Year']].head())

# Check the minimum and maximum dates to understand the timescale we're working with
print("Min date in merged:", merged['Date'].min())
print("Max date in merged:", merged['Date'].max())


# Plotting the 'Sales' data over time
plt.figure()
plt.plot(merged['Date'], merged['Sales'])
plt.show()
##################################################################

# This is for viewing the 'Sales' data on a smaller timescale than the entire set
# Here we only look at the year 2014
##################################################################
start_date = '2014-01-01'
end_date = '2014-01-31'

# Filter the DataFrame for the specified date range
filtered_data = merged[(merged['Date'] >= start_date) & (merged['Date'] <= end_date)]

# Plotting the 'Sales' for the filtered date range
plt.figure()
plt.plot(filtered_data['Date'], filtered_data['Sales'], marker='o', linestyle='-')
plt.title('Sales from January 1, 2020, to January 31, 2020')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)  # Rotate date labels for better readability
plt.tight_layout()  # Adjust layout to make room for the rotated date labels
plt.show()
##################################################################


### 3) DATA CLEANING AND MODEL PREPARATION
# Assuming 'Sales' is the target variable and excluding 'Date' from the features
# Get X and y which will be used for model training
# Change categorical data to numerical
# Drop NaN values (for now, may implement different NaN handling in the future)
##################################################################
X = merged.drop(['Sales', 'Date'], axis=1) # Drop Sales and Date since they are the target variables
y = merged['Sales']
X = pd.get_dummies(X) # this is for changing categorical data to numerical (either 0 or 1) and create new columns for these

# Identify rows with NaN values in X
# we want to drop these but also need to drop them from y, that's what this is for.
nan_rows = X[X.isna().any(axis=1)].index

# Convert to numeric, setting errors to NaN
# Then drop those errors.
X = X.apply(pd.to_numeric, errors='coerce')  
X.dropna(inplace=True)  

# Drop corresponding rows from y
y = y.drop(index=nan_rows)
##################################################################


### 4) MODEL TRAINING AND EVALUATION
# We need to split the X,y features into a training and test set
##################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE) to evaluate the model
# given that this is linear regression we can expect a large value.
# more intelligent models will be more accurate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# # Optional: Print the model's coefficients to interpret the model
# print(f"Coefficients: {model.coef_}")
##################################################################

# Model visualization
# We create a scatter plot to view the result of the linear regression
# viewing the actual vs predicted sales
# Then we view a timeseries of sales vs time showing actual and predicted data
##################################################################
# Scatter plot of actual vs predicted sales values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs Predicted Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')

# Line of perfect predictions
max_val = max(max(y_test), max(y_pred))
min_val = min(min(y_test), min(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], color='red')  # Perfect predictions line

plt.show()


# Step 1: Add predicted sales to the merged DataFrame
# Ensure 'Date' is already converted to datetime and is the index or a column
merged['Predicted_Sales'] = np.nan  # Initialize the column with NaNs
merged.loc[X_test.index, 'Predicted_Sales'] = y_pred  # Fill in predictions for the test set

# Step 2: Sort the DataFrame by 'Date' if it's not already
merged.sort_values('Date', inplace=True)

# Step 3: Plotting
plt.figure(figsize=(12, 6))
plt.plot(merged['Date'], merged['Sales'], label='Actual Sales', marker='o', linestyle='-', color='blue')
plt.plot(merged['Date'], merged['Predicted_Sales'], label='Predicted Sales', marker='x', linestyle='--', color='red')

plt.title('Actual vs Predicted Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
##################################################################

# Continue with model fitting and evaluation as before


# # Check the data types of each column in X
# print(X.dtypes)

# model = LinearRegression()







# # Splitting the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize the model
# model = LinearRegression()

# # Fit the model
# model.fit(X_train, y_train)

# # Predict on the testing set
# y_pred = model.predict(X_test)

# # Calculate the Mean Squared Error (MSE) to evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")

# # You might also want to print the model's coefficients to interpret the model
# print(f"Coefficients: {model.coef_}")



# # Display the shapes of the splits to verify
# # print("X_train shape:", X_train.shape)
# # print("X_test shape:", X_test.shape)
# # print("y_train shape:", y_train.shape)
# # print("y_test shape:", y_test.shape)