import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Let's read the CSV file and package it into a DataFrame:
df = pd.read_csv("winequality-red[1].csv")

#Once the data is loaded in, let's take a quick peek at the first 5 values using the head() method:
print(df.head())

#We can also check the shape of our dataset via the shape property:
print("df.shape:         " , df.shape)

#So, what's the relationship between these variables? A great way to explore relationships between variables is through Scatter plots. We'll plot the hours on the X-axis and scores on the Y-axis, and for each pair, a marker will be positioned based on their values:
df.plot.scatter(x='Hours', y='Scores', title='Scatter Plot of hours and scores percentages');
plt.show()

print("df.describe():                    " , df.describe())


print(" df['Scores'] :     " , df['Scores'])
print("  df['Hours']   :    ", df['Hours']   )

"""To separate the target and features, we can attribute the dataframe column values to our y and X variables:"""
#The .reshape() method takes in two arguments: the first is the number of columns you want the dataframe to have, and the second is the number of rows you want the dataframe to have.
y = df['Scores'].values.reshape(-1, 1)
X = df['Hours'].values.reshape(-1, 1)

print("y :  " , y)
print("X :   " , X)

#Scikit-Learn's linear regression model expects a 2D input, and we're really offering a 1D array if we just extract the values:

print(df['Hours'].values) # [2.5 5.1 3.2 8.5 3.5 1.5 9.2 ... ]
print(df['Hours'].values.shape) # (25,)
#It expects a 2D input because the LinearRegression() class (more on it later) expects entries that may contain more than a single value (but can also be a single value). In either case - it has to be a 2D array, where each element (hour) is actually a 1-element array:

print(X.shape) # (25, 1)
print(X)      # [[2.5] [5.1]  [3.2] ... ]

SEED = 42

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)

#Now, if you print your X_train array - you'll find the study hours, and y_train contains the score percentages:

print(X_train) # [[2.7] [3.3] [5.1] [3.8] ... ]
print(y_train) # [[25] [42] [47] [35] ... ]


#Training a Linear Regression Model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()


#Now, we need to fit the line to our data, we will do that by using the .fit() method along with our X_train and y_train data:

regressor.fit(X_train, y_train)
#If no errors are thrown - the regressor found the best fitting line! The line is defined by our features and the intercept/slope. In fact, we can inspect the intercept and slope by printing the regressor.intecept_ and regressor.coef_ attributes, respectively:

print(regressor.intercept_)

#For retrieving the slope (which is also the coefficient of x):

print(regressor.coef_)

def calc(slope, intercept, hours):
    return slope*hours+intercept

score = calc(regressor.coef_, regressor.intercept_, 9.5)
print(score) # [[94.80663482]]

#However - a much handier way to predict new values using our model is to call on the predict() function:

# Passing 9.5 in double brackets to have a 2 dimensional array
score = regressor.predict([[9.5]])
print(score) # 94.80663482

""""Our result is 94.80663482, or approximately 95%. Now we have a score percentage estimate for each and every hour we can think of. But can we trust those estimates? In the answer to that question is the reason why we split the data into train and test in the first place. Now we can predict using our test data and compare the predicted with our actual results - the ground truth results."""

#To make predictions on the test data, we pass the X_test values to the predict() method. We can assign the results to the variable y_pred:

y_pred = regressor.predict(X_test)
#The y_pred variable now contains all the predicted values for the input values in the X_test. We can now compare the actual output values for X_test with the predicted values, by arranging them side by side in a dataframe structure:

df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)

#Though our model seems not to be very precise, the predicted percentages are close to the actual ones. Let's quantify the difference between the actual and predicted values to gain an objective view of how it's actually performing.


from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
#We will also print the metrics results using the f string and the 2 digit precision after the comma with :.2f:

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')