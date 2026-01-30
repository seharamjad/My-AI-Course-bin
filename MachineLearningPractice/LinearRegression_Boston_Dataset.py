import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Let's read the CSV file and package it into a DataFrame:
df = pd.read_csv('Boston.csv')
# Step 3: Drop unnecessary column (index/serial number)
if "" in df.columns:
    df = df.drop(columns=[""])

# Step 4: Convert all columns to numeric (safety check)
df = df.apply(pd.to_numeric, errors='coerce')

# Step 5: Handle missing values
df = df.dropna()  # remove rows with missing values

# Step 6: Define Features (X) and Target (y)
X = df.drop(columns=['medv'])  # independent variables
y = df['medv']                 # target variable

# Step 7: Check correlation (optional, useful for 

#Once the data is loaded in, let's take a quick peek at the first 5 values using the head() method:
print(df.head())

#We can also check the shape of our dataset via the shape property:
print("df.shape:         " , df.shape)

#So, what's the relationship between these variables? A great way to explore relationships between variables is through Scatter plots. We'll plot the hours on the X-axis and scores on the Y-axis, and for each pair, a marker will be positioned based on their values:
df.plot.scatter(x='tax', y='medv', title='Scatter Plot of displacement and acceleration percentages')
plt.show()

print("df.corr():        " , df.corr())

print("df.describe():                    " , df.describe())

#print(" df['displacement'] :     " , df['displacement'])
print("  df['medv']   :    ", df['medv']   )

#y = df['displacement'].values.reshape(-1, 1)
y = df['medv'].values.reshape(-1, 1)

print("y :  " , y)
print("X :   " , X)

print(df['medv'].values) # [2.5 5.1 3.2 8.5 3.5 1.5 9.2 ... ]
print(df['medv'].values.shape) # (25,)

print(X.shape) # (25, 1)
print(X)      # [[2.5] [5.1]  [3.2] ... ]

SEED = 42

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)

#Now, if you print your X_train array - you'll find the study hours, and y_train contains the score percentages:

print(X_train) # [[2.7] [3.3] [5.1] [3.8] ... ]
print(y_train) # [[25] [42] [47] [35] ... ]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)

def calc(slope, intercept, hours):
    return slope*hours+intercept

medv = calc(regressor.coef_, regressor.intercept_, 9.5)
print(medv)
#medv = regressor.predict([[9.5]])
#print(medv) # 94.80663482
y_pred = regressor.predict(X_test)

df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze()})
print(df_preds)
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
