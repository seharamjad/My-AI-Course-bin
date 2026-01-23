# Classification Problem (Load Digit Dataset)

import pandas as pd 
from sklearn.datasets import load_digits

digit_data = load_digits()

digit_data_return_X_y = load_digits(return_X_y=True)
"""(data, target)tuple if return_X_y is True
A tuple of two ndarrays by default. The first contains a 2D array of shape (1797, 64) with each row representing one sample and each column representing the features. The second array of shape (1797,64) contains the target samples."""
print("digit_data_return_X_Y :    ", digit_data_return_X_y)
print("digit_data_return_X_Y[0] :    ", digit_data_return_X_y[0])
print("digit_data_return_X_Y[1] :    ", digit_data_return_X_y[1])

digit_data_as_frame = load_digits(as_frame=True)
print("digit_data_as_frame", digit_data_as_frame )

# Convert data to pandas dataframe
digit_data_df = pd.DataFrame(digit_data.data, columns=digit_data.feature_names)

print("digit_data_df - dataFrame: ", digit_data_df)

# Add the target label
digit_data_df["target"] = digit_data.target


# Take a preview
print("wine_df.head() : ", digit_data_df.head())


print(" wine_df.info() ", digit_data_df.info() )

print(" wine_df.describe()  ", digit_data_df.describe()  )


print("wine_df.tail()" , digit_data_df.tail() )


from sklearn.preprocessing import StandardScaler

# Split data into features and label 
X = digit_data_df[digit_data.feature_names].copy()
y = digit_data_df["target"].copy() 

print("X:" , X)
print("y:" , y)

# Instantiate scaler and fit on features
scaler = StandardScaler()
scaler.fit(X)

# Transform features
X_scaled = scaler.transform(X.values)

# View first instance
print(X_scaled[0])


from sklearn.model_selection import train_test_split

X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled,
                                                                  y,
                                                             train_size=.7,
                                                           random_state=25)

# Check the splits are correct
print(f"Train size: {round(len(X_train_scaled) / len(X) * 100)}% \n\
Test size: {round(len(X_test_scaled) / len(X) * 100)}%")

"""
Train size: 70% 
Test size: 30%"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Instnatiating the models 
logistic_regression = LogisticRegression()
svm = SVC()
tree = DecisionTreeClassifier()

# Training the models 
logistic_regression.fit(X_train_scaled, y_train)     # fit command -> for training 
svm.fit(X_train_scaled, y_train)
tree.fit(X_train_scaled, y_train)

# Making predictions with each model
log_reg_preds = logistic_regression.predict(X_test_scaled)
svm_preds = svm.predict(X_test_scaled)
tree_preds = tree.predict(X_test_scaled)

from sklearn.metrics import classification_report

# Store model predictions in a dictionary
# this makes it's easier to iterate through each model
# and print the results. 
model_preds = {
    "Logistic Regression": log_reg_preds,
    "Support Vector Machine": svm_preds,
    "Decision Tree": tree_preds
}

for model, preds in model_preds.items():
    print(f"{model} Results:\n{classification_report(y_test, preds)}", sep="\n\n")