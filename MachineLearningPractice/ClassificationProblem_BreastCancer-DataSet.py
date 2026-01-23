# Classification Problem

import pandas as pd
from sklearn.datasets import load_breast_cancer

Breast_Cancer_Data = load_breast_cancer()

Breast_Cancer_Data_return_X_Y = load_breast_cancer( return_X_y=True)
"""(data, target)tuple if return_X_y is True
A tuple of two ndarrays by default. The first contains a 2D array of shape (569, 30) with each row representing one sample and each column representing the features. The second array of shape (569,30) contains the target samples."""
print("Breast_Cancer_Data_return_X_Y", Breast_Cancer_Data_return_X_Y)
print("Breast_Cancer_Data_return_X_Y[0]", Breast_Cancer_Data_return_X_Y[0])
print("Breast_Cancer_Data_return_X_Y[1]", Breast_Cancer_Data_return_X_Y[1])

Breast_Cancer_Data_as_frame = load_breast_cancer(as_frame=True)
print('Breast_Cancer_Data_as_frame',Breast_Cancer_Data_as_frame)

# Convert data to pandas Data Frame 
Breast_Cancer_df = pd.DataFrame(Breast_Cancer_Data.data, columns=Breast_Cancer_Data.feature_names)
print("Breast_Cancer_df - DataFrame", Breast_Cancer_df)

# Add the target label 
Breast_Cancer_df['target'] = Breast_Cancer_Data.target

# Take a preview
print("Breast_Cancer_df.head() : ",Breast_Cancer_df.head())


print("Breast_Cancer_df.info() ",Breast_Cancer_df.info() )

print("Breast_Cancer_df.describe()  ",Breast_Cancer_df.describe()  )


print("BreasBreast_Cancer_df.tail()" ,Breast_Cancer_df.tail() )


from sklearn.preprocessing import StandardScaler

# Split data into features and label 
X = Breast_Cancer_df[Breast_Cancer_Data.feature_names].copy()
y = Breast_Cancer_df["target"].copy() 

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

# Split data into train and test
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