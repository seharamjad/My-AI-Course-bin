# Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

#  load csv file
df= pd.read_csv("FINAL ASSESSMENT\Final Assessment ML Project\InsurranceData.csv")
print("Print first 5 rows below:", df.head())
print("Print no of rows and columns belows:" , df.shape)

# Describe
print("Print Describe Here:" , df.describe())
#EDA 
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['Charges'], kde=True)
plt.title("Distribution of Charges")
plt.xlabel("Charges")
plt.ylabel("Frequency")
plt.show()

sns.displot(df['Charges'] , kde=True)
plt.show()

sns.boxplot(x=df['Charges'])
plt.title("BoxPlot of Charges")
plt.show()

sns.scatterplot(x='age', y='Charges', data=df)
plt.title("Age vs Charges")
plt.show()

sns.scatterplot(x='bmi', y='Charges', data=df)
plt.title("Age vs Charges")
plt.show()
 

sns.boxplot(x='smoker' , y='Charges' , data=df)
plt.title("smoker vs charges")
plt.show()

sns.boxplot(x='sex' , y='Charges' , data=df)
plt.title("sex vs charges")
plt.show()

sns.boxplot(x='region' , y='Charges' , data=df)
plt.title("region vs charges")
plt.show()

sns.barplot(x='children' , y='Charges' , data=df)
plt.title("children vs charges")
plt.show()

sns.countplot(x='sex' , data=df)
sns.countplot(x='region' , data=df)
sns.countplot(x='smoker' , data=df)

# Convert Categorical Variables
# Encoding 
df['smoker_encoded'] = df['smoker'].replace({'no': 0, 'yes': 1})
df['sex_encoded']= df['sex'].replace({'female': 1, 'male':0})
df['region_encoded']= df['region'].replace({'female': 1, 'male':0})



# Load your dataset (replace with your CSV path if needed)
df = pd.read_csv('ection1_Solution/Insurance_Dataset/Section1-Question1-InsurranceData.csv')

# Features and target
X = df.drop('charges', axis=1)
y = df['charges']

# Identify categorical columns
categorical_features = ['sex', 'smoker', 'region']
numerical_features = [col for col in X.columns if col not in categorical_features]

# Preprocessing: One-hot encode categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'  # Keep numerical features as is
)

X_processed = preprocessor.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} -> RMSE: {rmse:.2f}, R2 Score: {r2:.2f}")