# FastFoodRestaurants.csv 
import pandas as pd

df = pd.read_csv('PandasPractice/FastFoodRestaurants.csv',delimiter=",")
print(df)

print("df - DataTypes",df.dtypes)
print("df.info(): ",df.info())

# Display the Last three Rows 
print("Last Three Rows: ")
print(df.tail(3))

# Display the First Three Rows
print("First Three Rows: ")
print(df.head(3))
print()

# Summary of Satistics of DataFrame using describe() method 
print("Summary of Satistics of DataFrame using describe() method: ",df.describe)

# Counting the Rows and Columns in DataFrame using shape(). It returns the no. of Rows and Columns enclosed in a tuple. 
print("Counting the Rows and Columns of DataFrame using shape(): ", df.shape)

# Access the name column 
longitude = df["longitude"] 
print("Access the Name Column: df : ",longitude)
print()

# Access Multiple Columns 
longitude_latitude = df[['longitude', 'latitude']]
print("Access Multiple Columns: df : ")
print(longitude_latitude)
print()

"""There are Four Primary Ways to select rows with .loc. These include: 
Selecting a single row 
Selecting a multiple row 
Selecting a slice of rows 
Conditional row selection"""

# Case 1 : using .loc - default case - starts here
# Selecting a Single row using .loc 

second_row = df.loc[1]
print("Selecting a single row using .loc")
print(second_row)
print()

# Selecting the multiple rows using .loc 
second_row2 = df.loc[[1, 3]]
print("Selecting the Multiple Rows using .loc")
print(second_row2)
print()

# Selecting the Slice of Rows using .loc 
second_row3 = df.loc[1:5]
print("Selecting the Slice of Rows using .loc")
print(second_row3)
print()

# Conditional Selection of rows using .loc 
second_row4 = df.loc[df['longitude'] == 'latitude']
print("Conditional Selection of rows using .loc")
print(second_row4)
print()

# Selecting the Single column using .loc 
second_row5 = df.loc[:1,'longitude']
print("Selecting the Single Column using .loc")
print(second_row5)
print()

# Selecting the Multiple Columns using .loc 
second_row6 = df.loc[:1, ['longitude', 'latitude']]
print("Slecting the multiple columns using .loc")
print(second_row6)
print()

# Selecting the Slice of the Column using .loc 
second_row7 = df.loc[:1, 'country' : 'longitude']
print("Selecting the Slice of the column using .loc")
print(second_row7)
print()

# Combined Rows and columns Selection using .loc 
second_row8 = df.loc[df['longitude'] == 'latitude', 'city': 'longitude']
print("Combined Rows and Columns Selection using .loc")
print(second_row8)
print()

# Case 1 - using .loc - Default case - ends here 
print("# Case 2 : using .loc with index_col - starts here")
# Case 2: using .loc with index_col - starts here 
# Second cycle - with index_col as "latitude"
# why second cycle - Note index, - index_col = "latitude"

df_index_col = pd.read_csv('PandasPractice/FastFoodRestaurants.csv', delimiter=',', index_col='latitude')

print(df_index_col)
print(df_index_col.dtypes)
print(df_index_col.info())
# Second cycle - with index_col as latitude 

print(df_index_col.index)
# Selecting a Single row using .loc 
second_row = df_index_col.loc[39.53255]                     
print("# Selecting the Single Row using .loc")
print(second_row)
print()


# Selecting the Multiple Rows using .loc 
second_row2 = df_index_col.loc[[39.53255, 38.62736 ]]        # 3 row->38.62736
print("# Selecting the Multiple Rows using .loc")
print(second_row2)
print()

'''Index([     44.9213,     39.53255,     38.62736,     44.95008,     39.35155,
            39.4176,     39.86969,     34.00598,     33.91335,     36.06107,
       ...
          39.186102, 45.512000098,    42.701994,     41.64076,    36.058624,
          33.415257,      42.2173,     40.18919,     33.78864,    33.860074],
      dtype='float64', name='latitude', length=10000)'''
#Selecting the Slice of Rows using .loc
second_row3 = df_index_col.loc[39.53255 : 44.95008]
print("# Selecting the slice Rows of using .loc")
print(second_row3)
print()

# Conditional Selection of Rows using .loc             ## Empty Data Set 
second_row4 = df_index_col.loc[df_index_col["keys"] == 'province']
print("# Conditional Selectional of Rows using .loc")
print(second_row4)
print()

# Selecting a single column using .loc
second_row5 = df_index_col.loc[:44.95008, ['longitude']]
print("# Selecting a Single column using .loc")
print(second_row5)
print()


# Selecting the multiple columns using .loc 
second_row6 = df_index_col.loc[:44.95008,['longitude', 'keys']]
print("# Selecting the multiple columns using .loc")
print(second_row6)
print()


# Selecting the Slice of the column using .loc 
second_row7 = df_index_col.loc[:44.95008, 'address': 'longitude']
print("# Selecting the Slice of the rows using .loc")
print(second_row7)
print()


# Combined row and column selection using .loc
second_row8 = df_index_col.loc[df_index_col["longitude"] == 'keys', 'address': 'longitude'] 
print("# Combined row and column selection using .loc")
print(second_row8)
print()

# Case 2 : using .loc with index_col  -  ends here

print("# Case 3 : Using .iloc - starts here")
# Case 3 : Using .iloc - starts here

# Selecting a Single row using .iloc
second_row = df_index_col.iloc[0]
print("Selecting a Single Row using .iloc")
print(second_row)
print()

# Selecting multiple rows using .iloc 
second_row2 = df_index_col.iloc[[1, 3, 5]]
print("Selecting multiple rows using .iloc")
print(second_row2)
print()

# Selecting Slice of rows using .iloc 
second_row3 = df_index_col.iloc[2:5]
print("Selecting slice of rows using .iloc")
print(second_row3)
print()

# Selecting a Single column using .iloc 
second_row5 = df_index_col.iloc[:,2]
print("Selecting a Single column using .iloc")
print(second_row5)
print()

# Selecting the multiple columns using .iloc 
second_row6 = df_index_col.iloc[:,[2,4]]
print("Selecting the multiple columns using .iloc")
print(second_row6)
print()

# Selecting Slice of column using .iloc 
second_row7 = df_index_col.iloc[:,2:4]
print("Selecting Slice of columns using .iloc")
print(second_row7)
print()

# Combined rows and columns selection using .iloc 
second_row8 = df_index_col.iloc[[1,3,5], 2:4]
print("Combined rows and column selection using .iloc")
print(second_row8)
print()

# Case 3: Using .iloc - ends here 

# Next Run 
print("Next Run")

""""Pandas DataFrame Manipulation
DataFrame manipulation in Pandas involves editing and modifying existing DataFrames. Some common DataFrame manipulation operations are:

Adding rows/columns
Removing rows/columns
Renaming rows/columns"""

#Add a New Row to a Pandas DataFrame
# add a new row
# Copy array from list and add to DataFrame
# 324 Main St,Massena,US,us/ny/massena/324mainst/-1161002137,44.9213,-74.89021,McDonald's,13662,NY,"http://mcdonalds.com,http://www.mcdonalds.com/?cid=RF:YXT_FM:TP::Yext:Referral"

df.loc[len(df.index)] = ["324 Main St","Massena","US","us/ny/massena/324mainst/-1161002137",44.9213,-74.89021,"McDonald's",13662,"NY","http://mcdonalds.com,http://www.mcdonalds.com/?cid=RF:YXT_FM:TP::Yext:Referral"]
print("Modified DataFrame - add new row:")
print(df)
print()

# Remove the Rows/Columns from Pandas Dataset 

# Delete Row with index 1
df.drop(1, axis=0, inplace=True)
# delete row with index 1
df.drop(index=2, inplace=True)
# delete rows with index 3 and 5
df.drop([3, 5], axis=0, inplace=True)
# display the modified DataFrame after deleting rows
print("Modified DataFrame - Remove Rows:")
print(df)

# delete country column
df.drop('country', axis=1, inplace=True)
# delete keys column
df.drop(columns='keys', inplace=True)
# delete longitude and name columns
df.drop(['longitude', 'name'], axis=1, inplace=True)
# display the modified DataFrame after deleting rows
print("Modified DataFrame -  delete country ,keys , longitude , name , column :")
print(df)

#Rename Labels in a DataFrame
# rename column 'Name' to 'First_Name'
df.rename(columns= {'province': 'province_Changed'}, inplace=True)
# rename columns 'Age' and 'City'
df.rename(mapper= {'city': 'city_Changed', 'latitude':'latitude_Changed'}, axis=1, inplace=True)
# display the DataFrame after renaming column
print("Modified DataFrame  - Rename Labels :")
print(df)

#Example: Rename Row Labels
# rename column one index label
df.rename(index={0: 7}, inplace=True)
# rename columns multiple index labels
df.rename(mapper={1: 10, 2: 100}, axis=0, inplace=True)
# display the DataFrame after renaming column
print("Modified DataFrame - Rename Row - 0  >>> 7 , 1 >>> 10 , 2 >>> 100  Labels:")
print(df)


#query() to Select Data
#The query() method in Pandas allows you to select data using a more SQL-like syntax.

# select the rows where the age is greater than 25
selected_rows = df.query('address == \'city \'')

print(selected_rows.to_string())
print("Length of the selectes row: ",len(selected_rows))


# sort DataFrame by price in ascending order
# sorted_df = df.sort_values(by='price')
# print(sorted_df.to_string(index=False))              input must be values 

# 1. Sort DataFrame by 'Age' and then by 'Score' (Both in ascending order)
# df1 = df.sort_values(by=['price', 'location_id'])

# print("Sorting by 'price' (ascending) and then by 'location_id' (ascending):\n")
# print(df1.to_string(index=False))



#Pandas groupby
#In Pandas, the groupby operation lets us group data based on specific columns. This means we can divide a DataFrame into smaller groups based on the values in these columns.

# group the DataFrame by the location_id column and
# calculate the sum of price for each category
print("Now the Column Names:", df.columns.tolist())

grouped = df.groupby('province_Changed')['latitude_Changed'].sum()

print(grouped.to_string())
print("grouped :" , len(grouped))

""""Pandas Data Cleaning
Data cleaning means fixing and organizing messy data. Pandas offers a wide range of tools and functions to help us clean and preprocess our data effectively.
"""
# use dropna() to remove rows with any missing values
df_cleaned = df.dropna()
print("Cleaned Data:\n",df_cleaned)


# filling NaN values with 0
df.fillna(0, inplace=True)

print("\nData after filling NaN with 0:\n", df)



# create a list named data
data = [2, 4, 6, 8]
# create Pandas array using data
array1 = pd.array(data)
print(array1)
"""<IntegerArray>
[2, 4, 6, 8]
Length: 4, dtype: Int64"""


# creating a pandas.array of integers
int_array = pd.array([1, 2, 3, 4, 5], dtype='int')
print(int_array)
print()

#Pandas Reshape
#In Pandas, reshaping data refers to the process of converting a DataFrame from one format to another for better data visualization and analysis.
#Pandas provides multiple methods like pivot(), pivot_table(), stack(), unstack() and melt() to reshape data. We can choose the method based on our analysis requirement.


# to be continued....