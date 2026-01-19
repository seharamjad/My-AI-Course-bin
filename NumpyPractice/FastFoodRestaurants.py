# FastFoodRestaurants.csv
import numpy as np

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

address, latitude , longitude , name, = np.genfromtxt('NumpyPractice/FastFoodRestaurants.csv', delimiter=',', usecols=(0, 4, 5, 6), unpack=True, dtype=('U100', 'f8', 'f8', 'U100'), encoding='utf-8', skip_header=1, invalid_raise=False)

print(address)
print(latitude)
print(longitude)
print(name)

cleaned_longitude = longitude[~np.isnan(longitude)]
# FastFoodRestaurants   - statistics operations
print("US Food Restaurant DataSet longitude mean:", np.nanmean(cleaned_longitude))
print("US Food Restaurant DataSet longitude average:", np.nanmean(cleaned_longitude))  # same as mean
print("US Food Restaurant DataSet longitude std:", np.nanstd(cleaned_longitude))
print("US Food Restaurant DataSet longitude median: ",np.nanmedian(cleaned_longitude))
print("US Food Restaurant DataSet longitude percentile - 25:", np.percentile(cleaned_longitude, 25))
print("US Food Restaurant DataSet longitude percentile - 75:", np.percentile(cleaned_longitude, 75))
print("US Food Restaurant DataSet longitude percentile - 3:", np.percentile(cleaned_longitude, 3))
print("US Food Restaurant DataSet longitude max:", np.nanmax(cleaned_longitude))
print("US Food Restaurant DataSet longitude min:", np.nanmin(cleaned_longitude))

# FastFoodRestaurants - Maths operations 

print("FastFoodRestaurants latitude square: " , np.square(cleaned_longitude))
print("FastFoodRestaurants latitude sqrt: " , np.sqrt(cleaned_longitude))
print("FastFoodRestaurants latitude pow: " , np.power(cleaned_longitude,cleaned_longitude))
print("FastFoodRestaurants latitude abs: " , np.abs(cleaned_longitude))

# Perform basic Arithmetic Operations 

addition = longitude + latitude
subtraction = longitude - latitude
multiplication = longitude * latitude
division = longitude / latitude

print("FastFoodRestaurants - lat + long - Addition:", addition)
print("FastFoodRestaurants - lat + long - Subtraction:", subtraction)
print("FastFoodRestaurants - lat + long - Multiplication:", multiplication)
print("FastFoodRestaurants - lat + long - Division:", division)

#Trigonometric Functions

latitudePie = (latitude/np.pi) +1
# Calculate sine, cosine, and tangent
sine_values = np.sin(latitudePie)
cosine_values = np.cos(latitudePie)
tangent_values = np.tan(latitudePie)

print("FastFoodRestaurants - div - pie  - Sine values:", sine_values)
print("FastFoodRestaurants - div - pie Cosine values:", cosine_values)
print("FastFoodRestaurants - div - pie Tangent values:", tangent_values)

print("FastFoodRestaurants - div - pie  - Exponential values:", np.exp(latitudePie))

# Calculate the natural logarithm and base-10 logarithm
log_array = np.log(latitudePie)         # error -> invalid value encountered in log10 
log10_array = np.log10(latitudePie)

print("FastFoodRestaurants - div - pie  - Natural logarithm values:", log_array)
print("FastFoodRestaurants - div - pie  = Base-10 logarithm values:", log10_array)


#Example: Hyperbolic Sine
# Calculate the hyperbolic sine of each element
sinh_values = np.sinh(latitudePie)
print("FastFoodRestaurants latitude - div - pie   - Hyperbolic Sine values:", sinh_values)


#Hyperbolic Cosine Using cosh() Function
# Calculate the hyperbolic cosine of each element
cosh_values = np.cosh(latitudePie)
print("FastFoodRestaurants latitude - div - pie   - Hyperbolic Cosine values:", cosh_values)

#Example: Hyperbolic Tangent
# Calculate the hyperbolic tangent of each element
tanh_values = np.tanh(latitudePie)
print("FastFoodRestaurants latitude - div - pie   -Hyperbolic Tangent values:", tanh_values)

#Example: Inverse Hyperbolic Sine

# Calculate the inverse hyperbolic sine of each element
asinh_values = np.arcsinh(latitudePie)
print("FastFoodRestaurants latitude - div - pie   -Inverse Hyperbolic Sine values:", asinh_values)

#Example: Inverse Hyperbolic Cosine
# Calculate the inverse hyperbolic cosine of each element
acosh_values = np.arccosh(latitudePie)                  # invalid value encountered in arccosh
print("FastFoodRestaurants latitude - div - pie   -Inverse Hyperbolic Cosine values:", acosh_values)


#  Long Plus Lat - 2 dimentional arrary
D2LongLat = np.array([longitude,
                  latitude])

print ("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - " ,D2LongLat)


# check the dimension of array1
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - dimension" , D2LongLat.ndim)  # Number of Dimension
# Output: 2

# return total number of elements in array1
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - total number of elements" ,D2LongLat.size)
# Output: 6

# return a tuple that gives size of array in each dimension
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - gives size of array in each dimension" ,D2LongLat.shape)
# Output: (2,3)

# check the data type of array1
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - data type" ,D2LongLat.dtype) 
# Output: int64

# Splicing array            In String -> [start:end:step]
#                           In Array -> [starting:end:step , starting:end:step]
D2LongLatSlice=  D2LongLat[:1,:5]
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - Splicing array - D2LongLat[:1,:5] " , D2LongLatSlice)
D2LongLatSlice2=  D2LongLat[:1, 4:15:4]
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - Splicing array - D2LongLat[:1, 4:15:4] " , D2LongLatSlice2)



# Indexing array
D2LongLatSliceItemOnly=  D2LongLatSlice[0,1]
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - Index array - D2LongLatSlice[1,5] " , D2LongLatSliceItemOnly)
D2LongLatSlice2ItemOnly=  D2LongLatSlice2[0, 2]
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - index array - D2LongLatSlice2[0, 2] " , D2LongLatSlice2ItemOnly)


#You should use the builtin function nditer, if you don't need to have the indexes values.
for elem in np.nditer(D2LongLat):
    print(elem)

#EDIT: If you need indexes (as a tuple for 2D table), then:
for index, elem in np.ndenumerate(D2LongLat):
    print(index, elem)

"""# for loop
rows = np.shape(D2LongLat[0])[0]
cols = np.shape(D2LongLat[1])[0]
for i in range(0, (rows + 1)):
    for j in range(0, (cols + 1)):
        print (D2LongLat[i,j])
"""


# 2 x 9990 ========>>>>> 1  x  9989 - reshape
D2LongLat1TO298 = np.reshape(D2LongLat, (2, 9990))
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - np.reshape(D2LongLat, (1, 9989)) : " , D2LongLat1TO298)
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - np.reshape(D2LongLat, (1, 9989)) : Size " , D2LongLat1TO298.size)
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - np.reshape(D2LongLat, (1, 9989)) : ndim " , D2LongLat1TO298.ndim)
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - np.reshape(D2LongLat, (1, 9989)) : shape " , D2LongLat1TO298.shape)
print("FastFoodRestaurants Long Plus Lat - 2 dimentional arrary - np.reshape(D2LongLat, (1, 9989)) : ndim " , D2LongLat1TO298.ndim)




print()


