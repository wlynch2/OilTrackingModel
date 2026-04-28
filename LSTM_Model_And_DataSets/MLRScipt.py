import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

#reading from a data set and getting the values from the two collumns we need
data = pd.read_csv("oilprice_tankercount.csv")
values = data[['WTI Spot Price (USD per Barrel)', 'Tanker Count']].values

#creating a scaler than scaling the data in the two collumns were using
scaler = MinMaxScaler()
scaledData = scaler.fit_transform(values)

#creating empty list that we will put data in
x = []
y = []

#putting the data inside of our vars
for i in range(len(scaledData) - 1):
    x.append(scaledData[i])
    y.append(scaledData[i + 1])

#converting our data to np arrays
X = np.array(x)
y = np.array(y)

#creating our testing traiing split at 80% and 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#choosing our model and putting in our training and testing
model = LinearRegression()
model.fit(X_train, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# predictions
y_pred = model.predict(X_test)

#creating the figure and adding the 3d subplot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')


#real scaling back to big boy numbers
xTestReal = scaler.inverse_transform(X_test)
yTestReal = scaler.inverse_transform(y_test)
yPredReal = scaler.inverse_transform(y_pred)

#plotting the points 
#when you uncomment the real values it plots how we want but no
#linear line this is what im stuck on HELP ME PLEASSEEE
ax.scatter(
    xTestReal[:, 0],
    xTestReal[:, 1],
    yTestReal[:, 0],
    #X_test[:, 0],
    #X_test[:, 1],
    #y_test[:, 0],
    color='blue'
)

#range of oil prices tankers and creating the grid
x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
x1, x2 = np.meshgrid(x1_range, x2_range)

#reformatting the grid and making predictions on said grid
grid = np.c_[x1.ravel(), x2.ravel()]
z = model.predict(grid)[:, 0].reshape(x1.shape)




#printing out the predictions for the random 20% of the yTest
for i in range(len(y_test)):
   
    print(f"Value {i}")
    print(f"Input: {xTestReal[i]}")
    print(f"Actual: {yTestReal[i]}")
    print(f"Predicted: {yPredReal[i][0]}")
    print("-" * 40)



#setting labels and draws a 3d surface showing the predictions
ax.plot_surface(x1, x2, z, alpha=0.5)
ax.set_xlabel("Oil Price (t)")
ax.set_ylabel("Oil Traffic (t)")
ax.set_zlabel("Oil Price (t+1)")

plt.show()