#THIS IS VERY SLOPPY BECAUSE ITS IN ONE SCRIPT BUT WHO CARESSS
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#reading the data set and getting the values of the collumns we need
data = pd.read_csv("oilprice_tankercount.csv")
values = data[['WTI Spot Price (USD per Barrel)', 'Tanker Count']].values

#scaling the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(values)

#how many past days were gonna train our model on
pastDays = 7

#two empty arrays to add teh data into
x = []
y = []

#adding all the data into these two arrays from day 31 to the length of our data set
for i in range(pastDays, len(scaled_data)):
    x.append(scaled_data[i-pastDays:i])
    y.append(scaled_data[i, 0])

#converting to a np array
x = np.array(x)
y = np.array(y)

#our traing test split
split = int(0.8 * len(x))

#creating training variables using torch 
X_train = torch.tensor(x[:split] , dtype = torch.float32)
Y_train = torch.tensor(y[:split], dtype= torch.float32)

#creating testing var using torch
X_test = torch.tensor(x[split:], dtype=torch.float32)
y_test = torch.tensor(y[split:], dtype=torch.float32)

#LSTM class whos purpose is to progress through our data set
class LSTM(nn.Module):
    
    #contrsuctor of pytroch lstm class - sets up the layers for our neural network
    def __init__(self, input = 2, hidden = 64, layers = 1, dropout = 0.2):
        super(LSTM, self).__init__()
        #takes input and outputs hidden state
        self.lstm = nn.LSTM(input, hidden, layers, batch_first=True)

        #takes lstm output and maps to a single value
        self.fc = nn.Linear(hidden, 1)

    #passing data forward and selecting final output
    def forward(self, x):
        out, ignoredState = self.lstm(x)

        out = out[:, -1, :]
        out = self.fc(out)

        return out;

#making a LSTM class object
model = LSTM()

#creating loss function
critera = nn.MSELoss()

#choosing our optimiser lr will bechanged depenidng on how the test goes
optimize = torch.optim.Adam(model.parameters(), lr = 0.005)


train_loss = []
test_loss = []
#the number of times our model will go through the entire data set
#is currently overfit for training accuracy but we can adjust when we are testing 
EPOCHS = 300

#trains the model in range of our EPOCHS - kinda self explanitory
for epoch in range(EPOCHS):
    model.train()

    #make our predictions and sees how wrong or right we are - training loss
    outputs = model(X_train)
    loss = critera(outputs.squeeze(), Y_train)



    #updates weights to improve the model
    optimize.zero_grad()
    loss.backward()
    optimize.step()

    #adding loss to a list
    train_loss.append(loss.item())

    #swithcing to testing mode and getting our testing accuarcy - helped with training 
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        t_loss = critera(test_outputs.squeeze(), y_test)
        test_loss.append(t_loss.item())


    #print loss every 5 epochs
    if ((epoch + 1 ) % 5 == 0):
        print(f"Epoch {epoch+1}, loss: {loss.item():.4f}")




#switches model to make predictions without learning
model.eval()

#disables gradient trackig and makes predictions
with torch.no_grad():
    predictions = model(X_test).squeeze().numpy()
    

# plotting testing and training loss
plt.plot(train_loss, label = "train loss")
plt.plot(test_loss, label= "test loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

#unscaling the data
pred_full = np.zeros((len(predictions), 2))
pred_full[:, 0] = predictions
predictions = scaler.inverse_transform(pred_full)[:, 0]

#unscaling the data
real = y[split:]
real_full = np.zeros((len(real), 2))
real_full[:, 0] = real
real = scaler.inverse_transform(real_full)[:, 0]

#used to plot the average diffrence from the actual point look at code below or run to see 
avgDiff = 0

#printing our predictions and are actual values
for i in range(len(predictions)):
    print(f"Day {i}: Predicted = {predictions[i]:.2f}")
    print(f"Actual= {real[i ]:.2f}")
    print("-" * 40)
    avgDiff += abs(real[i] - predictions[i])
print("Average cost from correct point: ", avgDiff/len(predictions))

#plotting data
plt.plot(real, label = "Real")
plt.plot(predictions, label = "Predicted")
plt.legend()
plt.xlabel("Day")
plt.ylabel("price")
plt.show()

#grabbing 500 untrained days to trian modle on grabbed from last 500 in data set and deleted from training data set
testData = pd.read_csv("newtestData.csv")
testValues = testData[['WTI Spot Price (USD per Barrel)', 'Tanker Count']].values

#scaling test data
scaledTest = scaler.fit_transform(testValues)

#new x and why to put data in wanted spots
newX = []
newY = []

#adding data points into new x and new y
for i in range(pastDays, len(scaledTest)):
    newX.append(scaledTest[i-pastDays:i])
    newY.append(scaledTest[i, 0])


#converting into np array
newX = np.array(newX)
newY = np.array(newY)

#creating the testing var using torch
new_X = torch.tensor(newX, dtype=torch.float32)
new_Y = torch.tensor(newY, dtype=torch.float32)

#swithching to eval mode
model.eval()

#disables gradient trackig and makes predictions
with torch.no_grad():
    preds = model(new_X).squeeze().numpy()

#uscaling data
newPredFull = np.zeros((len(preds),2))
newPredFull[:, 0] = preds
preds = scaler.inverse_transform(newPredFull)[:,0]

#unscaling data
newRealFull = np.zeros((len(new_Y), 2))
newRealFull[:, 0] = new_Y.numpy()
newReal = scaler.inverse_transform(newRealFull)[:, 0]

#getting mean average error
mae = np.mean(np.abs(preds - newReal))
print("mean absolute error", mae);

#printing predicted values and actual values 
avgDiff = 0
for i in range(len(preds)):
    actualDay = i + pastDays
    print(f"Day {actualDay}: Predicted = {preds[i]:.2f}")
    print(f"Actual= {newReal[i ]:.2f}")
    print("-" * 40)

    avgDiff += abs(newReal[i] - preds[i])

print("average cost from correct points: ", avgDiff/len(preds))

#plotting the data
plt.plot(newReal, label="Real")
plt.plot(preds, label="Predicted")
plt.legend()
plt.show()

#creating a new list for user input
userList1 = []
userList2 = []

#prompts the user to input past days price and oil traffic 
#than appends that to the respective list
for i in range(pastDays): 
    price = float(input(f"Day {i+1} - Enter oil price: "))
    traffic = float(input(f"Day {i+1} - Enter oil traffic: "))

    userList1.append(price)
    userList2.append(traffic)

#combining the list so it can be read by our model
combined = []
for i in range(pastDays):
    combined.append([userList1[i], userList2[i]])

#converting to np arrray
combined = np.array(combined)

#scailing the data
scaled = scaler.transform(combined)

inputTensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)


#switching to test mode AGAIN
model.eval()

#disables gradient trackig and makes predictions
with torch.no_grad():
    pred = model(inputTensor).item()

#unscaling data
pred_full = np.zeros((1, 2))
pred_full[0, 0] = pred
pred_price = scaler.inverse_transform(pred_full)[0, 0]

#our prediction for the next day i believe i cooked with this one 
print(f"\nPredicted oil price for next day: {pred_price:.2f}")
