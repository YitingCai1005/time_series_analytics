
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DD=pd.read_csv('before2017_count.csv')
AA=pd.read_csv('2007.csv')


train=DD.iloc[:,:2]

test=AA.iloc[:,:2]


look_back=5
def create_dataset(dataset, look_back):
    dataX = []
    dataY = []
    for i in range(len(dataset)-look_back-1):
        a = dataset.iloc[i, 1]
        b = dataset.iloc[i+look_back, 1]
        dataX.append(a)
        dataY.append(b)
    return np.array(dataX), np.array(dataY)


from keras.models import Sequential
from keras.layers import Dense


trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)



model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(trainX, trainY, epochs=150, batch_size=2, verbose=2)


trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f MAE)' % (trainScore, trainScore))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f MAE)' % (testScore, testScore))


Total=pd.concat((train,test))


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
 
#plotting
trainPredictPlot = np.empty_like(Total)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
 

testPredictPlot = np.empty_like(Total)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(df_dl)-1, :] = testPredict
 
plt.plot(np.array(df_dl['#EVENT']))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.title('Predicition with Keras')
plt.show()

