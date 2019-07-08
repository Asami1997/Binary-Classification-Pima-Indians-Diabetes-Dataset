from keras.models import Sequential 
from keras.layers import Dense 

import pandas as p


dataset = p.read_csv("diabetes.csv")

X = dataset.iloc[:,0:8]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

model = Sequential()
model.add(Dense(12,input_dim =8 , activation ='relu'))
model.add(Dense(12,activation = 'relu'))
model.add(Dense(1,activation = 'sigmoid'))


model.compile(loss = 'binary_crossentropy' , optimizer = 'adam', metrics=['accuracy'])


model.fit(X_train,y_train,epochs = 200, batch_size = 10)


accuracy = model.evaluate(X_test,y_test)

print(accuracy)




