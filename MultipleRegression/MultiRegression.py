
import pandas
import numpy as np
#load csv file
df=pandas.read_csv('./DataSet/HousePrice.csv')
print(df.describe())
df=df.drop(['Date'],axis=1)
X=df[list(df.columns)[:-1]]
Y=df[list(df.columns)[-1]]
print(list(df.columns)[-1])
#print(Y)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(xtrain,ytrain)
print(list(reg.predict(xtrain))[:5])
print("Accuracy Score:",reg.score(xtest,ytest))
