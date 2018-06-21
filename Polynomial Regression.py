
import pandas as pd
df=pd.read_csv("/home/indresh/PycharmProjects/MLCoursera/DataSet/Salaries.csv")
x=df.iloc[:,1:2].values
y=df.iloc[:,2].values
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=3)
poly_x=poly.fit_transform(x)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(poly_x,y)
import matplotlib.pyplot as plt
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(poly.fit_transform(x)),color='blue')
plt.show()

