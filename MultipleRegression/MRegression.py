from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

boston = load_boston()
X = boston.data
print(type(X))
print(boston.keys())
print('Feature names:',boston['feature_names'])
Y = boston.target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, shuffle= True)

lineReg = LinearRegression()
lineReg.fit(X_train, y_train)
print(lineReg.score(X_test, y_test ))