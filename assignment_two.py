import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def binary_step(x, thresh=10):
  return np.where(x<thresh, 0, 1)


def decision_boundary(model):
    b = model.intercept_[0]
    w1, w2 = model.coef_[0]

    c = -b / w2
    m = -w1 / w2

    xd = np.linspace(X.x_1.min(), X.x_1.max())
    yd = m * xd + c

    return xd, yd


size = 250
# NO LINEAR CORELATION
X = pd.DataFrame()
X['x_1'] = np.random.rand(size, )
X['x_2'] = np.random.randint(0, 60, size=size)


y = binary_step(10 + np.random.normal(0.0,1.0,size))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42)

model_no_linear_corr = LogisticRegression().fit(X_train, y_train)
ypred = model_no_linear_corr.predict(X)

print("Logistic Regression - No Colinearity")
print(f"Model Score: {model_no_linear_corr.score(X, y)}")
print(confusion_matrix(y, model_no_linear_corr.predict(X)))
print(classification_report(y, model_no_linear_corr.predict(X)))


pred_actual_logi = pd.DataFrame({'x_1': X['x_1'],
                'x_2': X['x_2'],
              'Ground Truth': y, 
              'Prediction': ypred})

# sns.scatterplot(x=X.x_2, y=X.x_1, style=y, hue=y, palette='deep')


xd, yd = decision_boundary(model_no_linear_corr)
for idx, val in enumerate(yd):
    if val < 0:
        yd[idx] = 0
    elif val > 50:
        yd[idx] = 50
        
sns.scatterplot(x=X.x_1, y=X.x_2, style=ypred, hue=ypred, palette='deep')
plt.plot(xd, yd)


# DATA IMBALANCE
X = pd.DataFrame()
size = 2000
X['x_1'] = np.arange(0,2,0.05)

y = 10 * X.x_1 + np.random.normal(0.0,1.0,X.size)

for idx, val in enumerate(y):
    if val < 2:
        y[idx] = int(0)
    else:
        y[idx] = int(1)
y = y.astype('int')

model_data_imbalance = LogisticRegression().fit(X, y)

print("Logistic Regression - Data Imbalance")
print(f"Model Score: {model_data_imbalance.score(X, y)}")

print(confusion_matrix(y, model_data_imbalance.predict(X)))
print(classification_report(y, model_data_imbalance.predict(X)))

sns.scatterplot(x=X.x_1, y=y, style=y, hue=y, palette='deep')
plt.legend(loc='lower right')
plt.ylabel('Target')
plt.xlabel('Feature')

ypred = model_data_imbalance.predict(X)
sns.scatterplot(x=X.x_1, y=ypred, style=ypred, hue=ypred, palette='deep')
plt.legend(loc='lower right')
ypred = model_data_imbalance.predict(X)
plt.ylabel('Predicted')
plt.xlabel('Feature')

pred_data_imbalance_logi = pd.DataFrame({'x_1': X['x_1'],
              'Ground Truth': y, 
              'Prediction': model_data_imbalance.predict(X)})


# DUMMY CLASSIFIER
model_dummy = DummyClassifier(strategy='most_frequent').fit(X, y)

print("Dummy Classifier - Data Imbalance")
print(f"Model Score: {model_dummy.score(X, y)}")
print(confusion_matrix(y, model_dummy.predict(X)))
print(classification_report(y, model_dummy.predict(X)))

pred_data_imbalance_dummy = pd.DataFrame({'x_1': X['x_1'],
              'Ground Truth': y, 
              'Prediction': model_dummy.predict(X)})



# HOLD OUT METHOD
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

X = np.arange(0,1,0.05).reshape(-1, 1)
y = 10 * X + np.random.normal(0.0,1.0,X.size).reshape(-1, 1)



intercept = []
slope = []
mean_error = []

for i in range(5):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2)
    
    model = LinearRegression().fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    
    intercept.append(model.intercept_[0])
    slope.append(model.coef_[0][0])
    mean_error.append(mean_squared_error(ytest, ypred))
    
    print('Intercept: {:.2f}\nSlope: {:.2f}\nSquared Error: {:.2f}'.format(model.intercept_[0],
                                                               model.coef_[0][0],
                                                               mean_squared_error(ytest, ypred)))
    print('\n\n')
    
    y_vals = model.intercept_ + X * model.coef_
    plt.plot(X, y_vals, label='{:.2f}'.format(mean_squared_error(ytest, ypred)))

vals = pd.DataFrame({
    'intercept': intercept,
    'slope': slope,
    'mean_error': mean_error})

plt.scatter(X, y, c='black')
plt.xlabel('Input X')
plt.ylabel('Target y')
plt.legend(title="MSE", fancybox=True)
plt.show();

import numpy as np
arr = np.arange(3.0, 5.5, 0.05)

# LAGGED OUTPUT
import pandas as pd
data = pd.DataFrame({
    'Date': ['2024-07-15',
             '2024-08-15',
             '2024-09-15',
             '2024-10-15',
             '2024-11-15',
             '2024-12-15',
             ], 
    'Product Usage': [ 1000,
                    2000,
                    4000,
                    8000,
                    16000,
                    32000
        ],
    'tag_1': [ None,
              1000,
              2000,
              4000,
              8000,
              16000,
        ],
    'tag_2': [None,
              None,
              1000,
              2000,
              4000,
              8000,
        ],
    })

print(data)
