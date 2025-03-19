# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Harini S
RegisterNumber:212223240048

```

```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset = pd.read_csv("C:/Users/Keert/Downloads/Ml exp/student_scores.csv")
print(dataset.head())
print(dataset.tail())
```

<img width="161" alt="image" src="https://github.com/user-attachments/assets/d4152761-5462-490e-b1cf-493765d99eef" />

```
dataset.info()
```

<img width="284" alt="image" src="https://github.com/user-attachments/assets/5c9cace7-95a8-4724-a857-5007a8557dd0" />

```
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,-1].values
print(y)
```

<img width="461" alt="image" src="https://github.com/user-attachments/assets/87efb915-5e7f-4885-8370-fc74eb50264b" />

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
```

<img width="90" alt="image" src="https://github.com/user-attachments/assets/2b0f324f-ab92-4014-aa1f-d5e18c175d81" />

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
x_test.shape
```

<img width="88" alt="image" src="https://github.com/user-attachments/assets/ff7f4310-2d78-479e-9cff-482279a157b9" />

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
x_test.shape
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
```

<img width="223" alt="image" src="https://github.com/user-attachments/assets/d9e68d2a-9cfa-47ad-8aa3-43d65179d3c1" />

```
y_pred=reg.predict(x_test)
print(y_pred)
print(y_test)
```

<img width="607" alt="image" src="https://github.com/user-attachments/assets/091e8c7d-3665-494b-ad64-545c2ea967db" />

```
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,reg.predict(x_train),color="red")
plt.title('Traning set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,reg.predict(x_test),color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
```

```
mse = mean_squared_error(y_test, y_pred)
print('MSE = ', mse)
```

<img width="184" alt="image" src="https://github.com/user-attachments/assets/307295c1-0088-4bec-acde-869604aa5cde" />

```
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
```

<img width="209" alt="image" src="https://github.com/user-attachments/assets/7cfba909-b2b6-4002-bb96-57806ba3605d" />

```
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
<img width="260" alt="image" src="https://github.com/user-attachments/assets/e6412fd9-e631-4ad2-a96a-af7f0b3eb7cf" />



## Output:

### Training Set:
<img width="416" alt="image" src="https://github.com/user-attachments/assets/ac125bc4-6138-4ea3-a13d-189d5a9581ba" />

### Testing Set:
<img width="406" alt="image" src="https://github.com/user-attachments/assets/fcf2a1d9-b77f-4e8e-8d30-d75afc06cf1c" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
