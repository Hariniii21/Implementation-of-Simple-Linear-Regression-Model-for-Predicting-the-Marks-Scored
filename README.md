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


## Program and Output:

Program to implement the simple linear regression model for predicting the marks scored.

Developed by: Harini S

Register Number: 212223240048

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:\\Users\\admin\\OneDrive\\Desktop\\ML\\DATASET-20250226\\student_scores.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/bde309e4-94cd-4622-a7c7-c086b63f6020)


```
df.tail()
```
![image](https://github.com/user-attachments/assets/73a6fcf1-a680-4253-a1d2-67056da52e45)


```
x=df.iloc[:,:-1].values
x
```
![image](https://github.com/user-attachments/assets/34138173-8cb5-4080-8f34-777d0ab584e8)


```
y=df.iloc[:,1].values
y
```
![image](https://github.com/user-attachments/assets/fb590fae-d0c0-4460-9b63-26aa32a117c3)


```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)

y_pred
```
![image](https://github.com/user-attachments/assets/261601d7-5f68-437b-b2db-044b76d72be0)


```
y_test
```
![image](https://github.com/user-attachments/assets/765bcb23-3d72-4862-82e6-c68cf1ad784d)


```
plt.scatter(x_train, y_train, color="orange")
plt.plot(x_train, reg.predict(x_train), color="blue")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
![image](https://github.com/user-attachments/assets/6e8a44d4-cf3a-49fc-a99b-032ff502885c)


```
plt.scatter(x_test, y_test, color="purple")
plt.plot(x_test, reg.predict(x_test), color="green")
plt.title("Hours vs Scores (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
![image](https://github.com/user-attachments/assets/33c3e445-5836-4999-adce-9704890bc78a)


```
mse = mean_squared_error(y_test, y_pred)
print('MSE = ', mse)

mae = mean_absolute_error(y_test, y_pred)
print('MAE = ', mae)

rmse = np.sqrt(mse)
print("RMSE = ", rmse)

```
![image](https://github.com/user-attachments/assets/45128e9e-9029-4453-9b8b-8faffeadd5a9)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
