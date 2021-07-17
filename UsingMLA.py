from MLA import KMeans, LinReg, LogReg, KNN
from random import randint,uniform

#Self made Linear Regression algrithm
#Testing on line of slope and intercept 2
x = [0,1,2,3,5,6]                                                       #(x,y) pairs on line : y=2x+2
y = [2,4,6,8,12,14]
LinReg.fit(x,y)                                                         #Training using fitness function
x = randint(-40,40)
print('Prediction for Linear Regression test:\n[line : y = 2x + 2] \n(',x,',',LinReg.pred(x),')')                                                   #Prediction should be 10



#Self made Logistic Regression algorithm
#Cheking if number is positive or negative
print('\nPredictions for Logistic Regression test:\n[1 indicates negative value]')
x_test = []
y_test = []

for i in range(0,500):                                                     #Making 500 test cases in positive numbers
    x_test.append(randint(-20,0))
    y_test.append(1)

for i in range(0,500):                                                     #Making 500 test cases in negative numbers
    x_test.append(randint(0,20))
    y_test.append(0)


LogReg.fit(x_test,y_test)                                                 #Training using fitness function

for i in range(0,5):                                                      #Testing on 5 cases
    test = randint(-40,40)
    print(test,LogReg.pred(test))                                         #Expecting 1 for negative cases




#Self made KNN classifier algorithm
#Cheking if point is inside circle using KNN
#x2 + y2 = 16
x = []                                                                    #x and y are the training features
y = []
z = []                                                                    #z is the target feature
for i in range(0,500):
    x.append(uniform(-5,5))
    y.append(uniform(-5,5))

for i in range(0,len(x)):                                                 #making z such that, it is 1 if point lies outside the circle else it is 0
    if ((x[i]**2)+(y[i]**2)-16)<=0:
        z.append(0)
    elif ((x[i]**2)+(y[i]**2)-16)>0:
        z.append(1)

KNN.fit(x,y,z)                                                            #Training the KNN funcion
x_test = randint(-5,5)
y_test = randint(-5,5)
print('\nPrediction for KNN :\n[1 indicates point lying outside the circle]')
print(x_test, y_test, KNN.pred(x_test,y_test,5))                          



#Self made KMeans algorithm
#Checking which quadrant the point lies on
print('\nPredictions for KMeans:\n')
x = []
y = []
z = []
for i in range(0,50):                                                     #Making training cases x, y, z
    x.append(uniform(-50,50))
    y.append(uniform(-50,50))
    if x[-1]>0:                                                           #Making z corresponding with the quadrant of the point
        if y[-1]>=0:
            z.append(1)
        else:
            z.append(4)
    elif x[-1]<=0:
        if y[-1]>=0:
            z.append(2)
        else:
            z.append(3)

KMeans.fit(x,y,z)                                                        #Training the KMeans using the fit() function

for i in range(0,5):                                                     #Making the testing data and printing the result
    x = randint(-50,50)     
    y = randint(-50,50)
    print('(',x,',',y,') is in quadrant :',KMeans.pred(x,y))             #Making Prediction using the pred() function