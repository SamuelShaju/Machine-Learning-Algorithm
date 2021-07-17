#MLA stands for Machile Learning Algorithms

#Linear Regression
class LinReg:
    inter = 0
    slope = 0
    def fit(x, y):                                                 #Defining a fitness function for Linear Regression model
        xmean = sum(x)/len(x)                                      #Finding means of x and y
        ymean = sum(y)/len(y)
        num = []
        den = []
        for i in range(0,len(x)):                                  #Calculation numerator and denomenator for slope
            num.append((x[i]-xmean)*(y[i]-ymean))
            den.append((x[i]-xmean)**2)
            
        LinReg.slope=(sum(num)/sum(den))
        LinReg.inter=(ymean-((LinReg.slope)*xmean))                #Calculating y-intercept
    
    def pred(x):
        return ((LinReg.slope)*x)+LinReg.inter


#Logistic Regression
from math import e
from random import randint, uniform

from numpy.lib.function_base import average, gradient
class LogReg:
    def fit(x,y):                                                  #Fitness is derived off of the fitness function of Linear regression for code reusability
        LinReg.fit(x,y)                                            #and lesser space consumption
    
    def pred(x):                                                   #pred() using logic of sigmoid function
        y = 1/(1+e**(-(LinReg.pred(x))))            #y in defined as output of the sigmoid equation
        if y<0.5:
            return 0                                               #If y is less than 0.5 there is a higher probability of it being zero
        else:                                                      #Similarly, if y >0.5(and <1.0 which is the upper limit of sigmoid function)
            return 1                                               #it has higher probability of being 1



#KNN
from math import sqrt
from statistics import mode
class KNN:
    X = []
    Y = []
    Z = []
    dist = []
    def fit(x,y,z):                                                 #Simply giving the algorithm data to memorize
        KNN.X = x
        KNN.Y = y
        KNN.Z = z
    
    def pred(x, y, k):                                              #prediction is done in 2 parts
        for i in range(0,len(KNN.X)):                               #Calculating distance from all points and storing them in dist[]
            dist = sqrt(((x-KNN.X[i])**2)+((y-KNN.Y[i])**2))
            KNN.dist.append(dist)
        
        k_fin = []
        for i in range(0,k):                                        #Choosing k number of nearest neighbours and adding their classification to k_fin
            index = KNN.dist.index(min(KNN.dist))
            k_fin.append(KNN.Z[index])
            KNN.dist.remove(KNN.dist[index])
        return mode(k_fin)                                          #Returning the most occuring k classification







#KMeans
class KMeans:
    Z = []
    Centres ={}

    def dist(p, c):                                                   #dist() is function to calculate distance between 2 points
        dist = sqrt((p[0]-c[0])**2+(p[1]-c[1])**2)
        return dist

    def fit(x, y, z):                                                 #Fitness function
        no_of_cent = len(set(z))                                      #Unique values in z == number of classifications
        groups = {}                                                   #Dictionary to store all groups
        for i in set(z):
            groups[i] = []                                            #Making a dictionary key for each groups
            for j in range(0, len(z)):
                if z[j]==i:
                    groups[i].append([x[j],y[j]])                     #Assign all points in the classification to the respective group with same key
        
        centres = {}                                                  #Dictionary to store the centres of each classification
        for i in set(z):
            sum_x = 0
            sum_y = 0
            for j in range(0,len(groups[i])):
                sum_x += groups[i][j][0]
                sum_y += groups[i][j][1]
            
            centres[i] = [(sum_x/len(z)), (sum_y/len(z))]             #Calculating the centre of the respective group by taking arithematic averages of both co-ordinates
        
        KMeans.Centres = centres                                      #Assigning them to the class variable



    def pred(x,y):
        dist = []
        centres = KMeans.Centres
        for i in centres:
            dist.append(KMeans.dist([x,y],centres[i]))                 #Making a list of distances from point to all centres
        
        index = dist.index(min(dist))+1                                #Taking the shortest distance and returning the classification it correlates to
        return index