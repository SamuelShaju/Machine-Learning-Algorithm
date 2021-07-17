import pandas as pd
from sklearn import cluster
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
lmonths = LabelEncoder()
#Asking User how many records to use
#SVM was restricted to 50000 records due to time required to calculate was much higher
n=int(input('Enter number of records to use:\n[0 if using full dataset]\n[Use less than 50000 to use SVM]\n'))
if n==0:
    n=638454

Acc={}

#Importing Data from csv file 'AmericanCrime'
#Choosing required features
#The size of the dataframe has been made one third for ease in computation(630000 to 20000)
#Converting catagorical values to numerical values
csv=pd.read_csv(r'AmericanCrime.csv')
df=pd.DataFrame(csv)
df=df[['Year','Month','Crime_Type','Victim Sex','Victim Age','Perpetrator Age','Crime Solved','Relationship']]
df['Month']=lmonths.fit_transform(df['Month'])
df['Crime_Type']=lmonths.fit_transform(df['Crime_Type'])
df['Relationship']=lmonths.fit_transform(df['Relationship'])



#To remove bias number of solved and unsolved cases were kept the same
#print(df['Crime Solved'].value_counts())
zero_df = df[df['Crime Solved']==0][0:n//2]
one_df  = df[df['Crime Solved']==1][0:n//2]
df = zero_df.append(one_df)



#Splitting the dataset for training and testing
#Target Feature: Crime Solved
X = df[['Year','Month','Crime_Type','Victim Sex','Victim Age','Perpetrator Age','Relationship']]
Y = df['Crime Solved']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)




#KNN Method
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
Acc['KNN'] = (accuracy_score(Y_pred_knn, Y_test))




#Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred_LR = lr.predict(X_test)
Acc['Logistic Regression'] = (accuracy_score(Y_pred_LR, Y_test))



#Random Forest Method
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
Y_pred_RF = rf.predict(X_test)
Acc['Random Forest'] = (accuracy_score(Y_pred_RF, Y_test))


#KMeans Method
km = KMeans()
km.fit(X_train, Y_train)
Y_pred_KM = km.predict(X_test)
Acc['KMeans'] = (accuracy_score(Y_pred_KM, Y_test))



#SVM Method
#Restricted to 50000 records
if n<=50000:
    svc = SVC()
    svc.fit(X_train, Y_train)
    Y_pred_SVM = svc.predict(X_test)
    Acc['SVM'] = (accuracy_score(Y_pred_SVM, Y_test))

#Printing most accurate method
print(max(Acc),'method has highest accuracy:', Acc[max(Acc)]*100,'%')