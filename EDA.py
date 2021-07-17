import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
lmonths = LabelEncoder()


#Importing Data from csv file 'AmericanCrime'
#Choosing required features
#The size of the dataframe has been made one third for ease in computation(630000 to 20000)
#Converting catagorical values to numerical values
csv=pd.read_csv(r'AmericanCrime.csv')
df=pd.DataFrame(csv)
df['Month']=lmonths.fit_transform(df['Month'])
df['Crime_Type']=lmonths.fit_transform(df['Crime_Type'])


#Q1]Which City has recorded the highest number of killings?
#This shows that most of the crimes have occured in Los Angeles and New York, while Cook and Wayne still have a higher number of crimes than most others
print('Crimes in every city:')
print(df['City'].value_counts(sort=True)[0:10 ])

#Q2]Percentage of killing were the perpetrator was related to the victim.
#Ignoring the ones where relationship is unknown
x=df[df['Relationship']!='1']
y=len(x)
x=x[x['Relationship']=='Stranger']
x=len(x)
unkn_rel = x/y
kn_rel   = 1-unkn_rel
print('Percentage of known relations:',kn_rel*100)
print('Percentage of unknown relations:',unkn_rel*100)


#Month with the highest crimes
#December has exceptionallly low number of crimes
x = df['Month'].unique()
y = df['Month'].value_counts()
sns.swarmplot(x+1,y)
plt.title('Crime vs Month')
plt.xlabel('Month')
plt.ylabel('No. of Crimes')
plt.show()



#Plot for number of victims, race wise
#This plot shows that 'white' are 'black' people are the most frequently killed in the USA
plt.plot(['White', 'Black','Asian','Unknown','Native American'], df['Victim Race'].value_counts())
plt.title('Crime vs Victim Race')
plt.xlabel('Race')
plt.ylabel('No. of Crimes')
plt.show()
print(df['Victim Race'].value_counts())


#Plot for number of perpetrator, race wise
#This plot should have been equivalent to the previous one, but we see that there is an increase in the number of 'asian' perpetrators
plt.plot(['White', 'Black','Asian','Unknown','Native American'], df['Perpetrator Race'].value_counts())
plt.title('Crime vs Perpetrator Race')
plt.xlabel('Race')
plt.ylabel('No. of Crimes')
plt.show()
print(df['Perpetrator Race'].value_counts())


#Plot for number of victims, age wise
#Shows most victims are between the ages of 20 and 40 years
plt.hist(df['Victim Age'],bins=[0,10,20,30,40,50,60,70,80,90,100])
plt.title('Crime vs Victim Age')
plt.xlabel('Age')
plt.ylabel('No. of Crimes')
plt.show()


#Plot for number of perpetrators, age wise
#There is an increase in the number of perpetrators below the age of 20
x=df[df['Perpetrator Age']!=0]
plt.hist(x['Perpetrator Age'],bins=[0,10,20,30,40,50,60,70,80,90,100])
plt.title('Crime vs Perpetrator Age')
plt.xlabel('Age')
plt.ylabel('No. of Crimes')
plt.show()

x=df[df['Perpetrator Age']<=20]
plt.plot(['White', 'Black','Asian','Unknown','Native American'], x['Perpetrator Race'].value_counts())
plt.title('Crime vs Perpetrator Race')
plt.xlabel('Race')
plt.ylabel('No. of Crimes')
plt.show()



#Plot for number of crimes, year wise
#The findings of this graph show that crime was, on average, more rampant before the year 2000 and reached its maximum in the year 1993
plt.hist(df["Year"],bins=35)
plt.title('Crime vs Year')
plt.xlabel('Year')
plt.ylabel('No. of Crimes')
plt.show()


#Number of solved and unsolved crimes[1==solved]
print(df['Crime Solved'].value_counts())