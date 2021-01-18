
#Data processing
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

#Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

#Model Selection
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV

#Model validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#Visualization
import csv
import seaborn as sns
from matplotlib import pyplot as plt

#misc
import warnings
warnings.filterwarnings("ignore") 
from datetime import datetime

#______________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________________________________

training_data = pd.read_csv("train.csv",header=0)
testing_data = pd.read_csv("test.csv",header=0)

data = training_data.append(testing_data)
data1 =data.copy(deep=True)
data = data.reset_index()
# data = [training_data, testing_data]
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 10000)

# print(data.info())
#exit()

'''process title'''
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.')
data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
data['Title'] = data['Title'].replace('Mlle', 'Miss')
data['Title'] = data['Title'].replace('Ms', 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')
data['Title'] = data['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5})



'''Process Mother'''
data['mother'] = 0
data.loc[(data.Title==3) & (data.Parch>=1), 'mother'] = 1

'''process title1'''
data['Title1'] = data['Name'].str.extract(' ([A-Za-z]+)\.')
data['Title1'] = data['Title1'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Master'], 'Other')
data['Title1'] = data['Title1'].replace('Mlle', 'Miss')
data['Title1'] = data['Title1'].replace('Ms', 'Miss')
data['Title1'] = data['Title1'].replace('Mme', 'Mrs')
data['Title1'] = data['Title1'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Other": 4})
'''
g = sns.factorplot(x='Title1',y="Survived",data=data,kind="bar", size = 6 , palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
plt.show()
exit()
'''

data.loc[ (data.Title1==1) , 'Title_tuned' ] = 1
data.loc[ (data.Title1==2) , 'Title_tuned' ] = 3
data.loc[ (data.Title1==3) , 'Title_tuned' ] = 4
data.loc[ (data.Title1==4) , 'Title_tuned' ] = 2

'''
g = sns.factorplot(x="Title1",y="Survived",data=data,kind="bar", size = 6 , palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
plt.show()
exit()
'''
#print(data['Title1'])
#exit()
'''Process Age'''
data['Age2'] = data['Age'].copy(deep =True)

Age_Mean = data[['Sex','Age']].groupby( by=['Sex'] ).mean().values.tolist()
data.loc[(data.Age.isnull())&(data.Sex=='female'),'Age'] = Age_Mean[0][0]
data.loc[(data.Age.isnull())&(data.Sex=='male'),'Age'] = Age_Mean[1][0]

Age_Mean1 = data[['Title','Age2']].groupby( by=['Title'] ).mean().values.tolist()
data.loc[(data.Age2.isnull())&(data.Title==1),'Age2'] = Age_Mean1[0][0]
data.loc[(data.Age2.isnull())&(data.Title==2),'Age2'] = Age_Mean1[1][0]
data.loc[(data.Age2.isnull())&(data.Title== 3),'Age2'] = Age_Mean1[2][0]
data.loc[(data.Age2.isnull())&(data.Title==4),'Age2'] = Age_Mean1[3][0]
data.loc[(data.Age2.isnull())&(data.Title==5),'Age2'] = Age_Mean1[4][0]
data['Age_range'] = pd.cut(data['Age2'],9)
data['Age_label'] = LabelEncoder().fit_transform(data['Age_range'])

#print(data['Age'].isnull().sum())
#exit()
data.loc[(data.Age<15),'Age1'] = 2
data.loc[(data.Age>=15)&(data.Age<30),'Age1'] = 1
data.loc[(data.Age>=30)&(data.Age<45),'Age1'] = 3
data.loc[(data.Age>=45)&(data.Age<60),'Age1'] = 4
data.loc[(data.Age>=60),'Age1'] = 5

data.loc[ (data.Age1==1) , 'Age_tuned' ] = 3
data.loc[ (data.Age1==2) , 'Age_tuned' ] = 1
data.loc[ (data.Age1==3) , 'Age_tuned' ] = 4
data.loc[ (data.Age1==4) , 'Age_tuned' ] = 2
data.loc[ (data.Age1==5) , 'Age_tuned' ] = 5
'''    

'''
#print(data['Age1'].isnull().sum())
#exit()
'''process children'''
data['is_Children'] = 0
data.loc[(data.Age <= 14), 'is_Children'] = 1


'''Process fare'''
data['Fare'].fillna(value=data1['Fare'].median(), inplace=True)
data.loc[(data.Fare<10),'Fare1'] = 1
data.loc[(data.Fare>=10)&(data.Fare<20),'Fare1'] = 2
data.loc[(data.Fare>=20)&(data.Fare<30),'Fare1'] = 3
data.loc[(data.Fare>=30)&(data.Fare<50),'Fare1'] = 4
data.loc[(data.Fare>=50)&(data.Fare<70),'Fare1'] = 5
data.loc[(data.Fare>=70),'Fare1'] = 6

data.loc[ (data.Fare1==1) , 'Fare_tuned' ] = 1
data.loc[ (data.Fare1==2) , 'Fare_tuned' ] = 3
data.loc[ (data.Fare1==3) , 'Fare_tuned' ] = 4
data.loc[ (data.Fare1==4) , 'Fare_tuned' ] = 2
data.loc[ (data.Fare1==5) , 'Fare_tuned' ] = 5
data.loc[ (data.Fare1==6) , 'Fare_tuned' ] = 6


'''Process fare label'''
data['Fare_range'] = pd.qcut(data['Fare'],4)
data['Fare_label'] = LabelEncoder().fit_transform(data['Fare_range'])

'''Process ticket'''

#  print(data['Ticket'].describe())
#  print(data['Ticket'].isnull().sum())
data['Ticket1'] = data['Ticket'].copy()
data.loc[(data.Ticket1.str.startswith('A')),'Ticket1'] = '4000000'
data.loc[(data.Ticket1.str.startswith('C')),'Ticket1'] = '4000001'
data.loc[(data.Ticket1.str.startswith('F')),'Ticket1'] = '4000002'
data.loc[(data.Ticket1.str.startswith('L')),'Ticket1'] = '4000003'
data.loc[(data.Ticket1.str.startswith('P')),'Ticket1'] = '4000004'
data.loc[(data.Ticket1.str.startswith('S.')),'Ticket1'] = '4000005'
data.loc[(data.Ticket1.str.startswith('SC/')),'Ticket1'] = '4000006'
data.loc[(data.Ticket1.str.startswith('SC ')),'Ticket1'] = '4000006'
data.loc[(data.Ticket1.str.startswith('SCO')),'Ticket1'] = '4000006'
data.loc[(data.Ticket1.str.startswith('ST')),'Ticket1'] = '4000007'
data.loc[(data.Ticket1.str.startswith('SO')),'Ticket1'] = '4000007'
data.loc[(data.Ticket1.str.startswith('W')),'Ticket1'] = '4000008'
data.loc[(data.Ticket1.str.startswith('SW')),'Ticket1'] = '4000008'
'''
print(data['Ticket1'])
print(data['Ticket1'])
'''
data.loc[(data.Ticket1.astype(float)<10000),'Ticket2'] = 1
data.loc[(data.Ticket1.astype(float)>=10000)&(data.Ticket1.astype(float)<20000),'Ticket2'] = 2
data.loc[(data.Ticket1.astype(float)>=20000)&(data.Ticket1.astype(float)<30000),'Ticket2'] = 3
data.loc[(data.Ticket1.astype(float)>=30000)&(data.Ticket1.astype(float)<100000),'Ticket2'] = 4
data.loc[(data.Ticket1.astype(float)>=100000)&(data.Ticket1.astype(float)<200000),'Ticket2'] = 5
data.loc[(data.Ticket1.astype(float)>=200000)&(data.Ticket1.astype(float)<300000),'Ticket2'] = 6
data.loc[(data.Ticket1.astype(float)>=300000)&(data.Ticket1.astype(float)<400000),'Ticket2'] = 7
data.loc[(data.Ticket1.astype(float)>=3000000)&(data.Ticket1.astype(float)<4000000),'Ticket2'] = 8
data.loc[(data.Ticket1.astype(float)==4000000),'Ticket2'] = 9
data.loc[(data.Ticket1.astype(float)==4000001),'Ticket2'] = 10
data.loc[(data.Ticket1.astype(float)==4000002),'Ticket2'] = 11
data.loc[(data.Ticket1.astype(float)==4000003),'Ticket2'] = 12
data.loc[(data.Ticket1.astype(float)==4000004),'Ticket2'] = 13
data.loc[(data.Ticket1.astype(float)==4000005),'Ticket2'] = 14
data.loc[(data.Ticket1.astype(float)==4000006),'Ticket2'] = 15
data.loc[(data.Ticket1.astype(float)==4000007),'Ticket2'] = 16
data.loc[(data.Ticket1.astype(float)==4000008),'Ticket2'] = 17


data.loc[ (data.Ticket2==1) , 'Ticket_tuned' ] = 9
data.loc[ (data.Ticket2==2) , 'Ticket_tuned' ] = 17
data.loc[ (data.Ticket2==3) , 'Ticket_tuned' ] = 10
data.loc[ (data.Ticket2==4) , 'Ticket_tuned' ] = 11
data.loc[ (data.Ticket2==5) , 'Ticket_tuned' ] = 13
data.loc[ (data.Ticket2==6) , 'Ticket_tuned' ] = 12
data.loc[ (data.Ticket2==7) , 'Ticket_tuned' ] = 3
data.loc[ (data.Ticket2==8) , 'Ticket_tuned' ] = 5
data.loc[ (data.Ticket2==9) , 'Ticket_tuned' ] = 1
data.loc[ (data.Ticket2==10) , 'Ticket_tuned' ] = 8
data.loc[ (data.Ticket2==11) , 'Ticket_tuned' ] = 15
data.loc[ (data.Ticket2==12) , 'Ticket_tuned' ] = 6
data.loc[ (data.Ticket2==13) , 'Ticket_tuned' ] = 16
data.loc[ (data.Ticket2==14) , 'Ticket_tuned' ] = 2
data.loc[ (data.Ticket2==15) , 'Ticket_tuned' ] = 14
data.loc[ (data.Ticket2==16) , 'Ticket_tuned' ] = 7
data.loc[ (data.Ticket2==17) , 'Ticket_tuned' ] = 4

'''Process family'''
data['FamilySize'] = data['SibSp'] +  data['Parch'] + 1

data.loc[ (data.FamilySize==1) , 'FamilySize_tuned' ] = 4
data.loc[ (data.FamilySize==2) , 'FamilySize_tuned' ] = 5
data.loc[ (data.FamilySize==3) , 'FamilySize_tuned' ] = 6
data.loc[ (data.FamilySize==4) , 'FamilySize_tuned' ] = 7
data.loc[ (data.FamilySize==5) , 'FamilySize_tuned' ] = 3
data.loc[ (data.FamilySize==6) , 'FamilySize_tuned' ] = 2
data.loc[ (data.FamilySize==7) , 'FamilySize_tuned' ] = 5
data.loc[ (data.FamilySize==8) , 'FamilySize_tuned' ] = 1
data.loc[ (data.FamilySize==11) , 'FamilySize_tuned' ] = 1

'''process family_label'''
data['FamilySize_range'] = pd.cut(data['FamilySize'],4)
data['FamilySize_label'] = LabelEncoder().fit_transform(data['FamilySize_range'])

'''Provess Alone'''
data['Alone'] = 0
data.loc[ data.FamilySize==1, 'Alone' ] = 1

'''process Age_class'''
data['Age_Pclass'] = data['Pclass'] * data['Age_tuned']

data.loc[ (data.Age_Pclass==1) , 'Age_Pclass_tuned' ] = 11
data.loc[ (data.Age_Pclass==2) , 'Age_Pclass_tuned' ] = 10
data.loc[ (data.Age_Pclass==3) , 'Age_Pclass_tuned' ] = 9
data.loc[ (data.Age_Pclass==4) , 'Age_Pclass_tuned' ] = 8
data.loc[ (data.Age_Pclass==5) , 'Age_Pclass_tuned' ] = 4
data.loc[ (data.Age_Pclass==6) , 'Age_Pclass_tuned' ] = 6
data.loc[ (data.Age_Pclass==8) , 'Age_Pclass_tuned' ] = 7
data.loc[ (data.Age_Pclass==9) , 'Age_Pclass_tuned' ] = 5
data.loc[ (data.Age_Pclass==10) , 'Age_Pclass_tuned' ] = 3
data.loc[ (data.Age_Pclass==12) , 'Age_Pclass_tuned' ] = 1
data.loc[ (data.Age_Pclass==15) , 'Age_Pclass_tuned' ] = 2


'''Process Cabin'''
data['Cabin'].fillna(value="NA", inplace=True)
data['have_Cabin'] = 1
data.loc[ data.Cabin=="NA", 'have_Cabin' ] = 0


'''Process Embarked'''
data.loc[(data.Embarked.isnull()),'Embarked'] = data['Embarked'].mode()[0]
data.loc[ (data.Embarked=='Q'), 'Embarked1' ] = 1
data.loc[ (data.Embarked=='S'), 'Embarked1' ] = 2
data.loc[ (data.Embarked=='C'), 'Embarked1' ] = 3

data.loc[ (data.Embarked1==1) , 'Embarked_tuned' ] = 2
data.loc[ (data.Embarked1==2) , 'Embarked_tuned' ] = 1
data.loc[ (data.Embarked1==3) , 'Embarked_tuned' ] = 3
'''

'''

'''Process sex_class1'''
data['Sex_Pclass1'] = 0
data.loc[ (data.Sex=='female') & (data.Pclass==1), 'Sex_Pclass1' ] = 1.0
data.loc[ (data.Sex=='female') & (data.Pclass==2), 'Sex_Pclass1' ] = 2.0
data.loc[ (data.Sex=='female') & (data.Pclass==3), 'Sex_Pclass1' ] = 3.0
data.loc[ (data.Sex=='male') & (data.Pclass==1), 'Sex_Pclass1' ] = 4.0
data.loc[ (data.Sex=='male') & (data.Pclass==2), 'Sex_Pclass1' ] = 5.0
data.loc[ (data.Sex=='male') & (data.Pclass==3), 'Sex_Pclass1' ] = 6.0
'''
g = sns.factorplot(x="Sex_Pclass1",y="Survived",data=data,kind="bar", size = 6 , palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
plt.show()
exit()
'''
'''Process sex_class'''
data['Sex_Pclass'] = 0
data.loc[ (data.Sex=='female') & (data.Pclass==1), 'Sex_Pclass' ] = "female 1st"
data.loc[ (data.Sex=='female') & (data.Pclass==2), 'Sex_Pclass' ] = "female 2nd"
data.loc[ (data.Sex=='female') & (data.Pclass==3), 'Sex_Pclass' ] = "female 3rd"
data.loc[ (data.Sex=='male') & (data.Pclass==1), 'Sex_Pclass' ] = "male 1st"
data.loc[ (data.Sex=='male') & (data.Pclass==2), 'Sex_Pclass' ] = "male 2nd"
data.loc[ (data.Sex=='male') & (data.Pclass==3), 'Sex_Pclass' ] = "male 3rd"

'''Process Embarked_sex'''
data['Embarked_sex'] = 0
data.loc[ (data.Sex=='female') & (data.Embarked=='Q'), 'Embarked_sex' ] = 2
data.loc[ (data.Sex=='female') & (data.Embarked=='S'), 'Embarked_sex' ] = 3
data.loc[ (data.Sex=='female') & (data.Embarked=='C'), 'Embarked_sex' ] = 1
data.loc[ (data.Sex=='male') & (data.Embarked=='Q'), 'Embarked_sex' ] = 3
data.loc[ (data.Sex=='male') & (data.Embarked=='S'), 'Embarked_sex' ] = 4
data.loc[ (data.Sex=='male') & (data.Embarked=='C'), 'Embarked_sex' ] = 2


'''Process Sex'''
data['Sex'] = data.Sex.map({'male':2,'female':1})


'''
ax = sns.countplot(x="Sex_Pclass1", hue="Survived", data=data) 
plt.show()
exit()
'''
data['Female_1&2'] = 0
data.loc[(data.Sex_Pclass == "female 1st"), 'Female_1&2'] = 1.0
data.loc[(data.Sex_Pclass == "female 2nd"), 'Female_1&2'] = 1.0

data['Male_2&3'] = 0
data.loc[(data.Sex_Pclass == "male 3rd"), 'Male_2&3'] = 1.0
data.loc[(data.Sex_Pclass == "male 2nd"), 'Male_2&3'] = 1.0

'''Process share_ticket'''
data['ShareTicket'] = 0
share_ticket = data.pivot_table(index=['Ticket'], aggfunc='size')
count = 0
dup = []
for item in share_ticket:
    if share_ticket.values[count] > 1:
        dup.append(share_ticket.index[count])
    count = count + 1
    
count = 0   
for dataframe in data['Ticket']:
    if dataframe in dup:
        data['ShareTicket'][count] = 1
    count = count + 1
print(data['Ticket1'].describe())
print(data['Fare_label'].describe())

#______________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________________________________

'''
g = sns.factorplot(x="Ticket_tuned",y="Survived",data=data,kind="bar", size = 6 , palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
plt.show()
exit()
'''
'''
data1 = data[['Survived','Fare_tuned','Title_tuned','Sex_Pclass1','have_Cabin','FamilySize_tuned','Ticket_tuned']]


correlation = data1.corr()
sns.heatmap(correlation, annot=True, cbar=True, cmap="RdYlGn")
plt.show()

exit()

'''
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
#                               Total features  (10 classic features +19 feature engineered features = 29)     
#                                                               
#   Name                Description                                                             Sample
#
#
#   Classical Features
#
#   PassengerId         : Id of passenger
#   Survived            : Survival status of passenger                                          0,1
#   Pclass              : Passenger class level                                                 1,2,3
#   Name                : Passenger Full name with title                                        Thomas, Master. Assad Alexander
#   Sex                 : Gender                                                                female = 1, male = 2 
#   Age                 : continuous age data (Missing data filled with sex mean)               0.83,11,23,25,35,55
#   SibSp               : Number of sister, brother, spouse on ship                             1,2,3,4
#   Parch               : Number of parent, children on ship                                    1,2,3,4
#   Ticket              : Raw ticket information, replce non number to number based by group    1234, 2132, CO/656428
#   Fare                : Raw fare data (Missing filled with mean)                              10,20,5.6,84.12
#   Cabin               : Cabin number                                                          C54, A32
#   Embarked            : Raw embarked data                                                     Q,S,C
#
#   Feature engineered features
#
#   Embarked1           : Processed embarked data                                               1,2,3
#   Age1                : Convert age into 5 ranges (0-15, 15-30, 30-45, 45-60, >60)            1,2,3,4,5
#   Class_age           : Class multuply with age group                                         1,2,3,4,5,6
#   Title               : Split passenger title into 5 groups (Jonathan)                        1,2,3,4,5
#   Title1              : Split passenger title into 4 groups (Gary)                            1,2,3,4
#   Mother              : Married and have children onboard female                              0,1
#   is_Children         : Aged under 14 children                                                0,1
#   have_Cabin          : Does passenger have cabin                                             0,1
#   Fare1               : Split Fare into 5 group (0-10, 10-20, 20-30, 30-50, 50-70, >70)       1,2,3,4,5               
#   Fare_label          : Split Fare into 4 group using pd.cut                                  1,2,3,4                        
#   Ticket1             : Split Ticket into 17 groups                                           1-15
#   ShareTicket         : Is the ticket shared by more than one passenger                       0,1
#   Age_Pclass          : Age group multiply with class                                         1-15
#   Sex_Pclass          : Sex multuply with pclass                                              1,2,3,4,5,6
#   Female_1&2          : First and second class female                                         0,1
#   Male_2&3            : Second and third class male                                           0,1
#   FamilySize          : Number of SibSp + Parch                                               1-11
#   FamilySize_label    : Split FamilySize into 4 group by pd.cut                               1,2,3,4
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------


#______________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________________________________

train = data[ pd.notnull(data.Survived) ]
test = data[ pd.isnull(data.Survived) ]
index = test['PassengerId']
index = index.reset_index()


#______________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________________________________


label0 = train[['Survived']]
test0 = test[['Fare_tuned','Title_tuned','Sex_Pclass1','have_Cabin','FamilySize_tuned','Ticket_tuned']]
train0 = train[['Fare_tuned','Title_tuned','Sex_Pclass1','have_Cabin','FamilySize_tuned','Ticket_tuned']]


model = ["Random Forest","SVC","Logistic Regression","KNN","Decision Tree","Naive Bayesian","Xgboost Random Forest"]
acc = []

clf = RandomForestClassifier(n_estimators=250)
clf.fit(train0, label0)
y_pred_random_forest = clf.predict(test0)
acc_random_forest = round(clf.score(train0, label0) * 100, 2)
print ("Random Forest: " + str(acc_random_forest) + '%')
acc.append(acc_random_forest)

clf = SVC()
clf.fit(train0, label0)
y_pred_svc = clf.predict(test0)
acc_svc = round(clf.score(train0, label0) * 100, 2)
print ("SVC: " + str(acc_svc) + '%')
acc.append(acc_svc)

clf = LogisticRegression()
clf.fit(train0, label0)
y_pred_svc = clf.predict(test0)
acc_log_reg = round(clf.score(train0, label0) * 100, 2)
print ("Logistic Regression: " + str(acc_log_reg) + '%')
acc.append(acc_svc)

clf = KNeighborsClassifier()
clf.fit(train0, label0)
y_pred_knn = clf.predict(test0)
acc_knn = round(clf.score(train0, label0) * 100, 2)
print ("KNN: " + str(acc_knn) + '%')
acc.append(acc_knn)

clf = DecisionTreeClassifier()
clf.fit(train0, label0)
y_pred_decision_tree = clf.predict(test0)
acc_decision_tree = round(clf.score(train0, label0) * 100, 2)
print ("Decision Tree: " + str(acc_decision_tree) + '%')
acc.append(acc_decision_tree)

clf = GaussianNB()
clf.fit(train0, label0)
y_pred_gnb = clf.predict(test0)
acc_gnb = round(clf.score(train0, label0) * 100, 2)
print ("Naive Bayesian: " + str(acc_gnb) + '%')
acc.append(acc_gnb)

xgbrf = xgb.XGBClassifier()
xgbrf.fit(train0, label0)
y_pred_xgbrf = xgbrf.predict(test0)
acc_xgb = round(xgbrf.score(train0, label0) * 100, 2)
print ("Xgboost Random Forest: " + str(acc_xgb) + '%')
acc.append(acc_xgb)

accs = {"Model":model,"Training Accuracy":acc}
df = pd.DataFrame(accs)
df = df.sort_values(by='Training Accuracy',ascending=False)
print(df)


#______________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________________________________

CV_10 = 0
CV_20 = 0
CV_30 = 0
temp = 100
# First Trial
test_1 = test[['Sex','Pclass','Age_Pclass','Sex_Pclass1','Fare1','FamilySize','SibSp','Parch','Title','have_Cabin','Embarked1','Alone']]
train_1 = train[['Sex','Pclass','Age_Pclass','Sex_Pclass1','Fare1','FamilySize','SibSp','Parch','Title','have_Cabin','Embarked1','Alone']]
label_1 = train[['Survived']]
X1, X1_test, y1, y1_test = train_test_split(train_1,  label_1, test_size=0.2, random_state =1, shuffle=True)
#rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=5, max_features='auto', max_leaf_nodes=40, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None, oob_score=True, random_state=1, verbose=0, warm_start=False)
rf = RandomForestClassifier()
rf.fit(X1,y1)                
pred = rf.predict(X1_test)
tn, fp, fn, tp = confusion_matrix(y1_test,pred).ravel()
acc_test = round((tn+tp)/(tn+fp+fn+tp),6)
cv_scorce = cross_val_score(rf, train_1, label_1, cv=10)
CV_10 = cv_scorce.mean()
cv_scorce = cross_val_score(rf, train_1, label_1, cv=20)
CV_20 = cv_scorce.mean()
cv_scorce = cross_val_score(rf, train_1, label_1, cv=30)
CV_30 = cv_scorce.mean()
print("-"*temp)
print("Features used: ",list(train_1.columns.values))
print("First Trial. The testing accuracy is: ",acc_test, "\nThe 10-fold mean score :",CV_10, "\nThe 20-fold mean score :",CV_20,"\nThe 30-fold mean score :",CV_30)
y_pred_random_forest = rf.predict(test_1)
with open('output1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for m in y_pred_random_forest:
        writer.writerow([int(m)])

# Second Trial
test_2 = test[['Pclass','Age_Pclass','Title1','Sex','SibSp','Parch','FamilySize','Alone','Ticket1','Fare1','have_Cabin','Embarked1','Male_2&3','Female_1&2']]
train_2 = train[['Pclass','Age_Pclass','Title1','Sex','SibSp','Parch','FamilySize','Alone','Ticket1','Fare1','have_Cabin','Embarked1','Male_2&3','Female_1&2']]
label_2 = train[['Survived']]
X1, X1_test, y1, y1_test = train_test_split(train_2,  label_2, test_size=0.2, random_state =1, shuffle=True)
rf = RandomForestClassifier()
rf.fit(X1,y1)                
pred = rf.predict(X1_test)
tn, fp, fn, tp = confusion_matrix(y1_test,pred).ravel()
acc_test = round((tn+tp)/(tn+fp+fn+tp),6)
cv_scorce = cross_val_score(rf, train_2, label_2, cv=10)
CV_10 = cv_scorce.mean()
cv_scorce = cross_val_score(rf, train_2, label_2, cv=20)
CV_20 = cv_scorce.mean()
cv_scorce = cross_val_score(rf, train_1, label_1, cv=30)
CV_30 = cv_scorce.mean()
print("-"*temp)
print("Features used: ",list(train_2.columns.values))
print("Second Trial. The testing accuracy is: ",acc_test, "\nThe 10-fold mean score :",CV_10, "\nThe 20-fold mean score :",CV_20,"\nThe 30-fold mean score :",CV_30)
y_pred_random_forest = rf.predict(test_2)
with open('output2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for m in y_pred_random_forest:
        writer.writerow([int(m)])


# Third Trial
test_3 = test[['Sex','Pclass','Sex_Pclass1','Fare1','FamilySize','SibSp','Parch','Title1','have_Cabin','Embarked1','Alone']]
train_3 = train[['Sex','Pclass','Sex_Pclass1','Fare1','FamilySize','SibSp','Parch','Title1','have_Cabin','Embarked1','Alone']]
label_3 = train[['Survived']]
X1, X1_test, y1, y1_test = train_test_split(train_3,  label_3, test_size=0.2, random_state =1, shuffle=True)
rf = RandomForestClassifier()
rf.fit(X1,y1)                
pred = rf.predict(X1_test)
tn, fp, fn, tp = confusion_matrix(y1_test,pred).ravel()
acc_test = round((tn+tp)/(tn+fp+fn+tp),6)
cv_scorce = cross_val_score(rf, train_3, label_3, cv=10)
CV_10 = cv_scorce.mean()
cv_scorce = cross_val_score(rf, train_3, label_3, cv=20)
CV_20 = cv_scorce.mean()
cv_scorce = cross_val_score(rf, train_1, label_1, cv=30)
CV_30 = cv_scorce.mean()
print("-"*temp)
print("Features used: ",list(train_3.columns.values))
print("Third Trial. The testing accuracy is: ",acc_test, "\nThe 10-fold mean score :",CV_10, "\nThe 20-fold mean score :",CV_20,"\nThe 30-fold mean score :",CV_30)
y_pred_random_forest = rf.predict(test_3)
with open('output3.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for m in y_pred_random_forest:
        writer.writerow([int(m)])


# Forth Trial
test_4 = test[['Pclass','Age_Pclass','Title1','Sex','SibSp','Parch','FamilySize','Alone','Ticket1','Fare1','have_Cabin','Embarked1']]
train_4 = train[['Pclass','Age_Pclass','Title1','Sex','SibSp','Parch','FamilySize','Alone','Ticket1','Fare1','have_Cabin','Embarked1']]
label_4 = train[['Survived']]
X1, X1_test, y1, y1_test = train_test_split(train_4,  label_4, test_size=0.2, shuffle=True)
rf = RandomForestClassifier()
rf.fit(X1,y1)                
pred = rf.predict(X1_test)
tn, fp, fn, tp = confusion_matrix(y1_test,pred).ravel()
acc_test = round((tn+tp)/(tn+fp+fn+tp),6)
cv_scorce = cross_val_score(rf, train_4, label_4, cv=10)
CV_10 = cv_scorce.mean()
cv_scorce = cross_val_score(rf, train_4, label_4, cv=20)
CV_20 = cv_scorce.mean()
cv_scorce = cross_val_score(rf, train_1, label_1, cv=30)
CV_30 = cv_scorce.mean()
print("-"*temp)
print("Features used: ",list(train_4.columns.values))
print("Forth Trial. The testing accuracy is: ",acc_test, "\nThe 10-fold mean score :",CV_10, "\nThe 20-fold mean score :",CV_20,"\nThe 30-fold mean score :",CV_30)
y_pred_random_forest = rf.predict(test_4)
with open('output4.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for m in y_pred_random_forest:
        writer.writerow([int(m)])


#Fifth Trial
test_5 = test[['Title1','Sex_Pclass1','have_Cabin','ShareTicket','Fare1']]
train_5 = train[['Title1','Sex_Pclass1','have_Cabin','ShareTicket','Fare1']]
label_5 = train[['Survived']]
X1, X1_test, y1, y1_test = train_test_split(train_5,  label_5, test_size=0.2, shuffle=True)
rf = RandomForestClassifier()
rf.fit(X1,y1)                
pred = rf.predict(X1_test)
tn, fp, fn, tp = confusion_matrix(y1_test,pred).ravel()
acc_test = round((tn+tp)/(tn+fp+fn+tp),6)
cv_scorce = cross_val_score(rf, train_5, label_5, cv=10)
CV_10 = cv_scorce.mean()
cv_scorce = cross_val_score(rf, train_5, label_5, cv=20)
CV_20 = cv_scorce.mean()
cv_scorce = cross_val_score(rf, train_1, label_1, cv=30)
CV_30 = cv_scorce.mean()
print("-"*temp)
print("Features used: ",list(train_5.columns.values))
print("Fifth Trial. The testing accuracy is: ",acc_test, "\nThe 10-fold mean score :",CV_10, "\nThe 20-fold mean score :",CV_20,"\nThe 30-fold mean score :",CV_30)
y_pred_random_forest = rf.predict(test_5)
with open('output5.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for m in y_pred_random_forest:
        writer.writerow([int(m)])

# Sixth Trial
test_6 = test[['Fare_tuned','Title_tuned','Sex_Pclass1','have_Cabin','FamilySize_tuned','ShareTicket','Ticket_tuned','Alone']]
train_6 = train[['Fare_tuned','Title_tuned','Sex_Pclass1','have_Cabin','FamilySize_tuned','ShareTicket','Ticket_tuned','Alone']]
label_6 = train[['Survived']]
X1, X1_test, y1, y1_test = train_test_split(train_6,  label_6, test_size=0.2, shuffle=True)
#rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=5, max_features='auto', max_leaf_nodes=40, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None, oob_score=True, random_state=1, verbose=0, warm_start=False)
rf = RandomForestClassifier()
rf.fit(X1,y1)                
pred = rf.predict(X1_test)
tn, fp, fn, tp = confusion_matrix(y1_test,pred).ravel()
acc_test = round((tn+tp)/(tn+fp+fn+tp),6)
cv_scorce = cross_val_score(rf, train_6, label_6, cv=10)
CV_10 = cv_scorce.mean()
cv_scorce = cross_val_score(rf, train_6, label_6, cv=20)
CV_20 = cv_scorce.mean()
cv_scorce = cross_val_score(rf, train_1, label_1, cv=30)
CV_30 = cv_scorce.mean()
print("-"*temp)
print("Features used: ",list(train_6.columns.values))
print("Sixth Trial. The testing accuracy is: ",acc_test, "\nThe 10-fold mean score :",CV_10, "\nThe 20-fold mean score :",CV_20,"\nThe 30-fold mean score :",CV_30)
y_pred_random_forest = rf.predict(test_6)
with open('output6.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for m in y_pred_random_forest:
        writer.writerow([int(m)])

# Seventh Trial
test_7 = test[['Fare_tuned','Title_tuned','Sex_Pclass1','have_Cabin','FamilySize_tuned','Ticket_tuned']]
train_7 = train[['Fare_tuned','Title_tuned','Sex_Pclass1','have_Cabin','FamilySize_tuned','Ticket_tuned']]
label_7 = train[['Survived']]
X1, X1_test, y1, y1_test = train_test_split(train_7,  label_7, test_size=0.2, shuffle=True)
#rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=5, max_features='auto', max_leaf_nodes=40, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None, oob_score=True, random_state=1, verbose=0, warm_start=False)
rf = RandomForestClassifier()
rf.fit(X1,y1)                
pred = rf.predict(X1_test)
tn, fp, fn, tp = confusion_matrix(y1_test,pred).ravel()
acc_test = round((tn+tp)/(tn+fp+fn+tp),6)
cv_scorce = cross_val_score(rf, train_7, label_7, cv=10)
CV_10 = cv_scorce.mean()
cv_scorce = cross_val_score(rf, train_7, label_7, cv=20)
CV_20 = cv_scorce.mean()
cv_scorce = cross_val_score(rf, train_1, label_1, cv=30)
CV_30 = cv_scorce.mean()
print("-"*temp)
print("Features used: ",list(train_7.columns.values))
print("Seventh Trial. The testing accuracy is: ",acc_test, "\nThe 10-fold mean score :",CV_10, "\nThe 20-fold mean score :",CV_20,"\nThe 30-fold mean score :",CV_30)
y_pred_random_forest = rf.predict(test_7)
with open('output7.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for m in y_pred_random_forest:
        writer.writerow([int(m)])


train_rfecv = train[['Fare_tuned','Title_tuned','Sex_Pclass1','have_Cabin','FamilySize_tuned','ShareTicket','Ticket_tuned','Fare1','Title1','Sex','Pclass','FamilySize','SibSp','Parch','Embarked1','Alone','Male_2&3','Female_1&2','Age1','Age2', 'mother','is_Children','FamilySize_label','Age_label']]
label_rfecv = train['Survived']
svc = SVC(kernel="linear")
rfecv = RFECV(estimator=svc, step=1, cv=20, scoring='accuracy')
rfecv.fit(train_rfecv, label_rfecv)
print("-"*temp)
print("Running RFECV to find best combination features.")
print("Features used: ",list(train_rfecv.columns.values))
print("Optimal number of features : %d" % rfecv.n_features_)
print("Optimal features :",rfecv.support_)
print("Optimal rank :",rfecv.ranking_)

#RFECV Trial
test_8 = test[['Title_tuned','Sex_Pclass1','FamilySize_tuned','Title1','Sex','Pclass','FamilySize','Parch']]
train_8 = train[['Title_tuned','Sex_Pclass1','FamilySize_tuned','Title1','Sex','Pclass','FamilySize','Parch']]
label_8 = train[['Survived']]
X1, X1_test, y1, y1_test = train_test_split(train_8,  label_8, test_size=0.2, shuffle=True)
#rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=5, max_features='auto', max_leaf_nodes=40, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None, oob_score=True, random_state=1, verbose=0, warm_start=False)
rf = RandomForestClassifier()
rf.fit(X1,y1)                
pred = rf.predict(X1_test)
tn, fp, fn, tp = confusion_matrix(y1_test,pred).ravel()
acc_test = round((tn+tp)/(tn+fp+fn+tp),6)
cv_scorce = cross_val_score(rf, train_8, label_8, cv=10)
CV_10 = cv_scorce.mean()
cv_scorce = cross_val_score(rf, train_8, label_8, cv=20)
CV_20 = cv_scorce.mean()
cv_scorce = cross_val_score(rf, train_1, label_1, cv=30)
CV_30 = cv_scorce.mean()
print("-"*temp)
print("Features used: ",list(train_8.columns.values))
print("RFECV Trial. The testing accuracy is: ",acc_test, "\nThe 10-fold mean score :",CV_10, "\nThe 20-fold mean score :",CV_20,"\nThe 30-fold mean score :",CV_30)
y_pred_random_forest = rf.predict(test_8)
with open('testing_output9.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for m in y_pred_random_forest:
        writer.writerow([int(m)])


#RFECV_modified Trial
test_9 = test[['Title_tuned','Sex_Pclass1','FamilySize_tuned','Parch']]
train_9 = train[['Title_tuned','Sex_Pclass1','FamilySize_tuned','Parch']]
label_9 = train[['Survived']]
X1, X1_test, y1, y1_test = train_test_split(train_9,  label_9, test_size=0.2, shuffle=True)
#rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=5, max_features='auto', max_leaf_nodes=40, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None, oob_score=True, random_state=1, verbose=0, warm_start=False)
rf = RandomForestClassifier()
rf.fit(X1,y1)                
pred = rf.predict(X1_test)
tn, fp, fn, tp = confusion_matrix(y1_test,pred).ravel()
acc_test = round((tn+tp)/(tn+fp+fn+tp),6)
cv_scorce = cross_val_score(rf, train_9, label_9, cv=10)
CV_10 = cv_scorce.mean()
cv_scorce = cross_val_score(rf, train_9, label_9, cv=20)
CV_20 = cv_scorce.mean()
cv_scorce = cross_val_score(rf, train_1, label_1, cv=30)
CV_30 = cv_scorce.mean()
print("-"*temp)
print("Features used: ",list(train_9.columns.values))
print("RFECV_modified Trial. The testing accuracy is: ",acc_test, "\nThe 10-fold mean score :",CV_10, "\nThe 20-fold mean score :",CV_20,"\nThe 30-fold mean score :",CV_30)
y_pred_random_forest = rf.predict(test_9)
with open('RFECV_output_1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for m in y_pred_random_forest:
        writer.writerow([int(m)])

#RFECV_modified_1 Trial
test_10 = test[['Title_tuned','Sex_Pclass1','FamilySize_tuned','Parch','SibSp']]
train_10 = train[['Title_tuned','Sex_Pclass1','FamilySize_tuned','Parch','SibSp']]
label_10 = train[['Survived']]
X1, X1_test, y1, y1_test = train_test_split(train_10,  label_10, test_size=0.2, shuffle=True)
#rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=5, max_features='auto', max_leaf_nodes=40, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None, oob_score=True, random_state=1, verbose=0, warm_start=False)
rf = RandomForestClassifier()
rf.fit(X1,y1)                
pred = rf.predict(X1_test)
tn, fp, fn, tp = confusion_matrix(y1_test,pred).ravel()
acc_test = round((tn+tp)/(tn+fp+fn+tp),6)
cv_scorce = cross_val_score(rf, train_10, label_10, cv=10)
CV_10 = cv_scorce.mean()
cv_scorce = cross_val_score(rf, train_10, label_10, cv=20)
CV_20 = cv_scorce.mean()
cv_scorce = cross_val_score(rf, train_1, label_1, cv=30)
CV_30 = cv_scorce.mean()
print("-"*temp)
print("Features used: ",list(train_10.columns.values))
print("RFECV_modified_2 Trial. The testing accuracy is: ",acc_test, "\nThe 10-fold mean score :",CV_10, "\nThe 20-fold mean score :",CV_20,"\nThe 30-fold mean score :",CV_30)
y_pred_random_forest = rf.predict(test_10)
with open('EFECV_output_2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for m in y_pred_random_forest:
        writer.writerow([int(m)])
    
#______________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________________________________
#fine tune 

#Using GridSearchCV
rfc=RandomForestClassifier()

param_grid = { 
    'n_estimators': [i for i in range(50,110,10)],
    'max_depth' : [j for j in range(1,7,1)],
    'max_leaf_nodes' : [k for k in range(8,33,1)],
    'criterion' :['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=10, verbose=2, n_jobs=1)
CV_rfc.fit(train_7, label_7)
print(CV_rfc.best_params_)
print(CV_rfc.best_score_)

#[Parallel(n_jobs=1)]: Done 18000 out of 18000 | elapsed: 21.7min finished
#{'criterion': 'entropy', 'max_depth': 5, 'max_leaf_nodes': 32, 'n_estimators': 100}
#0.8249687890137329
#Kaggle score = 0.767943


param_grid = { 
    'n_estimators': [i for i in range(50,110,10)],
    'max_depth' : [j for j in range(1,7,1)],
    'max_leaf_nodes' : [k for k in range(8,33,1)],
    'criterion' :['gini', 'entropy'],
    'random_state' : [1]
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=10, verbose=2, n_jobs=1)
CV_rfc.fit(train_7, label_7)
print(CV_rfc.best_params_)
print(CV_rfc.best_score_)
#[Parallel(n_jobs=1)]: Done 18000 out of 18000 | elapsed: 21.7min finished
#{'criterion': 'gini', 'max_depth': 5, 'max_leaf_nodes': 20, 'n_estimators': 60, 'random_state': 1}
#0.8260799001248438
#Kaggle score = 0.767943




#Using human intelligence search 
tn = 0 
fp = 0 
fn = 0
tp = 0 

label8 = train[['Survived']]
test8 = test[['Fare_tuned','Title_tuned','Sex_Pclass1','have_Cabin','FamilySize_tuned','Ticket_tuned']]
train8 = train[['Fare_tuned','Title_tuned','Sex_Pclass1','have_Cabin','FamilySize_tuned','Ticket_tuned']]

X1, X1_test, y1, y1_test = train_test_split(train8,  label8, test_size=0.2, random_state =1, shuffle=True)

rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=3, max_features='auto', max_leaf_nodes=32, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None, oob_score=True, random_state=1, verbose=0, warm_start=False)
rf.fit(X1,y1)                
pred = rf.predict(X1_test)
tn, fp, fn, tp = confusion_matrix(y1_test,pred).ravel()
acc_test = round((tn+tp)/(tn+fp+fn+tp),6) 
y_pred_random_forest = rf.predict(test8)
with open('testing_output1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for m in y_pred_random_forest:
        writer.writerow([int(m)])

rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=3, max_features='auto', max_leaf_nodes=32, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=60, n_jobs=None, oob_score=True, random_state=1, verbose=0, warm_start=False)
rf.fit(X1,y1)                
pred = rf.predict(X1_test)
tn, fp, fn, tp = confusion_matrix(y1_test,pred).ravel()
acc_test = round((tn+tp)/(tn+fp+fn+tp),6) 
y_pred_random_forest = rf.predict(test8)
with open('testing_output2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for m in y_pred_random_forest:
        writer.writerow([int(m)])
        
rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=3, max_features='auto', max_leaf_nodes=32, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=70, n_jobs=None, oob_score=True, random_state=1, verbose=0, warm_start=False)
rf.fit(X1,y1)                
pred = rf.predict(X1_test)
tn, fp, fn, tp = confusion_matrix(y1_test,pred).ravel()
acc_test = round((tn+tp)/(tn+fp+fn+tp),6) 
y_pred_random_forest = rf.predict(test8)
with open('testing_output3.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for m in y_pred_random_forest:
        writer.writerow([int(m)])

rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=3, max_features='auto', max_leaf_nodes=32, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=80, n_jobs=None, oob_score=True, random_state=1, verbose=0, warm_start=False)
rf.fit(X1,y1)                
pred = rf.predict(X1_test)
tn, fp, fn, tp = confusion_matrix(y1_test,pred).ravel()
acc_test = round((tn+tp)/(tn+fp+fn+tp),6) 
y_pred_random_forest = rf.predict(test8)
with open('testing_output4.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for m in y_pred_random_forest:
        writer.writerow([int(m)])
        
rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=3, max_features='auto', max_leaf_nodes=32, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=90, n_jobs=None, oob_score=True, random_state=1, verbose=0, warm_start=False)
rf.fit(X1,y1)                
pred = rf.predict(X1_test)
tn, fp, fn, tp = confusion_matrix(y1_test,pred).ravel()
acc_test = round((tn+tp)/(tn+fp+fn+tp),6) 
y_pred_random_forest = rf.predict(test8)
with open('testing_output5.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for m in y_pred_random_forest:
        writer.writerow([int(m)])        

rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=3, max_features='auto', max_leaf_nodes=31, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=60, n_jobs=None, oob_score=True, random_state=1, verbose=0, warm_start=False)
rf.fit(X1,y1)                
pred = rf.predict(X1_test)
tn, fp, fn, tp = confusion_matrix(y1_test,pred).ravel()
acc_test = round((tn+tp)/(tn+fp+fn+tp),6) 
y_pred_random_forest = rf.predict(test8)
with open('testing_output6.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for m in y_pred_random_forest:
        writer.writerow([int(m)])        
        
        
#______________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________________________________
#______________________________________________________________________________________________________________________________________________
        
# Final Solution

label_final = train[['Survived']]
test_final = test[['Fare_tuned','Title_tuned','Sex_Pclass1','have_Cabin','FamilySize_tuned','Ticket_tuned']]
train_final = train[['Fare_tuned','Title_tuned','Sex_Pclass1','have_Cabin','FamilySize_tuned','Ticket_tuned']]

X1, X1_test, y1, y1_test = train_test_split(train_final,  label_final, test_size=0.2, random_state =1, shuffle=True)



'''
#Solution 1 
rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=3, max_features='auto', max_leaf_nodes=32,
    min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=60,
    n_jobs=None, oob_score=True, random_state=1, verbose=0, warm_start=False)
rf.fit(X1,y1)                
pred = rf.predict(X1_test)
tn, fp, fn, tp = confusion_matrix(y1_test,pred).ravel()
acc_test = round((tn+tp)/(tn+fp+fn+tp),6) 
y_pred_random_forest = rf.predict(test_final)
with open('Final_solution_1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for m in y_pred_random_forest:
        writer.writerow([int(m)])        
'''        
# Solution 2
rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=3, max_features='auto', max_leaf_nodes=31, 
    min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=60, 
    n_jobs=None, oob_score=True, random_state=1, verbose=0, warm_start=False)
rf.fit(X1,y1)                
pred = rf.predict(X1_test)
tn, fp, fn, tp = confusion_matrix(y1_test,pred).ravel()
acc_test = round((tn+tp)/(tn+fp+fn+tp),6) 
y_pred_random_forest = rf.predict(test_final)
with open('Final_solution_2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for m in y_pred_random_forest:
        writer.writerow([int(m)])        
                
