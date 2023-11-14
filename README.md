# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file
## Program:
```
Developed By : LOKESH RAHUL V V
Reg No : 212222100024
```

DATA PREPROCESSING BEFORE FEATURE SELECTION:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/titanic_dataset.csv')
df.head()
```
![282422251-d6b5b843-e914-4778-bb45-ea97a6f4098a](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex-07/assets/119559694/92f903ba-16ff-41e0-ad86-a083a6096b98)


CHECKING NULL VALUES:
````python
df.isnull().sum()
````
![282422310-69876b7c-8239-43e9-b116-736a41a2d45f](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex-07/assets/119559694/a5a546c7-f01c-45ee-badf-86a8d57fcf79)

DROPPING UNWANTED DATAS:

````python
df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Parch',axis=1,inplace=True)
df.head()
````
![282422477-5ee0835e-b287-460c-87ea-ea4f920ab248](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex-07/assets/119559694/d98ca570-08b1-4b26-97b2-a97d2a25242e)

DATA CLEANING:
```python
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()
```

![282422572-0466abc2-07b8-4a1c-85a7-6252021b3863](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex-07/assets/119559694/86f550f0-729a-46d4-b577-c688b95a8765)

REMOVING OUTLIERS:
Before
```python
plt.title("Dataset with outliers")
df.boxplot()
plt.show()
```

![282422740-fa1f06a9-280c-4146-ad8b-fe126976b3ec](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex-07/assets/119559694/f7e41946-c236-4bf0-b2b2-cfaf82bb3cf6)

After
```python
cols = ['Age','SibSp','Fare']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()
```

![282422846-afbbd455-5c7e-4225-ac74-faa6a3eddccd](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex-07/assets/119559694/16c1f4de-e017-48b2-bfee-b275bae5f27a)

FEATURE SELECTION:
```pthon
from sklearn.preprocessing import OrdinalEncoder
climate = ['C','S','Q']
en= OrdinalEncoder(categories = [climate])
df['Embarked']=en.fit_transform(df[["Embarked"]])
df.head()
```
![282422925-cff6aac9-216e-49c3-be5a-ea21c16c2a32](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex-07/assets/119559694/78f7e7b1-c8cb-44ee-9e75-f823de533db3)

```python
from sklearn.preprocessing import OrdinalEncoder
gender = ['male','female']
en= OrdinalEncoder(categories = [gender])
df['Sex']=en.fit_transform(df[["Sex"]])
df.head()
```
![282422989-06c53273-b8d2-4cdb-a50e-81650296413e](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex-07/assets/119559694/68572f7e-140e-4b78-b2da-2410dcd19db7)

````python
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])
df.head()
````
![282423083-a3696bd7-7863-49dc-aade-2031e62a17b6](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex-07/assets/119559694/0d2a7fb1-84f9-4b82-b01c-31d6c4e93f28)

````python
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)
````
````python
df1=pd.DataFrame()
df1["Survived"]=np.sqrt(df["Survived"])
df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])
df1["Sex"]=np.sqrt(df["Sex"])
df1["Age"]=df["Age"]
df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])
df1["Fare"],parameters=stats.yeojohnson(df["Fare"])
df1["Embarked"]=df["Embarked"]
df1.skew()
````
![282423180-e98cd744-9038-4cb6-9ac9-b55b070354db](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex-07/assets/119559694/1d5f8599-c03d-4dc4-87ed-a3d289e2eeb6)

```python
import matplotlib
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
```
X = df1.drop("Survived",1) 
y = df1["Survived"] 

 FILTER METHOD:
```python
plt.figure(figsize=(7,6))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)
plt.show()
```
![282423297-d7b7662a-d08b-4304-ac76-f010da979f5d](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex-07/assets/119559694/361b2585-0bfc-4bea-853e-00bf274dafff)

HIGHLY CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED:
```python
cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features
```

![282423401-e7e2bbf2-9b7a-42a4-abec-154741ff5cfa](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex-07/assets/119559694/158e6a41-f620-49ae-ba46-bcb7cdad8982)

BACKWARD ELIMINATION:
```python
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
```
![282423500-f1a0f451-2035-4b2a-b7e9-09d11a771a11](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex-07/assets/119559694/20a153ae-7171-4a00-a7d2-c1451547c2c2)

GPTIMUM NUMBER OF FEATURES THAT HAVE HIGH ACCURACY:
```python
nof_list=np.arange(1,6)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,step=nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
```

![282423607-500ddf02-1cd4-4723-b90f-4bfe7cde8c7f](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex-07/assets/119559694/d323d7e9-8508-4a00-b199-7879d46a7e28)

FINAL SET OF FEATURE:
```python
cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, step=2)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)
```

![282423719-e5549170-70ee-4334-8af9-d206957209e4](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex-07/assets/119559694/c79838aa-0819-4ec6-aa14-8c35cd57c822)

EMBEDDED METHOD:
````python
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (5.0, 5.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()
````
![282423939-e008452f-c3f8-4b14-b78c-46ce6fa83942](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex-07/assets/119559694/e031896f-0d88-452c-b309-8ff5e41c350b)

## RESULT:
Thus, the various feature selection techniques have been performed on a given dataset successfully.
