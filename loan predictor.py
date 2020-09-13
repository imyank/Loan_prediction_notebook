#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df_train=pd.read_csv("train_ctrUa4K.csv")
df_test=pd.read_csv("test_lAUu6dG.csv")
df_train1=df_train.copy()
df_test1=df_test.copy()
combine=[df_train,df_test]
pd.set_option("display.max_columns",500)


# In[5]:


df_train.head()


# In[10]:


df_train['Loan_Amount_Term'].unique()
#df_train.isnull().sum()


# In[7]:


df_train.corr()


# In[6]:


List=['Urban', 'Rural', 'Semiurban']
for dataset in combine:
    Mean_property_data=dataset.groupby('Property_Area')['Credit_History'].agg(pd.Series.mode)
    Mean_property_data1=dataset.groupby('Property_Area')['LoanAmount'].mean()
for dataset in combine:
    for i in List:
        d={i:Mean_property_data[i]}
        d1={i:Mean_property_data1[i]}
        s=dataset.Property_Area.map(d)
        s1=dataset.Property_Area.map(d1)
        dataset.Credit_History=dataset.Credit_History.combine_first(s)
        dataset.LoanAmount=dataset.LoanAmount.combine_first(s1)
for dataset in combine:
    dataset['Dependents']=dataset['Dependents'].fillna(dataset['Dependents'].mode()[0])


# In[7]:


df_train.isnull().sum()


# In[8]:


n_nan_train=[f for f in df_train.columns if df_train[f].isnull().sum()>1 and df_train[f].dtypes!='O']
n_nan_test=[f for f in df_test.columns if df_test[f].isnull().sum()>1 and df_test[f].dtypes!='O']
for f in n_nan_train:
    med_val=df_train[f].median()
    df_train[f].fillna(med_val,inplace=True)
for f in n_nan_test:
    med_val=df_test[f].median()
    df_test[f].fillna(med_val,inplace=True)
combine=[df_train,df_test]


# In[9]:


for dataset in combine:
    dataset['Gender']=dataset['Gender'].fillna(dataset['Gender'].mode()[0])
    dataset['Self_Employed']=dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0])
    #dataset['Dependents']=dataset['Dependents'].fillna(dataset['Dependents'].mode()[0])
#for dataset in df_train:   
df_train['Married']=df_train['Married'].fillna(df_train['Married'].mode()[0])
#df_train['Loan_Status']=df_train['Loan_Status'].fillna(df_train['Loan_Status'].mode()[0])
for dataset in combine:
    dataset['Gender'] = dataset['Gender'].map( {'Female': 1, 'Male': 0} ).astype(int)

df_train['Loan_Status'] = df_train['Loan_Status'].map( {'Y': 1, 'N': 0} ).astype(int)


# In[10]:


for dataset in combine:
    dataset['totalincome'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
for dataset in combine:
    dataset['debtratio'] = dataset['totalincome']/dataset['LoanAmount']
for dataset in combine:
    dataset['debtratio'] = np.log(dataset['debtratio']) 
for dataset in combine:
    dataset['totalincome_log'] = np.log(dataset['totalincome']) 
for dataset in combine:
    dataset['EMI'] = dataset['LoanAmount']/dataset['Loan_Amount_Term']
for dataset in combine:
    dataset['balance'] = dataset['totalincome']-(dataset['EMI']*1000)
for dataset in combine:
    dataset['balance_log'] = np.log(dataset['balance']) 


# In[11]:


df_train['LoanAmount_log'] = np.log(df_train['LoanAmount'])
df_test['LoanAmount_log'] = np.log(df_test['LoanAmount'])

df_train = df_train.drop(['Loan_ID','ApplicantIncome','CoapplicantIncome','totalincome','LoanAmount','balance'],axis=1) 
df_test = df_test.drop(['Loan_ID','ApplicantIncome','CoapplicantIncome','totalincome','LoanAmount','balance'], axis=1)
combine = [df_train, df_test]
df_test.head()
#sns.countplot(x='Gender',hue='Loan_Status',data=df_train)
#c_nan=[f for f in dataset.columns if dataset[f].isnull().sum()>1 and dataset[f].dtypes=='O']


# In[12]:


for dataset in combine:
    dataset['Education'] = dataset['Education'].map( {'Graduate': 1, 'Not Graduate': 0} ).astype(int)
for dataset in combine:
    dataset['Self_Employed'] = dataset['Self_Employed'].map( {'Yes': 1, 'No': 0} ).astype(int)
for dataset in combine:
    dataset['Property_Area'] = dataset['Property_Area'].map( {'Semiurban': 0, 'Urban': 1, 'Rural':2} ).astype(int)
for dataset in combine:
    dataset['Married'] = dataset['Married'].map( {'Yes': 1, 'No': 0} ).astype(int)


# In[13]:


#n_nan_train=[f for f in df_train.columns if df_train[f].isnull().sum()>=0 and df_train[f].dtypes!='O']
'''n_nan_test=[f for f in df_test.columns if df_test[f].isnull().sum()>=0 and df_test[f].dtypes!='O']
for f in n_nan_test:
    med_val=df_test[f].median()
    df_test[f]=df_test[f].fillna(med_val,inplace=True)'''

df_train.head()


# In[14]:


df_test.isnull().sum()


# In[15]:


for dataset in combine:
    dataset['Dependents'] = dataset['Dependents'].map( {'0': 0, '1': 1,'2': 2, '3+': 3} ).astype(int)
df_train['depenBand'] = pd.cut(df_train['Dependents'],4)
df_train[['depenBand', 'Loan_Status']].groupby(['depenBand'], as_index=False).mean().sort_values(by='depenBand', ascending=True)


# In[16]:


df_train['loanBand'] = pd.cut(df_train['Loan_Amount_Term'],5)
df_train[['loanBand', 'Loan_Status']].groupby(['loanBand'], as_index=False).mean().sort_values(by='loanBand', ascending=True)


# In[17]:


for dataset in combine:    
    dataset.loc[ dataset['Dependents'] <= 0.75, 'Loan_Amount_Term'] = 0
    dataset.loc[(dataset['Dependents'] > 0.75) & (dataset['Dependents'] <= 1.5), 'Dependents'] = 1
    dataset.loc[(dataset['Dependents'] > 1.5) & (dataset['Dependents'] <= 2.25), 'Dependents'] = 2
    dataset.loc[(dataset['Dependents'] > 2.25) & (dataset['Dependents'] <= 3.0), 'Dependents'] = 3
for dataset in combine:    
    dataset.loc[ dataset['Loan_Amount_Term'] <= 105, 'Loan_Amount_Term'] = 0
    dataset.loc[(dataset['Loan_Amount_Term'] > 105) & (dataset['Loan_Amount_Term'] <= 199), 'Loan_Amount_Term'] = 1
    dataset.loc[(dataset['Loan_Amount_Term'] > 199) & (dataset['Loan_Amount_Term'] <= 292), 'Loan_Amount_Term'] = 2
    dataset.loc[(dataset['Loan_Amount_Term'] > 292) & (dataset['Loan_Amount_Term'] <= 386), 'Loan_Amount_Term'] = 3
    dataset.loc[(dataset['Loan_Amount_Term'] > 386) & (dataset['Loan_Amount_Term'] <= 480), 'Loan_Amount_Term'] = 4
df_train = df_train.drop(['loanBand'], axis=1)
df_train = df_train.drop(['depenBand'], axis=1)
#df_test = df_test.drop(['loanBand'], axis=1)
combine = [df_train, df_test]
df_test.head()


# In[18]:



df_train.isnull().sum()


# In[19]:


df_test.head()


# In[20]:


n_nan_train=[f for f in df_train.columns if df_train[f].isnull().sum()>1 and df_train[f].dtypes!='O']
n_nan_test=[f for f in df_test.columns if df_test[f].isnull().sum()>1 and df_test[f].dtypes!='O']
for f in n_nan_train:
    med_val=df_train[f].median()
    df_train[f].fillna(med_val,inplace=True)
for f in n_nan_test:
    med=df_test[f].median()
    df_test[f].fillna(med,inplace=True)
#df_test['balance_log'].isnull().sum()
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = df_train.drop('Loan_Status',axis=1)  #independent columns
y = df_train['Loan_Status']  
X=X.astype(int)
y=y.astype(int)
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
#featureScores
print(featureScores.nlargest(10,'Score'))


# In[21]:


for dataset in combine:
    dataset['totalincome_log'] = dataset['totalincome_log']/(dataset["totalincome_log"].median())
for dataset in combine:
    dataset['balance_log'] = dataset['balance_log']/(dataset["balance_log"].median())
for dataset in combine:
    dataset['LoanAmount_log'] = dataset['LoanAmount_log']/(dataset["LoanAmount_log"].median())
for dataset in combine:
    dataset['debtratio'] = dataset['debtratio']/(dataset["debtratio"].median())
df_train = df_train.drop(['Self_Employed','totalincome_log','balance_log'], axis=1)
df_test = df_test.drop(['Self_Employed','totalincome_log','balance_log'], axis=1)
combine = [df_train, df_test]

df_test.head()


# In[22]:


#from sklearn.model_selection import train_test_split
#X_train,X_test,y_train,y_test=train_test_split(df_train.drop('Loan_Status',axis=1),df_train['Loan_Status'],test_size=0.2,random_state=42)

X_train=df_train.drop('Loan_Status',axis=1)
y_train=df_train['Loan_Status']
X_test=df_test
X_train=np.array(X_train)
y_train=np.array(y_train)
X_test=np.array(X_test)
X_train = X_train.astype(float)
y_train = y_train.astype(float)
X_test = X_test.astype(float)


# In[23]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[32]:


'''import tpot#85.016
from tpot import TPOTClassifier

tpot = TPOTClassifier(#verbosity = 2,
                     #generations=8,
                    # population_size=150,
                     verbosity = 4,
                     generations=6,
                     population_size=170
                     #random_state=42
                     ) 
                      
                      #n_jobs=-1, 
                      #generations=9, 
                      #population_size=150,
                      #early_stop = 5,
                      #memory = None)
tpot.fit(X_train,y_train)
Y_pred = tpot.predict(X_test)
tpot.score(X_train, y_train)
acc_random_forest = round(tpot.score(X_train, y_train) * 100, 3)
acc_random_forest'''

#XGBClassifier(input_matrix, learning_rate=0.1, max_depth=3, min_child_weight=1, n_estimators=100, nthread=1, subsample=0.35000000000000003)


# In[58]:



import xgboost as xgb#1025.234932671499#1022.5823395094012
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
'''parameters = [{'learning_rate' : [0.05,0.04,0.1,0.09],
    'n_estimators' : [100,150,200,250],
    'max_depth' : [5,10,15,20],
    'min_child_weight' : [2,4,6,8,10]}]
grid_search = GridSearchCV(estimator = xgb,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, Y_train)                                                                          
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)'''
reg=xgb.XGBClassifier(learning_rate =0.11, n_estimators=100, max_depth=5, min_child_weight=2, gamma=0, subsample=0.35000000000000003,
                     #colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
#reg=xgb.XGBClassifier(learning_rate =0.19, n_estimators=100, max_depth=5, min_child_weight=2, gamma=0, subsample=0.38000000000000003,
                     #colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
reg.fit(X_train,y_train)
Y_pred = reg.predict(X_test)
reg.score(X_train, y_train)
acc_random_forest = round(reg.score(X_train, y_train) * 100, 3)
acc_random_forest


# In[59]:


pred['Loan_Status']=pd.DataFrame(Y_pred)
pred['Loan_Status'].replace(0, 'N',inplace=True) 
pred['Loan_Status'].replace(1, 'Y',inplace=True) 
sub_df=pd.read_csv('sample_submission_49d68Cx.csv')
datasets=pd.concat([sub_df['Loan_ID'],pred['Loan_Status']],axis=1)
datasets.columns=['Loan_ID','Loan_Status']
datasets.to_csv('sample_submission_49d68Cx.csv',index=False)
#datasets.head(20)


# In[18]:


'''from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
#random_forest = SVC(kernel = 'linear', random_state = 0)
#classifier.fit(X_train, y_train)
random_forest = LogisticRegression()
random_forest.fit(X_train,Y_train)
from sklearn.model_selection import GridSearchCV
parameters = [{'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]}]
grid_search = GridSearchCV(estimator = random_forest,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, Y_train)                                                                          
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 3)
#acc_random_forest#80.945#81.1184#80.9571'''


# In[11]:


'''random_forest = LogisticRegression(C=0.012742749857031334,
 max_iter=100,
 penalty='l1',
 solver='liblinear')
random_forest.fit(X_train,Y_train)
grid_search = grid_search.fit(X_train, Y_train) 
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 3)
#acc_random_forest
grid_search.best_score_'''


# In[ ]:





# In[ ]:





# In[12]:


#sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False)


# In[ ]:




