import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import warnings
warnings.filterwarnings('ignore')

from pandas import DataFrame, Series

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score

from lightgbm import LGBMClassifier as LGBMC

from optuna import Trial, trial, create_study



#データセット
df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')



fare_median=df[(df['Embarked'] == "S") & (df['Pclass'] == 3)].Fare.median()
df['Fare']=df['Fare'].fillna(fare_median)
df['Embarked'].fillna('C', inplace=True)


df.isnull().sum()
df_test.isnull().sum()


#訓練データ前処理
df.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)
df_test.replace({'Sex': {'male': 0, 'female': 1}}, inplace=True)

embarked = pd.concat([df['Embarked'], df_test['Embarked']])

embarked_ohe = pd.get_dummies(embarked, dtype=int)

embarked_ohe_train = embarked_ohe[:891]
embarked_ohe_test = embarked_ohe[891:]

df = pd.concat([df, embarked_ohe_train], axis=1)
df_test = pd.concat([df_test, embarked_ohe_test], axis=1)

df.drop('Embarked', axis=1, inplace=True)
df_test.drop('Embarked', axis=1, inplace=True)



df.head(10)



Ticket_Count = dict(df['Ticket'].value_counts())
df['TicketGroup'] = df['Ticket'].apply(lambda x:Ticket_Count[x])

def Ticket_Label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0

def Family_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0
    

#チケットによる区分
df['TicketGroup'] = df['TicketGroup'].apply(Ticket_Label)



#敬称による区分
df['Honorifics'] = df['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
Honorifics_Dict = {}
Honorifics_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Honorifics_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Honorifics_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Honorifics_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Honorifics_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Honorifics_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
df['Honorifics'] = df['Honorifics'].map(Honorifics_Dict)


#家族人数による区分
df['FamilySize']=df['SibSp']+df['Parch']+1
df['FamilyLabel']=df['FamilySize'].apply(Family_label)


#苗字による区分
df['Surname'] = df['Name'].apply(lambda x:x.split(',')[0].strip())
Surname_Count = dict(df['Surname'].value_counts())
df['Surname_Count'] = df['Surname'].apply(lambda x:Surname_Count[x])
Female_Child_Group=df.loc[(df['Surname_Count']>=2) & ((df['Age']<=12) | (df['Sex']== 1))]
Male_Adult_Group=df.loc[(df['Surname_Count']>=2) & (df['Age']>12) & (df['Sex']=='0')]
Female_Child_mean = Female_Child_Group.groupby('Surname')['Perished'].mean()
Female_Child_mean_count = pd.DataFrame(Female_Child_mean.value_counts())
Male_Adult_mean = Male_Adult_Group.groupby('Surname')['Perished'].mean()
Male_Adult_mean_count = pd.DataFrame(Male_Adult_mean.value_counts())
Dead_List = set(Female_Child_mean[Female_Child_mean.apply(lambda x:x==1)].index)
Survived_List = set(Male_Adult_mean[Male_Adult_mean.apply(lambda x:x==0)].index)


df.loc[(df['Surname'].apply(lambda x:x in Dead_List)),'Sex'] = 0
df.loc[(df['Surname'].apply(lambda x:x in Dead_List)),'Age'] = 60
df.loc[(df['Surname'].apply(lambda x:x in Dead_List)),'Honorifics'] = 'Mr'
df.loc[(df['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 1
df.loc[(df['Surname'].apply(lambda x:x in Survived_List)),'Age'] = 5
df.loc[(df['Surname'].apply(lambda x:x in Survived_List)),'Honorifics'] = 'Miss'



df['CabinDeck'] = df['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')
df['CabinDeck'] = df['CabinDeck'].replace(['T'], 'U')


#敬称をラベルエンコーディング
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Royalty": 4, "Officer":5}
df['Title'] = df['Honorifics'].map(title_mapping)
df['Title'] = df['Title'].fillna(0)



#年齢の欠損値補完
df_age_imputation_sample = df.copy()
age_features_sample = ['Pclass', 'FamilyLabel', 'Title']
known_age_sample = df_age_imputation_sample[df_age_imputation_sample['Age'].notna()]
unknown_age_sample = df_age_imputation_sample[df_age_imputation_sample['Age'].isna()]
X_age_train_sample = known_age_sample[age_features_sample]
y_age_train_sample = known_age_sample['Age']
X_age_predict_sample = unknown_age_sample[age_features_sample]

age_model_sample = RandomForestRegressor(n_estimators=100, random_state=42)
age_model_sample.fit(X_age_train_sample, y_age_train_sample)


predicted_ages_sample = age_model_sample.predict(X_age_predict_sample)

df.loc[df['Age'].isna(), 'Age'] = predicted_ages_sample

dataset6 = df[['Perished','Pclass','Sex','Age','Fare','Title','FamilyLabel','CabinDeck','C','Q','S','TicketGroup']]
# ダミー変数を作成
dataset_dummies = pd.get_dummies(dataset6, dtype=int)


#テストデータ前処理
Ticket_Count = dict(df_test['Ticket'].value_counts())
df_test['TicketGroup'] = df_test['Ticket'].apply(lambda x:Ticket_Count[x])

df_test['TicketGroup'] = df_test['TicketGroup'].apply(Ticket_Label)

df_test['Honorifics'] = df_test['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
Honorifics_Dict = {}
Honorifics_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Honorifics_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Honorifics_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Honorifics_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Honorifics_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Honorifics_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
df_test['Honorifics'] = df_test['Honorifics'].map(Honorifics_Dict)

df_test['FamilySize']=df_test['SibSp']+df_test['Parch']+1

df_test['FamilyLabel']=df_test['FamilySize'].apply(Family_label)

df_test['Surname'] = df_test['Name'].apply(lambda x:x.split(',')[0].strip())
df_test.loc[(df_test['Surname'].apply(lambda x:x in Dead_List)),'Sex'] = 0
df_test.loc[(df_test['Surname'].apply(lambda x:x in Dead_List)),'Age'] = 60
df_test.loc[(df_test['Surname'].apply(lambda x:x in Dead_List)),'Honorifics'] = 'Mr'
df_test.loc[(df_test['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 1
df_test.loc[(df_test['Surname'].apply(lambda x:x in Survived_List)),'Age'] = 5
df_test.loc[(df_test['Surname'].apply(lambda x:x in Survived_List)),'Honorifics'] = 'Miss'

df_test['CabinDeck'] = df_test['Cabin'].apply(lambda x: str(x)[0] if pd.notnull(x) else 'U')
df_test['CabinDeck'] = df_test['CabinDeck'].replace(['T'], 'U')
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Royalty": 4, "Officer":5}
df_test['Title'] = df_test['Honorifics'].map(title_mapping)
df_test['Title'] = df_test['Title'].fillna(0)

df_age_imputation_sample_test = df_test.copy()
age_features_sample_test = ['Pclass', 'FamilyLabel', 'Title']
known_age_sample_test = df_age_imputation_sample_test[df_age_imputation_sample_test['Age'].notna()]
unknown_age_sample_test = df_age_imputation_sample_test[df_age_imputation_sample_test['Age'].isna()]
X_age_train_sample_test = known_age_sample_test[age_features_sample_test]
y_age_train_sample_test = known_age_sample_test['Age']
X_age_predict_sample_test = unknown_age_sample_test[age_features_sample_test]


age_model_sample = RandomForestRegressor(n_estimators=100, random_state=42)
age_model_sample.fit(X_age_train_sample_test, y_age_train_sample_test)


predicted_ages_sample_test = age_model_sample.predict(X_age_predict_sample_test)


df_test.loc[df_test['Age'].isna(), 'Age'] = predicted_ages_sample_test

dataset6_test = df_test[['Pclass','Sex','Age','Fare','Title','FamilyLabel','CabinDeck','C','Q','S','TicketGroup']]

dataset_dummies_test = pd.get_dummies(dataset6_test, dtype=int)




#パラメータチューニング
X = dataset_dummies.iloc[:, 1:].values
y = dataset_dummies.iloc[:, 0].values

X_tes = dataset_dummies_test.iloc[:, 0:].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

def objective(trial):

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

    param = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_error'},
        'max_depth'  : trial.suggest_int('max_depth', 2,  36),
        'min_depth'  : trial.suggest_int('min_depth', 2,  36),


        'lambda_l1'         : trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2'         : trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.08, 1.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),

        'force_col_wise':True,
        'random_state': 42
        }


    model = lgb.train(
                    params=param,
                    train_set=lgb_train,
                    valid_sets=[lgb_train, lgb_test],
                    valid_names=['Train', 'Test'],
                    num_boost_round=100,
                    callbacks=[lgb.early_stopping(stopping_rounds=50)]
                    )


    pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = np.where(pred < 0.53, 0, 1)

    score = fbeta_score(y_test, y_pred, average='binary', beta=0.5)

    return score

study = optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100)

print(study.best_params)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

param = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_error'},
    'force_col_wise':True,
    'random_state': 42,
    }
param.update(study.best_params)


model = lgb.train(params=param,
                train_set=lgb_train,
                valid_sets=[lgb_train, lgb_test],
                valid_names=['Train', 'Test'],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(stopping_rounds=50)]
                  )



pred = model.predict(X_test, num_iteration=model.best_iteration)
y_pred = np.where(pred < 0.53, 0, 1)

X_pred= model.predict(np.array(X_tes), num_iteration=model.best_iteration)
for i in range(418):
    if X_pred[i]>=0.53:
        X_pred[i]= 1
    else:
        X_pred[i]= 0





# gender_submissionを書き換えて提出データ作成

submission = pd.read_csv('gender_submission.csv')
submission['Perished'] = X_pred.astype(int)
submission

# 提出データ
submission.to_csv('submissionLast.csv',index=False)

