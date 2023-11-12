
# dataframe and plotting
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# machine learning
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import os
import warnings
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import pickle

warnings.filterwarnings('ignore')

"""### 1. Load the dataset"""

# Load files into a pandas dataframe
DATA_PATH = 'https://raw.githubusercontent.com/ayamlearning/Financial_Inclusion_Africa/main/data'
df = pd.read_csv(os.path.join(DATA_PATH, 'Train.csv'))
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical = list(df.select_dtypes(include=['object']).columns)
categorical.remove("uniqueid")
categorical.remove("bank_account")
numerical = list(df.select_dtypes(include=['int64']).columns)
target = ['bank_account']


df['bank_account']=df['bank_account'].replace({'Yes': 1,
                                               'No': 0})

df_full_train, df_test = train_test_split(df[categorical + numerical + target], test_size=0.2, random_state=1)

def train(df_train, y_train):
    min_child_weighth =1
    gamma = 0.5
    subsample =0.8

    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = XGBClassifier(min_child_weighth=min_child_weighth,gamma=gamma,
                                          subsample=subsample)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

n_splits = 5

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.bank_account.values
    y_val = df_val.bank_account.values

    dv, model = train(df_train, y_train)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('validation results:')
print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))

dv, model = train(df_full_train, df_full_train.bank_account.values)
y_pred = predict(df_test, dv, model)

y_test = df_test.bank_account.values
auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')

output_file = 'model_financial_inclusion.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')
