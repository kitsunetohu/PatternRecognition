from IPython.display import display
import pandas as pd
from pandas import DataFrame, Series
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
from sklearn.model_selection import cross_val_score

df=pd.read_csv("train.csv")
df=df[[ 'age',  'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week']];
print(df.head())
df.info()#データセットの情報
df.to_csv('Bank10Row.csv')#不要のデータを削除


for key in  df:
    if key !='age' and key!='marital':
        df_tmp = df[key]#Age以外の文字データを数字データーに変える
        df_tmp_encoded, df_tmp_categories = df_tmp.factorize()
        df[key] = df_tmp_encoded;

df_marital=df['marital'];
df_marital_encoded, df_marital_categories = df_marital.factorize()
df['marital']=df_marital_encoded;
print(df_marital_categories[:10])


print(df.head())

x = df[[ 'age',  'job', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week']]#date
y =df['marital']#tag
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.25, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

some_testX=x_test[:5]
some_testY=y_test[:5]



scores = accuracy_score(log_reg.predict(some_testX),list(some_testY))
print(scores)
