import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("bank-additional.csv")
df=df['age;"job";"marital";"education";"default";"housing";"loan";"contact";"month";"day_of_week";"duration";"campaign";"pdays";"previous";"poutcome";"emp.var.rate";"cons.price.idx";"cons.conf.idx";"euribor3m";"nr.employed";"y"'].str.split(';',expand=True)
df.rename(columns={0: 'age', 1: 'job',2: 'marital',3: 'education',4: 'default',5: 'housing',6: 'loan',7: 'contact',8: 'month',9: 'day_of_week'}, inplace=True)
df=df[[ 'age',  'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week']];
print(df.head())
df.to_csv('Bank10Row.csv')


vec_count = CountVectorizer()#文字列をベクトル化する　https://qiita.com/fujin/items/b1a7152c2ec2b4963160
vec_count.fit(df['job'])
X = vec_count.transform(df['job'])

print(X)
print('Vocabulary content: {}'.format(vec_count.vocabulary_))
