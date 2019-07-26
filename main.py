import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression


#df=pd.read_csv("bank.csv",usecols=range(0,1))
#选取特定范围的列

df=pd.read_csv("bank.csv")
df=df['age;"job";"marital";"education";"default";"balance";"housing";"loan";"contact";"day";"month";"duration";"campaign";"pdays";"previous";"poutcome";"y"'].str.split(';',expand=True)
df.rename(columns={0: 'age', 1: 'job',2: 'marital',3: 'education',4: 'default',5: 'balance',6: 'housing',7: 'loan',8: 'contact',9: 'day',10: 'month',11: 'duration',12: 'campaign',13: 'pdays',14: 'previous',15: 'poutcome',16:"y"}, inplace=True)
print(df)