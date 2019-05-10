import pandas as pd

df = pd.read_csv('test.csv')
print(df.head())

print('==================')
print(df.loc[[0,0,1]])
print('==================')
print(df.loc[1])
