import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

data_path = 'D:/MCsoft Course AI&ML/Heart_Disease_Project/data/HeartDiseaseTrain-Test.csv'
df = pd.read_csv(data_path)
print(df)

df.head()
print(df.head())

df.head(10)
print(df.head(10))

df.tail(14)
print(df.tail(14))

df.info()
print(df.info())

missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])
