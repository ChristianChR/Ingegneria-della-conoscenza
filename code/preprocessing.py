import pandas as pd

df = pd.read_csv('data/mushrooms.csv')

#one hot encoding
df_encoded = pd.get_dummies(df)


df_encoded.to_csv('data/mushrooms_encoded.csv', index=False)