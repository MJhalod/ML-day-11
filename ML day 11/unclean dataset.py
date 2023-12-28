import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'smartphones.csv'
df = pd.read_csv(file_path)

print("Dataset Info:")
print(df.info())

print("\nFirst few rows of the dataset:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna()
df = df.fillna(df.mean())

corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


sns.pairplot(df)
plt.show()

sns.boxplot(x='categorical_column', y='numerical_column', data=df)
plt.show()
