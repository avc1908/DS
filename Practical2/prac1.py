import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# pd.set_option('display.width', 1000)
data = pd.read_csv('Training.csv')
print(data)


# Basic Data Preprocessing
print(data.shape)
print(data.columns)

## Duplicate values in data
print(data.duplicated().sum())

## Removing duplicate values
data.drop_duplicates(inplace=True)
print(data.shape)

def describe(data):
    format = "%-30s %-10s %6s %6s %s"
    print(format % ("Column Name", "Data Type", "Null", "Unique", "Unique"))
    print(format % ("", "", "Count", "Count", "Values"))
    print('-' * 65)
    for col in data.columns:
        print(format % (col, data[col].dtype, data[col].isnull().sum(),
                        data[col].nunique(), data[col].unique()))
print(describe(data))

num = [x for x in data.columns if data[x].dtype != 'int64']
print(num)

print(data['prognosis'].value_counts())
print(data['Unnamed: 133'].value_counts())

### Dropping unwanted columns
data.drop('Unnamed: 133', axis=1, inplace=True)
print(data)


# EDA
sns.set(style='darkgrid')
sns.countplot(data, x='skin_rash')
def count_plot(data):
    for col in data.columns:
        # Depending on where you run this, one of the lines
        # below will work for creating a countplot
        sns.countplot(data[col])
        plt.show()
print(data)