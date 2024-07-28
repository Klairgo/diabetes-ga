import sys
sys.path.append('../../Library/')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


dataset = pd.read_csv("./Dataset_of_Diabetes.csv")

names = dataset.columns.values

dataset.rename(columns={'CLASS': 'Outcome'}, inplace=True)


def map_outcome(x):
    if x == 'N':
        return 0
    elif x == 'P':
        return 1
    elif x == 'Y':
        return 1

dataset['Gender'] = dataset['Gender'].apply(lambda x: 0 if x == 'M' else 1)

# dataset_p = dataset[dataset['Outcome'] == 'P']

# dataset = dataset.drop(dataset_p.index)

X = dataset.iloc[:, :-1]  # Features
y = dataset.iloc[:, -1]   # Target variable (Outcome)

# Initialize the MinMaxScaler with the desired range (-1, 1)
scaler = MinMaxScaler(feature_range=(-1, 1))

# Fit the scaler to the features and transform them
X_scaled = scaler.fit_transform(X)

# Create a new DataFrame with the scaled features
X_scaled_df = pd.DataFrame(X_scaled, columns=names[:-1])

# Concatenate the scaled features with the target variable
dataset = pd.concat([X_scaled_df, y], axis=1)




dataset['Outcome'] = dataset['Outcome'].apply(map_outcome)

print(dataset['Outcome'].value_counts())

dataset_1 = dataset[dataset['Outcome'] == 1]
dataset_0 = dataset[dataset['Outcome'] == 0]

train_data = pd.concat([dataset_1.sample(frac=0.1, random_state=26), dataset_0.sample(frac=0.7, random_state=26)])

dataset = dataset.drop(train_data.index)

dataset_1 = dataset[dataset['Outcome'] == 1]
dataset_0 = dataset[dataset['Outcome'] == 0]

test_data = pd.concat([dataset_1.sample(frac=0.05, random_state=26), dataset_0])
#test_data = pd.concat([dataset_1, dataset_0])

# train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=26, stratify=dataset['Outcome'])
# train_data, test_data = train_test_split(dataset, test_size=0.3, random_state=26)

print(train_data['Outcome'].value_counts())
print(test_data['Outcome'].value_counts())

train_data.to_csv('./train.csv', index=False)
test_data.to_csv('./test.csv', index=False)