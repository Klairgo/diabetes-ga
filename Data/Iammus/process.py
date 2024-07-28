import sys
sys.path.append('../../Library/')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


dataset = pd.read_csv("./diabetes_prediction_dataset.csv")

dataset.rename(columns={'diabetes': 'Outcome'}, inplace=True)

names = dataset.columns.values

def map_smoking_history(x):
    if x == 'never':
        return 0
    elif x == 'ever':
        return 1
    elif x == 'current':
        return 2
    elif x == 'No Info':
        return 3
    elif x == 'former':
        return 4
    elif x == 'not current':
        return 5

dataset['gender'] = dataset['gender'].apply(lambda x: 0 if x == 'Male' else 1)
dataset['smoking_history'] = dataset['smoking_history'].apply(map_smoking_history)


# columns_to_standardize = dataset.columns.difference(['Outcome'])
# dataset[columns_to_standardize] = dataset[columns_to_standardize] / dataset[columns_to_standardize].max()

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



dataset_1 = dataset[dataset['Outcome'] == 1]
dataset_0 = dataset[dataset['Outcome'] == 0]

# train_data = pd.concat([dataset_1.sample(frac=0.01, random_state=26), dataset_0.sample(frac=0.001, random_state=26)])
train_data = pd.concat([dataset_1.sample(frac=0.01, random_state=26), dataset_0.sample(frac=0.001, random_state=26)])

dataset = dataset.drop(train_data.index)

dataset_1 = dataset[dataset['Outcome'] == 1]
dataset_0 = dataset[dataset['Outcome'] == 0]

# test_data = pd.concat([dataset_1.sample(frac=0.01, random_state=26), dataset_0.sample(frac=0.001, random_state=26)])
test_data = pd.concat([dataset_1.sample(frac=0.01, random_state=26), dataset_0.sample(frac=0.001, random_state=26)])

# train_data, test_data = train_test_split(dataset, test_size=0.1, train_size=0.02, random_state=26, stratify=dataset['Outcome'])
# train_data, test_data = train_test_split(dataset, test_size=0.1, train_size=0.02, random_state=26)

print(train_data['Outcome'].value_counts())
print(test_data['Outcome'].value_counts())

train_data.to_csv('./train.csv', index=False)
test_data.to_csv('./test.csv', index=False)