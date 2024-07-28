import sys
sys.path.append('../../Library/')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
dataset = pd.read_csv("./diabetes.csv")

data = dataset.copy()

data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)

def get_median(attribute):
    # temp = dataset[dataset[attribute].notnull()]
    temp = data[data[attribute].notnull()]
    mean = temp[[attribute, 'Outcome']].groupby('Outcome')[[attribute]].median().reset_index()
    return mean

mean = get_median('Glucose')
data.loc[(data['Outcome'] == 0) & (data['Glucose'].isnull()), 'Glucose'] = mean['Glucose'][0]
data.loc[(data['Outcome'] == 1) & (data['Glucose'].isnull()), 'Glucose'] = mean['Glucose'][1]

mean = get_median('BloodPressure')
data.loc[(data['Outcome'] == 0) & (data['BloodPressure'].isnull()), 'BloodPressure'] = mean['BloodPressure'][0]
data.loc[(data['Outcome'] == 1) & (data['BloodPressure'].isnull()), 'BloodPressure'] = mean['BloodPressure'][1]

mean = get_median('SkinThickness')
data.loc[(data['Outcome'] == 0) & (data['SkinThickness'].isnull()), 'SkinThickness'] = mean['SkinThickness'][0]
data.loc[(data['Outcome'] == 1) & (data['SkinThickness'].isnull()), 'SkinThickness'] = mean['SkinThickness'][1]

mean = get_median('Insulin')
data.loc[(data['Outcome'] == 0) & (data['Insulin'].isnull()), 'Insulin'] = mean['Insulin'][0]
data.loc[(data['Outcome'] == 1) & (data['Insulin'].isnull()), 'Insulin'] = mean['Insulin'][1]

mean = get_median('BMI')
data.loc[(data['Outcome'] == 0) & (data['BMI'].isnull()), 'BMI'] = mean['BMI'][0]
data.loc[(data['Outcome'] == 1) & (data['BMI'].isnull()), 'BMI'] = mean['BMI'][1]

# columns_to_standardize = dataset.columns.difference(['Outcome'])
# data[columns_to_standardize] = dataset[columns_to_standardize] / dataset[columns_to_standardize].max()

# standardize the data
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target variable (Outcome)

# Initialize the MinMaxScaler with the desired range (-1, 1)
scaler = MinMaxScaler(feature_range=(-1, 1))

# Fit the scaler to the features and transform them
X_scaled = scaler.fit_transform(X)

# Create a new DataFrame with the scaled features
X_scaled_df = pd.DataFrame(X_scaled, columns=names[:-1])

# Concatenate the scaled features with the target variable
data = pd.concat([X_scaled_df, y], axis=1)

dataset_1 = data[data['Outcome'] == 1]
dataset_0 = data[data['Outcome'] == 0]

train_data = pd.concat([dataset_1.sample(frac=0.5, random_state=26), dataset_0.sample(frac=0.3, random_state=26)])

data = data.drop(train_data.index)

dataset_1 = data[data['Outcome'] == 1]
dataset_0 = data[data['Outcome'] == 0]

test_data = pd.concat([dataset_1, dataset_0.sample(frac=0.5, random_state=26)])

# train_data, test_data = train_test_split(data, test_size=0.4, random_state=26, stratify=data['Outcome'])

print(train_data['Outcome'].value_counts())
print(test_data['Outcome'].value_counts())

train_data.to_csv('./train.csv', index=False)
test_data.to_csv('./test.csv', index=False)