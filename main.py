import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def train_test_split(data, test_size=0.2, random_state=None, shuffle=True):
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    if random_state is not None:
        np.random.seed(random_state)

    num_samples = data.shape[0]
    indices = np.arange(num_samples)

    if shuffle:
        np.random.shuffle(indices)

    split_index = int(num_samples * (1 - test_size))
    train_indices, test_indices = indices[:split_index], indices[split_index:]

    train_data, test_data = data[train_indices], data[test_indices]

    return train_data, test_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
train_data = pd.read_csv('/Train.csv')
print("Shape of train_data:", train_data.shape)


X = train_data.iloc[:, 1:]  
y = train_data.iloc[:, 0]   

print("Shape of X after separating features:", X.shape)
if not isinstance(X, pd.DataFrame):
    X = pd.DataFrame(X)
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)  
X = X.values / 255.0
X = X.reshape(-1, 28, 28, 1)
print("Shape of X after reshaping:", X.shape)
y = to_categorical(y, num_classes=10)
print("Shape of y after one-hot encoding:", y.shape)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)