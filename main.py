import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns


def read_data_from_csv(filename):
    base_df = pd.read_csv(filename)
    base_df = base_df.drop('search_id', axis=1)
    return base_df.drop('target', axis=1), base_df['target']


def data_preprocessing(X, is_train=True):
    if is_train:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
    else:
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
            X_scaled = scaler.transform(X)
    return X_scaled


def create_tensors(X, y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor


class ClickPredictionNN(nn.Module):
    def __init__(self, input_size):
        super(ClickPredictionNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


def train_model(X_train, y_train, x_train_shape):
    input_size = x_train_shape.shape[1]
    model = ClickPredictionNN(input_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    batch_size = 128
    for epoch in range(num_epochs):
        for i in range(0, len(X_train), batch_size):
            inputs = X_train[i:i + batch_size]
            targets = y_train[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    torch.save(model.state_dict(), 'model_weights.pth')


def make_prediction(X_test, y_test, metric, size):
    model = ClickPredictionNN(size)
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        print(metric([y_test.numpy()], [np.where(preds[:, 0] > 0.2, 1, 0)]))
    return preds
def plot_distribution(predictions):
    plt.hist(predictions[:, 0], color='skyblue', edgecolor='black')

    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Basic Histogram')

    # Display the plot
    plt.show()


if __name__ == '__main__':
    X_train, y_train = read_data_from_csv('train_df.csv')
    X_test, y_test = read_data_from_csv('test_df.csv')
    X_train_pr = data_preprocessing(X_train)
    X_test_pr = data_preprocessing(X_test, False)
    X_train_tensor, y_train_tensor = create_tensors(X_train_pr, y_train)
    X_test_tensor, y_test_tensor = create_tensors(X_train_pr, y_train)
    train_model(X_train_tensor, y_train_tensor, X_train)
    predictions = make_prediction(X_test_tensor, y_test_tensor, ndcg_score, X_test.shape[1])
    plot_distribution(predictions)
