import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def read_csv_file(file_path):
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
        print("CSV file read successfully.")
        return data
    except UnicodeDecodeError:
        print("Error reading the CSV file.")
        return None

def prepare_data(file_path):
    data = read_csv_file(file_path)
    
    AccX_real = data['AccX'].values[:247]
    AccY_real = data['AccY'].values[:247]
    AccZ_real = data['AccZ'].values[:247]
    GyroX_real = data['GyroX'].values[:247]
    GyroY_real = data['GyroY'].values[:247]
    GyroZ_real = data['GyroZ'].values[:247]
    
    X_train = np.vstack((AccX_real, AccY_real, AccZ_real, GyroX_real, GyroY_real, GyroZ_real)).T
    y_train = np.random.randn(247, 1)  # Placeholder for demonstration
    
    return X_train, y_train

class PINN(nn.Module):
    def __init__(self, input_size):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        return torch.sigmoid(self.fc4(h))

def train_model(X_train, y_train, input_size, num_epochs=100, batch_size=32, learning_rate=0.001):
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = PINN(input_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), 'pinn_model.pth')

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    mse = mean_squared_error(y_test, predictions)
    r_squared = r2_score(y_test, predictions)
    return predictions, mse, r_squared

def plot_results(actual, predicted):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title('Actual vs Predicted')
    plt.xlabel('Samples')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.show()

def compute_anomaly_probability(predictions):
    anomaly_threshold = 0.5
    anomaly_count = np.sum(predictions > anomaly_threshold)
    total_samples = len(predictions)
    anomaly_probability = (anomaly_count / total_samples) * 100
    return anomaly_probability

def main():
    file_path = r'C:\Users\elija\Desktop\TrainPINNV3\train_Data\train-real-data.csv'
    
    X_train, y_train = prepare_data(file_path)
    input_size = X_train.shape[1]
    
    train_model(X_train, y_train, input_size)
    
    num_samples_test = 100
    X_test = np.random.randn(num_samples_test, input_size)
    y_test = np.random.randn(num_samples_test, 1)
    
    model = PINN(input_size)
    model.load_state_dict(torch.load('pinn_model.pth'))
    predictions, mse, r_squared = evaluate_model(model, X_test, y_test)
    
    plot_results(y_test, predictions)
    anomaly_probability = compute_anomaly_probability(predictions)
    print(f"Anomaly Probability: {anomaly_probability:.2f}%")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r_squared:.4f}")

if __name__ == "__main__":
    main()
