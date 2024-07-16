import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

def generate_sensor_data(num_samples):
    AccX = np.random.uniform(-2.0, 2.0, num_samples)
    AccY = np.random.uniform(-2.0, 2.0, num_samples)
    AccZ = np.random.uniform(9.71, 10.21, num_samples)
    GyroX = np.random.uniform(-200.0, 200.0, num_samples)
    GyroY = np.random.uniform(-200.0, 200.0, num_samples)
    GyroZ = np.random.uniform(-200.0, 200.0, num_samples)
    return AccX, AccY, AccZ, GyroX, GyroY, GyroZ

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

def prepare_data(num_samples=1000):
    AccX, AccY, AccZ, GyroX, GyroY, GyroZ = generate_sensor_data(num_samples)
    X_train = np.column_stack((AccX, AccY, AccZ, GyroX, GyroY, GyroZ))
    y_train = np.random.randn(num_samples, 1)
    np.save('X_train_sensor.npy', X_train)
    np.save('y_train_sensor.npy', y_train)

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
    prepare_data()
    X_train = np.load('X_train_sensor.npy')
    y_train = np.load('y_train_sensor.npy')
    input_size = X_train.shape[1]
    train_model(X_train, y_train, input_size)

    X_test = np.random.randn(100, input_size)
    y_test = np.random.randn(100, 1)
    model = PINN(input_size)
    model.load_state_dict(torch.load('pinn_model.pth'))
    predictions, mse, r_squared = evaluate_model(model, X_test, y_test)

    plot_results(y_test, predictions)  # Use y_test instead of zeros
    anomaly_probability = compute_anomaly_probability(predictions)
    print(f"Anomaly Probability: {anomaly_probability:.2f}%")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r_squared:.4f}")

if __name__ == "__main__":
    main()
