import numpy as np
import torch
import torch.nn as nn  # Add this import
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class PINN(nn.Module):  # Change to nn.Module
    def __init__(self, input_size):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
    def forward(self, x):
        time, track, weather, signal, boarding, distance = torch.split(x, 1, dim=1)
        
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        
        v_max = 100  # km/h, adjust based on Philippine train systems
        t_delay = signal + 0.5 * boarding
        v_effective = v_max * (1 - 0.2 * track) * (1 - 0.1 * weather)
        
        t_travel = distance / v_effective
        arrival_time = time + t_travel + t_delay
        
        return self.fc4(h) + arrival_time

def load_model(model_path, input_size):
    model = PINN(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_model(model, X, y, scaler):
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        predictions = model(X_tensor).numpy()
    
    if scaler is not None:
        dummy = np.zeros_like(X)
        dummy[:, -1:] = predictions
        dummy = scaler.inverse_transform(dummy)
        predictions = dummy[:, -1]
    
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    return predictions.flatten()

def plot_results(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual Arrival Time")
    plt.ylabel("Predicted Arrival Time")
    plt.title("Actual vs Predicted Arrival Times")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    scaler = joblib.load('scaler.pkl')
    
    input_size = X_test.shape[1]
    model = load_model('pinn_model.pth', input_size)
    
    predictions = evaluate_model(model, X_test, y_test, scaler)
    
    plot_results(y_test, predictions)