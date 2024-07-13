import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class PINN(nn.Module):
    def __init__(self, input_size):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)
        
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        return self.fc4(h)

def load_model(model_path, input_size):
    model = PINN(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def evaluate_model(model, X, y):
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        predictions = model(X_tensor).numpy()
    
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    return predictions

def plot_results(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    X_test = np.load('X_train_sensor.npy')  # Use same data for simplicity
    y_test = np.load('y_train_sensor.npy')
    
    model = load_model('pinn_model.pth', X_test.shape[1])
    
    predictions = evaluate_model(model, X_test, y_test)
    
    plot_results(y_test[:, 0], predictions[:, 0])  # Example plot for the first output
