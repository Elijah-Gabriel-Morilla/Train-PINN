import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

class PINN(nn.Module):
    def __init__(self, input_size):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)  # Adjusted output to 1 for binary classification
        
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        h = torch.relu(self.fc3(h))
        return torch.sigmoid(self.fc4(h))  # Sigmoid for binary classification

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
    
    # Compute accuracy and accident-prone percentage
    threshold = 0.5  # Adjust threshold as needed
    y_pred_binary = (predictions > threshold).astype(int)
    accuracy = accuracy_score(y, y_pred_binary)
    accident_prone_percentage = 100 - accuracy * 100
    
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Accident-Prone Percentage: {accident_prone_percentage:.2%}")
    
    return predictions, y_pred_binary

if __name__ == "__main__":
    # Load and prepare your test data (X_test, y_test)
    # Example:
    X_test = np.load('X_test_sensor.npy')  # Adjust path as necessary
    y_test = np.load('y_test_sensor.npy')
    
    model = load_model('pinn_model.pth', X_test.shape[1])
    
    predictions, y_pred_binary = evaluate_model(model, X_test, y_test)
