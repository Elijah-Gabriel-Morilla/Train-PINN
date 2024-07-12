import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class PINN(nn.Module):
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
        
        # Physics-informed output
        v_max = 100  # km/h, adjust based on Philippine train systems
        t_delay = signal + 0.5 * boarding  # Simplified delay calculation
        v_effective = v_max * (1 - 0.2 * track) * (1 - 0.1 * weather)  # Simplified speed adjustment
        
        t_travel = distance / v_effective
        arrival_time = time + t_travel + t_delay
        
        return self.fc4(h) + arrival_time

def physics_loss(model, x):
    time, track, weather, signal, boarding, distance = torch.split(x, 1, dim=1)
    predictions = model(x)
    
    # Physics-based constraints
    v_max = 100  # km/h
    t_delay = signal + 0.5 * boarding
    v_effective = v_max * (1 - 0.2 * track) * (1 - 0.1 * weather)
    t_travel = distance / v_effective
    expected_arrival = time + t_travel + t_delay
    
    return torch.mean(torch.abs(predictions - expected_arrival))

def train_model(X_train, y_train, input_size, num_epochs=100, batch_size=32, learning_rate=0.001):
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    
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
            p_loss = physics_loss(model, batch_x)
            total_loss = loss + 0.1 * p_loss
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')
    
    return model

if __name__ == "__main__":
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    
    input_size = X_train.shape[1]
    model = train_model(X_train, y_train, input_size)
    
    torch.save(model.state_dict(), 'pinn_model.pth')
    print('Model training completed and saved.')