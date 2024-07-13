import numpy as np

def generate_sensor_data(num_samples=1000):
    AccX = np.random.uniform(-10, 10, num_samples)
    AccY = np.random.uniform(-10, 10, num_samples)
    AccZ = np.random.uniform(9.71, 9.91, num_samples)
    
    GyroX = np.random.uniform(-180, 180, num_samples)
    GyroY = np.random.uniform(-180, 180, num_samples)
    GyroZ = np.random.uniform(-180, 180, num_samples)
    
    AngX = np.arctan(AccY / np.sqrt(AccX**2 + AccZ**2)) * 180 / np.pi
    AngY = np.arctan(-AccX / np.sqrt(AccY**2 + AccZ**2)) * 180 / np.pi
    Yaw_new = np.zeros(num_samples)
    for i in range(1, num_samples):
        Yaw_new[i] = Yaw_new[i-1] + (GyroZ[i] * 1)

    sensor_data = np.stack((AccX, AccY, AccZ, GyroX, GyroY, GyroZ, AngX, AngY, Yaw_new), axis=-1)
    return sensor_data

def prepare_data():
    sensor_data = generate_sensor_data()
    
    np.save('sensor_data.npy', sensor_data)
    
    X = sensor_data[:, :6]
    y = sensor_data[:, 6:]
    
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    
    X_scaled = X
    
    np.save('X_train_sensor.npy', X_scaled)
    np.save('y_train_sensor.npy', y)

if __name__ == "__main__":
    prepare_data()
