import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    
    time = np.random.uniform(0, 24, num_samples)  # Time in hours
    track_condition = np.random.choice(['good', 'fair', 'poor'], num_samples)
    weather_condition = np.random.choice(['clear', 'rainy', 'stormy'], num_samples)
    signal_delay = np.random.uniform(0, 10, num_samples)  # Delay in minutes
    boarding_time = np.random.uniform(1, 5, num_samples)  # Boarding time in minutes
    distance = np.random.uniform(10, 100, num_samples)  # Distance in km
    
    # Calculate arrival time based on simplified physics
    v_max = 80  # km/h, maximum speed for Philippine trains
    track_factor = {'good': 1, 'fair': 0.9, 'poor': 0.7}
    weather_factor = {'clear': 1, 'rainy': 0.9, 'stormy': 0.7}
    
    v_effective = v_max * np.array([track_factor[t] for t in track_condition]) * np.array([weather_factor[w] for w in weather_condition])
    travel_time = distance / v_effective
    arrival_time = time + travel_time + (signal_delay + boarding_time) / 60  # Convert minutes to hours
    
    data = pd.DataFrame({
        'time': time,
        'track_condition': track_condition,
        'weather_condition': weather_condition,
        'signal_delay': signal_delay,
        'boarding_time': boarding_time,
        'distance': distance,
        'arrival_time': arrival_time
    })
    
    return data

def prepare_data(num_samples=1000, test_size=0.2, random_state=42):
    data = generate_synthetic_data(num_samples)
    
    # Convert categorical variables to numerical
    data['track_condition'] = pd.Categorical(data['track_condition']).codes
    data['weather_condition'] = pd.Categorical(data['weather_condition']).codes
    
    features = ['time', 'track_condition', 'weather_condition', 'signal_delay', 'boarding_time', 'distance']
    target = 'arrival_time'
    
    X = data[features]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = prepare_data()
    
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    joblib.dump(scaler, 'scaler.pkl')
    
    print("Synthetic data generated, prepared, and saved.")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
