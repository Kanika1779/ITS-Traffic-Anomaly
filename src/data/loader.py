import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# ── 1. Load UCI Metro (Temporal Stream) ──────────────────────────────────────
def load_uci_metro(path="data/raw/uci_metro/Metro_Interstate_Traffic_Volume.csv"):
    df = pd.read_csv(path)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time').reset_index(drop=True)

    # Extract time features
    df['hour'] = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month

    # Keep useful numeric columns
    features = ['traffic_volume', 'temp', 'rain_1h', 'snow_1h',
                'clouds_all', 'hour', 'dayofweek', 'month']
    df = df[features].fillna(0)

    # Normalize
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    print(f"UCI Metro loaded: {df.shape}")
    return df, scaler

# ── 2. Load Indian Violations (Anomaly Labels) ────────────────────────────────
def load_indian_violations(path="data/raw/indian_accident/Indian_Traffic_Violations.csv"):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                      errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Create anomaly label
    # High speed + alcohol + previous violations = anomaly
    df['anomaly'] = 0
    df.loc[
        (df['Recorded_Speed'] > df['Speed_Limit']) |
        (df['Alcohol_Level'] > 0) |
        (df['Previous_Violations'] >= 3),
        'anomaly'
    ] = 1

    print(f"Indian Violations loaded: {df.shape}")
    print(f"Anomaly distribution:\n{df['anomaly'].value_counts()}")
    return df

# ── 3. Load Indian Accident (Severity Labels) ─────────────────────────────────
def load_indian_accident(path="data/raw/indian_violation/accident_prediction_india.csv"):
    df = pd.read_csv(path)

    # Create binary anomaly label from severity
    df['anomaly'] = df['Accident Severity'].apply(
        lambda x: 0 if str(x).lower() == 'minor' else 1
    )

    print(f"Indian Accident loaded: {df.shape}")
    print(f"Anomaly distribution:\n{df['anomaly'].value_counts()}")
    return df

# ── 4. Sliding Window Generator ───────────────────────────────────────────────
def create_windows(data, window_size=60, step=1):
    X, y = [], []
    values = data.values if isinstance(data, pd.DataFrame) else data

    for i in range(0, len(values) - window_size, step):
        window = values[i:i + window_size, :-1]   # features
        label  = values[i + window_size - 1, -1]  # last label
        X.append(window)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"Windows created: X={X.shape}, y={y.shape}")
    return X, y

# ── 5. Train/Val/Test Split ───────────────────────────────────────────────────
def split_data(X, y, train=0.7, val=0.15):
    n = len(X)
    t = int(n * train)
    v = int(n * (train + val))

    X_train, y_train = X[:t], y[:t]
    X_val,   y_val   = X[t:v], y[t:v]
    X_test,  y_test  = X[v:], y[v:]

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ── 6. Main Pipeline ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    # Load
    uci_df, scaler = load_uci_metro()
    violations_df  = load_indian_violations()
    accident_df    = load_indian_accident()

    # Add a dummy anomaly column to uci for windowing
    uci_df['anomaly'] = 0

    # Create windows
    X, y = create_windows(uci_df, window_size=60)

    # Split
    train, val, test = split_data(X, y)

    # Save
    np.save("data/processed/X_train.npy", train[0])
    np.save("data/processed/y_train.npy", train[1])
    np.save("data/processed/X_val.npy",   val[0])
    np.save("data/processed/y_val.npy",   val[1])
    np.save("data/processed/X_test.npy",  test[0])
    np.save("data/processed/y_test.npy",  test[1])

    print("\n✅ All data preprocessed and saved to data/processed/")