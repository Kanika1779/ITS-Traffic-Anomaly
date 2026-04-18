import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import os


# ── 1. Load UCI Metro (Temporal Stream) ──────────────────────────────────────
def load_uci_metro(path="data/raw/uci_metro/Metro_Interstate_Traffic_Volume.csv"):
    df = pd.read_csv(path)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time').reset_index(drop=True)

    df['hour']      = df['date_time'].dt.hour
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['month']     = df['date_time'].dt.month

    features = ['traffic_volume', 'temp', 'rain_1h', 'snow_1h',
                'clouds_all', 'hour', 'dayofweek', 'month']
    df = df[features].fillna(0)

    # ── Anomaly labels BEFORE normalizing ──
    # Percentile method: bottom 5% + top 10% = anomaly
    low_thresh  = df['traffic_volume'].quantile(0.05)
    high_thresh = df['traffic_volume'].quantile(0.90)

    anomaly_labels = (
        (df['traffic_volume'] <= low_thresh) |
        (df['traffic_volume'] >= high_thresh)
    ).astype(float)

    print(f"Anomaly distribution (Percentile method):")
    print(pd.Series(anomaly_labels).value_counts())
    print(f"Anomaly rate: {anomaly_labels.mean()*100:.2f}%")

    # ── Normalize ──
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    df['anomaly'] = anomaly_labels.values
    print(f"UCI Metro loaded: {df.shape}")
    return df, scaler


# ── 2. Load Indian Violations ─────────────────────────────────────────────────
def load_indian_violations(path="data/raw/indian_accident/Indian_Traffic_Violations.csv"):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(
        df['Date'] + ' ' + df['Time'], errors='coerce'
    )
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    df['anomaly'] = 0
    df.loc[
        (df['Recorded_Speed'] > df['Speed_Limit']) |
        (df['Alcohol_Level'] > 0) |
        (df['Previous_Violations'] >= 3),
        'anomaly'
    ] = 1

    print(f"\nIndian Violations loaded: {df.shape}")
    print(f"Anomaly distribution:\n{df['anomaly'].value_counts()}")
    return df


# ── 3. Load Indian Accident ───────────────────────────────────────────────────
def load_indian_accident(path="data/raw/indian_violation/accident_prediction_india.csv"):
    df = pd.read_csv(path)
    df['anomaly'] = df['Accident Severity'].apply(
        lambda x: 0 if str(x).lower() == 'minor' else 1
    )
    print(f"\nIndian Accident loaded: {df.shape}")
    print(f"Anomaly distribution:\n{df['anomaly'].value_counts()}")
    return df


# ── 4. Sliding Window Generator ───────────────────────────────────────────────
def create_windows(data, window_size=60, step=1):
    X, y = [], []
    values = data.values if isinstance(data, pd.DataFrame) else data

    for i in range(0, len(values) - window_size, step):
        window = values[i:i + window_size, :-1]
        label  = values[i + window_size - 1, -1]
        X.append(window)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    print(f"\nWindows created: X={X.shape}, y={y.shape}")
    print(f"Label distribution: 0={int((y==0).sum())} | 1={int((y==1).sum())}")
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

    uci_df, scaler = load_uci_metro()
    violations_df  = load_indian_violations()
    accident_df    = load_indian_accident()

    X, y = create_windows(uci_df, window_size=60)
    train, val, test = split_data(X, y)

    np.save("data/processed/X_train.npy", train[0])
    np.save("data/processed/y_train.npy", train[1])
    np.save("data/processed/X_val.npy",   val[0])
    np.save("data/processed/y_val.npy",   val[1])
    np.save("data/processed/X_test.npy",  test[0])
    np.save("data/processed/y_test.npy",  test[1])

    print("\n✅ All data preprocessed and saved to data/processed/")