import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.pytorch.bilstm import AGTSFNet


# ── 1. Model Wrapper for SHAP ─────────────────────────────────────────────────
class ModelWrapper:
    def __init__(self, model, device):
        self.model  = model
        self.device = device
        self.model.eval()

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(X_tensor, X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs


# ── 2. SHAP Explainer ─────────────────────────────────────────────────────────
class TrafficSHAPExplainer:
    def __init__(self, model, device, feature_names=None):
        self.wrapper = ModelWrapper(model, device)
        self.feature_names = feature_names or [
            'traffic_volume', 'temp', 'rain_1h', 'snow_1h',
            'clouds_all', 'hour', 'dayofweek', 'month'
        ]
        self.explainer   = None
        self.shap_values = None

    def fit(self, X_background, n_background=100):
        print(f"Fitting SHAP explainer on {n_background} background samples...")
        # Use a small random background for efficiency
        idx = np.random.choice(len(X_background), n_background, replace=False)
        background = X_background[idx]

        # Flatten for SHAP (batch, seq*features)
        background_flat = background.reshape(n_background, -1)

        def predict_flat(X_flat):
            X_3d = X_flat.reshape(-1, background.shape[1], background.shape[2])
            return self.wrapper.predict(X_3d)

        self.explainer = shap.KernelExplainer(predict_flat, background_flat)
        print("✅ SHAP explainer ready!")

    def explain(self, X_samples, n_samples=20):
        print(f"Explaining {n_samples} predictions...")
        samples = X_samples[:n_samples]
        samples_flat = samples.reshape(n_samples, -1)
        self.shap_values = self.explainer.shap_values(
            samples_flat, nsamples=100, silent=True
        )
        print("✅ SHAP values computed!")
        return self.shap_values

    def plot_summary(self, X_samples, n_samples=20, save_path="experiments/results"):
        if self.shap_values is None:
            self.explain(X_samples, n_samples)

        os.makedirs(save_path, exist_ok=True)
        samples_flat = X_samples[:n_samples].reshape(n_samples, -1)

        # Feature names repeated for each timestep
        seq_len = X_samples.shape[1]
        flat_feature_names = [
            f"{f}_t{t}" for t in range(seq_len)
            for f in self.feature_names
        ]

        # Summary bar plot
        shap_arr = np.array(self.shap_values)
        mean_abs  = np.abs(shap_arr).mean(axis=0)

        # Aggregate by feature (sum across timesteps)
        n_features = len(self.feature_names)
        feature_importance = np.zeros(n_features)
        for i, name in enumerate(self.feature_names):
            indices = [j for j, fn in enumerate(flat_feature_names) if fn.startswith(name)]
            feature_importance[i] = mean_abs[indices].sum()

        # Plot
        plt.figure(figsize=(10, 6))
        sorted_idx = np.argsort(feature_importance)
        plt.barh(
            [self.feature_names[i] for i in sorted_idx],
            feature_importance[sorted_idx],
            color='steelblue'
        )
        plt.xlabel("Mean |SHAP Value|")
        plt.title("AGTSF-Net — Feature Importance (SHAP)\nWhich features drive anomaly predictions?")
        plt.tight_layout()
        plt.savefig(f"{save_path}/shap_feature_importance.png", dpi=150)
        plt.close()
        print(f"✅ SHAP plot saved to {save_path}/shap_feature_importance.png")

        # Print top features
        print("\nTop features driving anomaly predictions:")
        for i in sorted_idx[::-1]:
            print(f"  {self.feature_names[i]:<20} {feature_importance[i]:.4f}")

        return feature_importance


# ── 3. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load model
    model = AGTSFNet(temporal_input_size=8, spatial_input_size=8)
    model.load_state_dict(
        torch.load("experiments/results/best_model.pt", map_location=device)
    )
    model = model.to(device)
    model.eval()
    print("✅ Model loaded!")

    # Load data
    X_train = np.load("data/processed/X_train.npy")
    X_test  = np.load("data/processed/X_test.npy")
    y_test  = np.load("data/processed/y_test.npy")

    # Get some anomaly samples to explain
    anomaly_idx = np.where(y_test == 1)[0][:20]
    X_anomalies = X_test[anomaly_idx]
    print(f"Explaining {len(X_anomalies)} anomaly samples...")

    # Run SHAP
    explainer = TrafficSHAPExplainer(model, device)
    explainer.fit(X_train, n_background=50)
    explainer.explain(X_anomalies, n_samples=20)
    explainer.plot_summary(X_anomalies, n_samples=20)

    print("\n🎉 SHAP explainability complete!")