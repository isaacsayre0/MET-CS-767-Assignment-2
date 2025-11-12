import pandas as pd
import numpy as np
import tensorflow as tf

import os

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===  Load raw AEP data ===
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

csv_path = os.path.join(script_dir, "AEP_hourly.csv")   
df = pd.read_csv(csv_path, parse_dates=["Datetime"])
df = df.set_index("Datetime").sort_index()

print("✅ Raw data loaded:")
print(df.head())

# === Feature Engineering Function ===
def engineer_features_with_sliding_windows(df, lookback_hours=24):
    df = df.copy()

    # --- Base time features ---
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Cyclic encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # Rolling mean over 7 days
    df["rolling_7d"] = df["AEP_MW"].rolling(window=24 * 7, min_periods=1).mean()

    # Previous-year monthly average
    monthly_avg = (
        df["AEP_MW"]
        .groupby([df.index.year, df.index.month])
        .transform("mean")
    )
    df["prev_year_month_mean"] = monthly_avg.shift(24 * 365)

    # --- Add sliding window features (previous 24 hours) ---
    for lag in range(1, lookback_hours + 1):
        df[f"lag_{lag}h"] = df["AEP_MW"].shift(lag)

    # --- Target is the current hour's load ---
    df["target"] = df["AEP_MW"]
    df = df.drop(columns=["AEP_MW","hour"])

    # Drop rows with missing values
    df = df.dropna()

    return df

# === Architecture search across lag intervals and node sizes ===

lag_intervals = [24, 48, 168]        # 1 day, 2 days, 1 week
node_sizes = [64, 128, 256]          # first layer nodes
dropout_rate = 0.2                   # fixed dropout for both layers

arch_results = []

for lag_hours in lag_intervals:
    print(f"\n🧩 Evaluating lag interval: {lag_hours} hours")

    # Rebuild feature set for each lag interval
    df_feat = engineer_features_with_sliding_windows(df, lookback_hours=lag_hours)

    # Scale numeric features
    to_scale = [col for col in df_feat.columns if col.startswith('lag_')] + ['prev_year_month_mean', 'rolling_7d']
    scaler = StandardScaler()
    df_feat[to_scale] = scaler.fit_transform(df_feat[to_scale])

    # Manual normalization
    df_feat["dayofweek"] /= 6.0
    df_feat["month"] /= 11.0

    # Split into train/test
    train_df = df_feat[df_feat.index.year < 2018]
    test_df  = df_feat[df_feat.index.year == 2018]

    X_train = train_df.drop(columns=["target"]).values
    y_train = train_df["target"].values
    X_test  = test_df.drop(columns=["target"]).values
    y_test  = test_df["target"].values

    for n in node_sizes:
        print(f"   🔹 Testing architecture: {n} → {n//2} nodes")

        # --- Build model ---
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(n, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(n//2, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        # --- Train model ---
        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=40,
            batch_size=128,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )]
        )

        # --- Record results ---
        final_val_loss = history.history['val_loss'][-1]
        final_val_mae  = history.history['val_mae'][-1]
        final_loss     = history.history['loss'][-1]
        final_mae      = history.history['mae'][-1]
        epochs_ran     = len(history.history['loss'])

        arch_results.append({
            'lag_hours': lag_hours,
            'nodes_1': n,
            'nodes_2': n//2,
            'train_loss': final_loss,
            'train_mae': final_mae,
            'val_loss': final_val_loss,
            'val_mae': final_val_mae,
            'epochs': epochs_ran
        })

        print(f"      ➤ Epochs: {epochs_ran}, Train MAE: {final_mae:.2f}, Val MAE: {final_val_mae:.2f}")

# --- Analyze results ---
arch_df = pd.DataFrame(arch_results)
best_arch = arch_df.sort_values(by='val_mae').iloc[0]

print("\n🏆 Best architecture:")
print(best_arch)

# --- Plot MAE heatmap ---
pivot = arch_df.pivot(index='lag_hours', columns='nodes_1', values='val_mae')
plt.figure(figsize=(8,6))
plt.title('Validation MAE by Lag Interval and Node Count')
plt.imshow(pivot, cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(label='Validation MAE')
plt.xticks(range(len(pivot.columns)), [f"{c}" for c in pivot.columns])
plt.yticks(range(len(pivot.index)), [f"{r}" for r in pivot.index])
plt.xlabel('First Layer Nodes')
plt.ylabel('Lag Interval (hours)')
plt.show()

print("\n📊 Full Architecture Search Results:")
print(arch_df.sort_values(by='val_mae').to_string(index=False))

# --- Use best architecture for dropout tuning ---
best_lag = int(best_arch['lag_hours'])
best_nodes = int(best_arch['nodes_1'])
print(f"\n🎯 Using best architecture for dropout tuning: lag={best_lag}, nodes={best_nodes}->{best_nodes//2}")

# Rebuild feature set for the chosen lag
df_feat = engineer_features_with_sliding_windows(df, lookback_hours=best_lag)

# Scale numeric features
to_scale = [col for col in df_feat.columns if col.startswith('lag_')] + ['prev_year_month_mean', 'rolling_7d']
scaler = StandardScaler()
df_feat[to_scale] = scaler.fit_transform(df_feat[to_scale])

df_feat["dayofweek"] /= 6.0
df_feat["month"] /= 11.0

train_df = df_feat[df_feat.index.year < 2018]
test_df  = df_feat[df_feat.index.year == 2018]

X_train = train_df.drop(columns=["target"]).values
y_train = train_df["target"].values
X_test  = test_df.drop(columns=["target"]).values
y_test  = test_df["target"].values

# === Model Builder  ===
def build_model(dropout_1, dropout_2, input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(best_nodes, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(dropout_1),
        tf.keras.layers.Dense(best_nodes//2, activation='relu'),
        tf.keras.layers.Dropout(dropout_2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model


# --- Dropout parameter grid ---
primary_rates = [0.10, 0.15, 0.20, 0.25, 0.30]
secondary_factors = [0, 0.5, 0.75]

results = []

for p in primary_rates:
    for factor in secondary_factors:
        s = round(p * factor, 3)
        print(f"\n🔹 Training model with primary={p:.2f}, secondary={s:.2f}")
        
        model = build_model(p, s, input_dim=X_train.shape[1])
        
        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=40,
            batch_size=128,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )]
        )

        # --- Retrieve final training metrics ---
        final_loss = history.history['loss'][-1]
        final_mae = history.history['mae'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_val_mae = history.history['val_mae'][-1]
        epochs_ran = len(history.history['loss'])

        print(f"    ➤ Epochs: {epochs_ran}")
        print(f"    ➤ Final loss: {final_loss:.2f}, MAE: {final_mae:.2f}")
        print(f"    ➤ Final val_loss: {final_val_loss:.2f}, val_MAE: {final_val_mae:.2f}")

        # Evaluate on test data (optional check)
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

        results.append({
            'primary_dropout': p,
            'secondary_dropout': s,
            'train_loss': final_loss,
            'train_mae': final_mae,
            'val_loss': final_val_loss,
            'val_mae': final_val_mae,
            'test_loss': test_loss,
            'test_mae': test_mae,
            'epochs': epochs_ran
        })

# --- Analyze results ---
results_df = pd.DataFrame(results)
best_config = results_df.sort_values(by='val_mae').iloc[0]

print("\n🏆 Best configuration:")
print(best_config)

# --- Visualization ---
pivot = results_df.pivot(index='primary_dropout', columns='secondary_dropout', values='val_mae')

plt.figure(figsize=(8,6))
plt.title('Validation MAE by Dropout Configuration')
plt.imshow(pivot, cmap='viridis', origin='lower', aspect='auto')
plt.colorbar(label='Validation MAE')
plt.xticks(range(len(pivot.columns)), [f"{c:.2f}" for c in pivot.columns])
plt.yticks(range(len(pivot.index)), [f"{r:.2f}" for r in pivot.index])
plt.xlabel('Secondary Dropout')
plt.ylabel('Primary Dropout')
plt.show()

print("\n📊 Summary of all configurations:")
print(results_df.sort_values(by='val_mae').to_string(index=False))

# ===  Final Model Retraining & Evaluation ===
print("\n🏗️ Retraining final model with best architecture + dropout configuration...")

final_model = tf.keras.Sequential([
    tf.keras.layers.Dense(best_nodes, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(best_nodes//2, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)
])

final_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

final_history = final_model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=int(best_config['epochs']),
    batch_size=128,
    verbose=1
)

# Predict on test data
y_pred = final_model.predict(X_test).flatten()

# Compute performance
test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\n✅ Final Test MAE: {test_mae:.2f}, RMSE: {test_rmse:.2f}")

# Plot predictions vs actual
plt.figure(figsize=(12, 5))
plt.plot(test_df.index, y_test, label="Actual", alpha=0.7)
plt.plot(test_df.index, y_pred, label="Predicted", alpha=0.7)
plt.title(f"Predicted vs Actual AEP Load (Best Lag={best_lag}, Nodes={best_nodes}, Dropout=({0.2:.2f}, {0.1:.2f}))")
plt.xlabel("Datetime")
plt.ylabel("AEP_MW")
plt.legend()
plt.tight_layout()
plt.show()

# === Illustrative Predictions by Time of Day ===
print("\n🌤️  Example Predictions by Time of Day (Morning, Evening, Night)")

# Combine predictions and timestamps
test_df = test_df.copy()
test_df["Predicted"] = y_pred
test_df["Hour"] = test_df.index.hour

# Define conditions
morning = test_df[(test_df["Hour"] >= 6) & (test_df["Hour"] < 10)]
evening = test_df[(test_df["Hour"] >= 11) & (test_df["Hour"] < 16)]
night = test_df[(test_df["Hour"] >= 19)]

# Sample one from each (for illustration)
samples = pd.concat([
    morning.sample(1, random_state=42),
    evening.sample(1, random_state=43),
    night.sample(1, random_state=44)
])

# Display results
for ts, row in samples.iterrows():
    print(f"\n🕒 {ts}  (Hour={row['Hour']})")
    print(f"   Actual   : {row['target']:.2f} MW")
    print(f"   Predicted: {row['Predicted']:.2f} MW")
    print(f"   Error    : {abs(row['target'] - row['Predicted']):.2f} MW")
    
plt.figure(figsize=(7,4))
plt.bar(["Morning", "Afternoon", "Evening"], 
        [samples.iloc[0]["target"], samples.iloc[1]["target"], samples.iloc[2]["target"]],
        alpha=0.6, label="Actual")
plt.bar(["Morning", "Afternoon", "Evening"], 
        [samples.iloc[0]["Predicted"], samples.iloc[1]["Predicted"], samples.iloc[2]["Predicted"]],
        alpha=0.6, label="Predicted")
plt.title("Example Predictions by Time of Day")
plt.ylabel("AEP_MW")
plt.legend()
plt.tight_layout()
plt.show()
