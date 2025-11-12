import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import os
import time

# === Load corrupted MNIST (motion_blur version) ===
ds_train = tfds.load("mnist_corrupted/motion_blur", split="train", as_supervised=True)
ds_test = tfds.load("mnist_corrupted/motion_blur", split="test", as_supervised=True)

def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

train_ds = ds_train.map(normalize_img).batch(128).prefetch(tf.data.AUTOTUNE)
test_ds = ds_test.map(normalize_img).batch(128).prefetch(tf.data.AUTOTUNE)

# === Model builder ===
def build_model(num_hidden_layers=1, learning_rate=0.001):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28, 1)))

    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(10))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


# === Learning Rate Search ===
def search_learning_rate(initial_lr, num_hidden_layers, tolerance=1e-4, run_counter_start=0):
    best_lr = initial_lr
    best_acc = 0.0
    step_factor = 2.0
    tested_lrs = set()
    history_log = []
    run_counter = run_counter_start

    while True:
        improved = False
        for lr_candidate in [best_lr / step_factor, best_lr * step_factor]:
            lr_candidate = round(lr_candidate, 6)
            if lr_candidate <= 1e-6 or lr_candidate >= 1.0:
                continue
            if lr_candidate in tested_lrs:
                continue
            tested_lrs.add(lr_candidate)

            model = build_model(num_hidden_layers, lr_candidate)
            start = time.time()
            model.fit(train_ds, epochs=3, verbose=0)
            test_loss, test_acc = model.evaluate(test_ds, verbose=0)
            elapsed = time.time() - start

            run_counter += 1
            history_log.append({
                "run_order": run_counter,
                "phase": "learning_rate",
                "initial_lr": initial_lr,
                "num_hidden_layers": num_hidden_layers,
                "lr_candidate": lr_candidate,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "train_time_sec": elapsed
            })

            print(f"LR Search | start={initial_lr:.5f} | candidate={lr_candidate:.6f} | acc={test_acc:.4f}")

            if test_acc > best_acc + tolerance:
                best_acc = test_acc
                best_lr = lr_candidate
                improved = True

        if not improved:
            break

    print(f"✅ Converged LR: {best_lr:.6f} (acc={best_acc:.4f})")
    return best_lr, best_acc, history_log, run_counter


# === Number of Layers Search ===
def search_num_layers(initial_layers, learning_rate, tolerance=1e-4, run_counter_start=0):
    best_layers = initial_layers
    best_acc = 0.0
    step_factor = 2
    tested_layers = set()
    history_log = []
    run_counter = run_counter_start

    while True:
        improved = False
        for layer_candidate in [max(1, best_layers // step_factor), best_layers * step_factor]:
            if layer_candidate > 6:
                continue
            if layer_candidate in tested_layers:
                continue
            tested_layers.add(layer_candidate)

            model = build_model(layer_candidate, learning_rate)
            start = time.time()
            model.fit(train_ds, epochs=3, verbose=0)
            test_loss, test_acc = model.evaluate(test_ds, verbose=0)
            elapsed = time.time() - start

            run_counter += 1
            history_log.append({
                "run_order": run_counter,
                "phase": "layers",
                "initial_layers": initial_layers,
                "learning_rate": learning_rate,
                "layer_candidate": layer_candidate,
                "test_acc": test_acc,
                "test_loss": test_loss,
                "train_time_sec": elapsed
            })

            print(f"Layer Search | start={initial_layers} | candidate={layer_candidate} | acc={test_acc:.4f}")

            if test_acc > best_acc + tolerance:
                best_acc = test_acc
                best_layers = layer_candidate
                improved = True

        if not improved:
            break

    print(f"✅ Converged Layers: {best_layers} (acc={best_acc:.4f})")
    return best_layers, best_acc, history_log, run_counter


# === Multiple Starting Conditions ===
starting_lrs = [1e-4, 1e-3, 1e-2]
starting_layers = [1, 5, 10]

all_results = []
run_counter = 0

for start_lr in starting_lrs:
    for start_layers in starting_layers:
        print(f"\n🚀 Starting new search: initial_lr={start_lr}, initial_layers={start_layers}")
        best_lr, lr_acc, lr_history, run_counter = search_learning_rate(start_lr, start_layers, run_counter_start=run_counter)
        best_layers, final_acc, layer_history, run_counter = search_num_layers(start_layers, best_lr, run_counter_start=run_counter)
        all_results.extend(lr_history + layer_history)
        print(f"🏁 Finished run: best_lr={best_lr:.6f}, best_layers={best_layers}, acc={final_acc:.4f}")

# === Combine and Save Results ===
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(by="run_order").reset_index(drop=True)

print("\n=== Full Optimization History (in run order) ===")
print(results_df)

# Save CSV to same directory as script
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()

csv_path = os.path.join(script_dir, "nn_task1_optimization_results.csv")
results_df.to_csv(csv_path, index=False)
print(f"\n✅ Results saved to: {csv_path}")
