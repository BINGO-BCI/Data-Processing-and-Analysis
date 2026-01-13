import os
import glob
import numpy as np
import torch
from torch import nn
from braindecode.datasets import create_from_X_y
from braindecode.models import EEGConformer
from braindecode.util import set_random_seeds
from braindecode import EEGClassifier
from skorch.callbacks import EarlyStopping
from skorch.helper import SliceDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import utils
import matplotlib.pyplot as plt
import pandas as pd
import json

days = ("DAY1", "DAY2", "DAY3")
data = {d: {} for d in days}

for p in glob.glob("/data/*.npz"):
    stem = os.path.splitext(os.path.basename(p))[0]  # "S1_DAY1"
    parts = stem.split("_")
    subj, day = parts

    z = np.load(p, allow_pickle=True)
    data[day][subj] = {
        "trials": z["trials"],   
        "labels": z["labels"],  
        "time":   z["time"]}

test_day=3
subject = "S20" #choose subject
d1 = data["DAY1"].get(subject)
d2 = data["DAY2"].get(subject)
d3 = data["DAY3"].get(subject)
# Train = DAY1+DAY2, Test = DAY3
trials_train = np.concatenate([d1["trials"], d2["trials"]], axis=0)
labels_train = np.concatenate([d1["labels"], d2["labels"]], axis=0)
trials_test = d3["trials"]
labels_test = d3["labels"]

time = d3["time"]
zero_idx = np.flatnonzero(time == 0)
start_index = int(zero_idx[0])

X_train = trials_train[:, :, start_index + 1:]
X_test  = trials_test[:, :, start_index + 1:]
y_train_words = labels_train
y_test_words  = labels_test

unique_words = np.unique(y_train_words)
n_classes = unique_words.size
print("Number of classes:", n_classes)
assert n_classes == 26, f"Expected 26 classes, got {n_classes}"

word_to_idx = {w: i for i, w in enumerate(unique_words)}
idx_to_word = {i: w for w, i in word_to_idx.items()}  

y_train_int = np.array([word_to_idx[w] for w in y_train_words], dtype=np.int64)
y_test_int  = np.array([word_to_idx[w] for w in y_test_words], dtype=np.int64)

sfreq = 300
channel_names = ['P3','C3','F3','Fz','F4','C4','P4','Cz','Pz','Fp1','Fp2','T3','T5','O1','O2','F7','F8','T6','T4']

#Band-pass filtering
X_train = utils.cheby_bandpass(X_train, l_freq=1.0, h_freq=145.0, sfreq=sfreq)
X_test  = utils.cheby_bandpass(X_test, l_freq=1.0, h_freq=145.0, sfreq=sfreq)

# Z-score
X_train, X_test = utils.zscore(X_train, X_test)

window_size_samples = X_train.shape[-1]
window_stride_samples = window_size_samples

windows_train_dataset = create_from_X_y(X_train, y_train_int,drop_last_window=False, sfreq=sfreq,
    ch_names=channel_names, window_size_samples=window_size_samples, window_stride_samples=window_stride_samples)

windows_test_dataset = create_from_X_y(X_test, y_test_int, drop_last_window=False,sfreq=sfreq,ch_names=channel_names,
    window_size_samples=window_size_samples, window_stride_samples=window_stride_samples)

print("Train trials:", len(windows_train_dataset))
print("Test trials:", len(windows_test_dataset))
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
global_seed = 42
set_random_seeds(seed=global_seed, cuda=cuda)

n_chans = windows_train_dataset[0][0].shape[0]
n_times = windows_train_dataset[0][0].shape[1]
input_window_seconds = n_times / sfreq
classes = list(range(n_classes))


search_space = [
    {"lr": 1e-3, "weight_decay": 1e-5},
    {"lr": 5e-4, "weight_decay": 1e-4},
    {"lr": 5e-4, "weight_decay": 1e-5}]

def run_cv_for_config(cfg, windows_train_dataset, seed):
    """Run 5-fold stratified CV"""
    y_train_all = np.array([y for y in SliceDataset(windows_train_dataset, idx=1)])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_accuracies = []
    cv_history = []
    fold_epochs = []
    train_all_indices = np.arange(len(windows_train_dataset))

    for fold, (fold_train_idx, fold_val_idx) in enumerate(skf.split(train_all_indices, y_train_all), start=1):
        fold_splits = windows_train_dataset.split(
            by={"fold_train": fold_train_idx.tolist(), "fold_val": fold_val_idx.tolist()})
        fold_train_set = fold_splits["fold_train"]
        fold_val_set = fold_splits["fold_val"]

        model = EEGConformer(n_outputs=n_classes,n_chans=n_chans, n_times=n_times,
            sfreq=sfreq, n_filters_time=40, filter_time_length=25, pool_time_length=75, pool_time_stride=15,
            drop_prob=0.5,num_layers=4,num_heads=8, att_drop_prob=0.5, final_fc_length="auto")
        if cuda:
            model.cuda()

        lr = cfg["lr"]
        batch_size = 64
        weight_decay = cfg["weight_decay"]
        max_epochs = 2000

        clf = EEGClassifier(model, criterion=nn.CrossEntropyLoss, optimizer=torch.optim.AdamW,
            optimizer__lr=lr, optimizer__weight_decay=weight_decay, batch_size=batch_size,
            callbacks=[ "accuracy", ("early_stopping", EarlyStopping(monitor="valid_loss", patience=10))],
            device=device, classes=classes, max_epochs=max_epochs)

        # Fit the model with training data
        clf.fit(fold_train_set, y=y_train_all[fold_train_idx])

        history = pd.DataFrame(clf.history)
        cv_history.append(history)

        n_epochs = len(clf.history)
        fold_epochs.append(n_epochs)
        fold_acc = clf.score(fold_val_set, y_train_all[fold_val_idx])
        fold_accuracies.append(fold_acc)

    fold_epochs = np.array(fold_epochs)
    mean_epochs = int(np.round(fold_epochs.mean()))
    fold_accuracies = np.array(fold_accuracies)
    mean_acc = fold_accuracies.mean()
    std_acc = fold_accuracies.std()

    return mean_acc, std_acc, mean_epochs, cv_history


# Track the best configuration based on CV performance
best_cfg = None
best_score = -np.inf
results = []

# Run cross-validation for each hyperparameter configuration        
for cfg in search_space:
    mean_acc, std_acc, mean_epochs, cv_history = run_cv_for_config(cfg, windows_train_dataset, seed=global_seed)
    results.append((cfg, mean_acc, std_acc, mean_epochs, cv_history))
    if mean_acc > best_score:
        best_score = mean_acc
        best_cfg = cfg
        best_mean_epochs=mean_epochs

# Save the best configuration (best hyperparameters) for final training
best_cfg_str = json.dumps(best_cfg, indent=4)
utils.save_results(test_day, subject, "best_cfg", file_name_text=best_cfg_str)

print("\nHyperparameter search results ")
for cfg, mean_acc, std_acc, mean_epochs, cv_history in results:
    print(f"{cfg} -> {mean_acc * 100:.2f} ± {std_acc * 100:.2f}%")
print("Best config:", best_cfg, "with mean CV accuracy:", f"{best_score * 100:.2f}%")

print("\n Final training on full 35/class training set with best config")
set_random_seeds(seed=global_seed, cuda=cuda)

final_model = EEGConformer(n_outputs=n_classes,n_chans=n_chans, n_times=n_times,
    sfreq=sfreq, n_filters_time=40, filter_time_length=25, pool_time_length=75, pool_time_stride=15,
    drop_prob=0.5,num_layers=4,num_heads=8, att_drop_prob=0.5, final_fc_length="auto")
if cuda:
    final_model.cuda()

final_clf = EEGClassifier(final_model, criterion=nn.CrossEntropyLoss, optimizer=torch.optim.AdamW,
    optimizer__lr=best_cfg["lr"],  # Use best configuration from hyperparameter search
    optimizer__weight_decay=best_cfg["weight_decay"], batch_size=64, train_split=None, callbacks=["accuracy"],
    device=device,  classes=classes, max_epochs=best_mean_epochs)

final_clf.fit(windows_train_dataset, y=None)
history = pd.DataFrame(final_clf.history_)

# Find the stored cv_history for best_cfg
best_entry = max(results, key=lambda x: x[1])   # x[1] is mean_acc
best_cfg, best_mean_acc, best_std_acc, best_mean_epochs, best_cv_history = best_entry
utils.plot_cv_curves(best_cv_history, title_prefix=f"Config {best_cfg}", show_mean=True)


# Evaluation on held-out test set from Day 3
print("\n=== Evaluation on held-out Day 3 test set ===")
y_test = np.array([y for y in SliceDataset(windows_test_dataset, idx=1)])
y_pred = final_clf.predict(windows_test_dataset)
test_acc = ((y_pred == y_test).mean() * 100)
print(f"Test accuracy: {test_acc:.2f}%")
utils.save_results(test_day, subject, "test_acc", file_name_text=str(test_acc))


# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))
plt.figure(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[idx_to_word[i] for i in range(n_classes)])
disp.plot(include_values=True, cmap="viridis", ax=plt.gca(), xticks_rotation="vertical")
plt.title("Confusion matrix on held-out test set")
plt.tight_layout()
utils.save_results(test_day, subject, "confusion_matrix", plot=plt.gcf(), confusion_matrix=cm)
#plt.show()

# Per-class accuracy
per_class_counts = cm.sum(axis=1)
per_class_correct = cm.diagonal()
per_class_acc = per_class_correct / per_class_counts
utils.save_results(test_day, subject, "per_class_acc", file_name_text=per_class_acc)

# Classification report
report = classification_report(y_test, y_pred, target_names=[idx_to_word[i] for i in range(n_classes)], digits=3)
utils.save_results(test_day, subject, "classification_report", file_name_text=report)
print(report)


weights = pd.read_excel('letter_weights.xlsx')
letter_weights = {row['Letter']: row['Weight'] for _, row in weights.iterrows()}
weights_values = np.array(list(letter_weights.values()))

per_class_weighted_acc = []
for i in range(n_classes):
    word = idx_to_word[i]
    first_letter = word[0].upper() if word[0].isalpha() else 'default'
    weight = letter_weights.get(first_letter, 0.0)
    class_acc = per_class_acc[i]
    weighted_class_acc = weight * class_acc
    per_class_weighted_acc.append(weighted_class_acc)

# Total weighted accuracy
total_weighted_acc = sum(per_class_weighted_acc) / sum([letter_weights.get(idx_to_word[i][0].upper(), 1.0) for i in range(n_classes)])
print(f"Total Weighted Accuracy: {total_weighted_acc * 100:.2f}%")

utils.save_results(test_day, subject, "total_weighted_acc", file_name_text=str(total_weighted_acc))
utils.save_results(test_day, subject, "per_class_weighted_acc", file_name_text=per_class_weighted_acc)






