import os
import glob
import numpy as np
import torch
from torch import nn
from braindecode.datasets import create_from_X_y
from braindecode.models import EEGNet
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
data_by_day_subject = {d: {} for d in days}  

for p in glob.glob("/data/*.npz"):
    fname = os.path.splitext(os.path.basename(p))[0]  
    parts = fname.split("_")
    subj, day = parts

    z = np.load(p, allow_pickle=True)
    data_by_day_subject[day][subj] = {
        "subject": subj,
        "trials": z["trials"],
        "labels": z["labels"],
        "time": z["time"]}


# Select subject and day
day = "DAY2"
subject = "S6"
d = data_by_day_subject[day][subject]
time = d["time"]
trials = d["trials"]
y_words = d["labels"]

zero_idx = np.flatnonzero(time == 0)
start_index = int(zero_idx[0])

X = trials[:, :, start_index + 1:]  # keep only after onset

unique_words, y_int = np.unique(y_words, return_inverse=True)
n_classes = unique_words.size
print("Number of classes:", n_classes)
assert n_classes == 13, f"Expected 13 classes, got {n_classes}"

word_to_idx = {w: i for i, w in enumerate(unique_words)}
idx_to_word = {i: w for i, w in enumerate(unique_words)}

# Per-class split: 30 train + 5 test 
sfreq = 300
channel_names = ['P3','C3','F3','Fz','F4','C4','P4','Cz','Pz','Fp1','Fp2','T3','T5','O1','O2','F7','F8','T6','T4']

rng = np.random.RandomState(42)
train_idx_all = []
test_idx_all = []

for c in range(n_classes):
    class_inds = np.flatnonzero(y_int == c)
    n_c = class_inds.size
    print(f"Class {c}: {n_c} trials")
    assert n_c == 35, f"Class {c} has {n_c} trials, expected 35"

    rng.shuffle(class_inds)
    test_idx_all.append(class_inds[:5])
    train_idx_all.append(class_inds[5:35])

train_idx_all = np.concatenate(train_idx_all)
test_idx_all  = np.concatenate(test_idx_all)

print("Total training trials (30/class):", train_idx_all.size)  # training trials-> 13*30 = 390
print("Total test trials (5/class):", test_idx_all.size)        # test trials-> 13*5 = 65

X = utils.cheby_bandpass(X, l_freq=1.0, h_freq=145.0, sfreq=sfreq)

#Z-score
X_train_pre = X[train_idx_all]
X_test_pre  = X[test_idx_all]
X_train_pre, X_test_pre = utils.zscore(X_train_pre, X_test_pre)

X = X.copy()
X[train_idx_all] = X_train_pre
X[test_idx_all]  = X_test_pre

window_size_samples = X.shape[-1]
window_stride_samples = window_size_samples

windows_dataset = create_from_X_y(X, y_int, drop_last_window=False, sfreq=sfreq,
    ch_names=channel_names,window_size_samples=window_size_samples,window_stride_samples=window_stride_samples)

# Split based on trial indices
splits_full = windows_dataset.split(by={"train_all": train_idx_all.tolist(),"test": test_idx_all.tolist()})

train_all_set = splits_full["train_all"]
test_set = splits_full["test"]

print("train_all_set windows:", len(train_all_set))
print("test_set windows:", len(test_set))

y_train_all = np.array([y for y in SliceDataset(train_all_set, idx=1)])
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"

global_seed = 42
set_random_seeds(seed=global_seed, cuda=cuda)

n_chans = windows_dataset[0][0].shape[0]
n_times = windows_dataset[0][0].shape[1]
input_window_seconds = n_times / sfreq
classes = list(range(n_classes))

# Hyperparameter search with 5-fold CV on train_all_set
search_space = [
    {"lr": 1e-3, "weight_decay": 1e-5},  
    {"lr": 1e-3, "weight_decay": 1e-4},  
    {"lr": 1e-4, "weight_decay": 0.0}]


def run_cv_for_config(cfg, train_all_set, y_train_all, seed):
    """Run 5-fold stratified CV"""

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_accuracies = []
    fold_epochs = [] 
    cv_history=[]

    train_all_indices = np.arange(len(train_all_set))

    for fold, (fold_train_idx, fold_val_idx) in enumerate(skf.split(train_all_indices, y_train_all), start=1):

        fold_splits = train_all_set.split(by={"fold_train": fold_train_idx.tolist(),"fold_val":fold_val_idx.tolist()})
        fold_train_set = fold_splits["fold_train"]
        fold_val_set   = fold_splits["fold_val"]
        set_random_seeds(seed=seed + fold, cuda=cuda)

        model = EEGNet(n_chans=n_chans, n_outputs=n_classes, input_window_seconds=input_window_seconds,sfreq=sfreq)
        if cuda:
            model.cuda()

        lr = cfg["lr"]
        batch_size = 64
        weight_decay = cfg["weight_decay"]
        max_epochs = 200

        clf = EEGClassifier(model, criterion=nn.CrossEntropyLoss, optimizer=torch.optim.AdamW,
            optimizer__lr=lr,optimizer__weight_decay=weight_decay,batch_size=batch_size,
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
        print(f"  fold {fold} val accuracy: {fold_acc * 100:.2f}%")
        
    fold_epochs = np.array(fold_epochs)
    mean_epochs = int(np.round(fold_epochs.mean()))
    fold_accuracies = np.array(fold_accuracies)
    mean_acc = fold_accuracies.mean()
    std_acc = fold_accuracies.std()

    return mean_acc, std_acc, mean_epochs, cv_history

# Run search
best_cfg = None
best_score = -np.inf
results = []

for cfg in search_space:
    mean_acc, std_acc, mean_epochs, cv_history = run_cv_for_config(cfg, train_all_set, y_train_all, seed=global_seed)
    results.append((cfg, mean_acc, std_acc, mean_epochs, cv_history))
    if mean_acc > best_score:
        best_score = mean_acc
        best_cfg = cfg
        best_mean_epochs=mean_epochs

print("\nHyperparameter search results")
for cfg, mean_acc, std_acc, mean_epochs, cv_history in results:
    print(f"{cfg} -> {mean_acc * 100:.2f} ± {std_acc * 100:.2f}%")
print("Best config:", best_cfg, "with mean CV accuracy:", f"{best_score * 100:.2f}%")

best_cfg_str = json.dumps(best_cfg, indent=4)
utils.save_results(day, subject, "best_cfg", file_name_text=best_cfg)

# Final training on all 30/class with best config + evaluation on 5/class held-out test set
print("\n Final training on full 30/class training set with best config")
set_random_seeds(seed=global_seed, cuda=cuda)

final_model = EEGNet(n_chans=n_chans, n_outputs=n_classes, input_window_seconds=input_window_seconds,sfreq=sfreq)
if cuda:
    final_model.cuda()

final_clf = EEGClassifier(final_model, criterion=nn.CrossEntropyLoss,optimizer=torch.optim.AdamW,
    optimizer__lr=best_cfg["lr"],optimizer__weight_decay=best_cfg["weight_decay"], callbacks=[ "accuracy"],
    batch_size=64,train_split=None, device=device,classes=classes, max_epochs=best_mean_epochs)

final_clf.fit(train_all_set, y=None)

history = pd.DataFrame(final_clf.history_)


# Find the stored cv_history for best_cfg
best_entry = max(results, key=lambda x: x[1])   # x[1] is mean_acc
best_cfg, best_mean_acc, best_std_acc, best_mean_epochs, best_cv_history = best_entry
utils.plot_cv_curves( best_cv_history, title_prefix=f"Config {best_cfg}",  show_mean=True)

# Get predictions on test_set
y_test = np.array([y for y in SliceDataset(test_set, idx=1)])        
y_pred = final_clf.predict(test_set)                                 
test_acc=((y_pred == y_test).mean() * 100)
print("\nOverall test accuracy: {:.2f}%".format(test_acc))

utils.save_results(day, subject, "test_acc", file_name_text=str(test_acc))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))
plt.figure(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[idx_to_word[i] for i in range(n_classes)])
disp.plot(include_values=True, cmap="viridis", ax=plt.gca(), xticks_rotation="vertical")
plt.title("Confusion matrix on held-out test set")
plt.tight_layout()
# Save the confusion matrix 
utils.save_results(day, subject, "confusion_matrix", plot=plt.gcf(), confusion_matrix=cm)
#plt.show()

# Per-class accuracy from confusion matrix
per_class_counts = cm.sum(axis=1)
per_class_correct = cm.diagonal()
per_class_acc = per_class_correct / per_class_counts 

utils.save_results(day, subject, "per_class_acc", file_name_text=per_class_acc)

print("\nPer-class accuracies:")
for i in range(n_classes):
    word = idx_to_word[i]
    print(f"{i:02d} ({word:15s}) : {per_class_acc[i] * 100:.2f}% "
          f"({per_class_correct[i]}/{per_class_counts[i]})")


target_names = [idx_to_word[i] for i in range(n_classes)]
print("\nClassification report (per-class precision/recall/F1):")
print(classification_report(y_test, y_pred, target_names=target_names, digits=3))
report = classification_report(y_test, y_pred, target_names=target_names, digits=3)

utils.save_results(day, subject, "classification_report", file_name_text=report)


per_class_counts = cm.sum(axis=1)  
per_class_correct = cm.diagonal() 
per_class_acc = per_class_correct / per_class_counts  

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
    
    print(f"Class {i:02d} ({word:15s}) : {class_acc * 100:.2f}% "
          f"({per_class_correct[i]}/{per_class_counts[i]}) Weight: {weight:.4f} "
          f"Weighted Accuracy: {weighted_class_acc * 100:.2f}%")

# Total weighted accuracy
total_weighted_acc = sum(per_class_weighted_acc) / sum([letter_weights.get(idx_to_word[i][0].upper(), 1.0) for i in range(n_classes)])

# Print the overall weighted accuracy
print(f"\nTotal Weighted Accuracy: {total_weighted_acc * 100:.2f}%")

utils.save_results(day, subject, "total_weighted_acc", file_name_text=str(total_weighted_acc))
utils.save_results(day, subject, "per_class_weighted_acc", file_name_text=per_class_weighted_acc)

