import os, glob
import numpy as np
import torch
from torch import nn
from braindecode.datasets import create_from_X_y
from braindecode.models import EEGConformer
from braindecode import EEGClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from skorch.helper import SliceDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)
import utils
import pandas as pd
import pickle


data_all_days = "/data/*.npz"
days = ["DAY1", "DAY2", "DAY3"]

train_day = "DAY2"       # set "DAY1" or "DAY2"
extra_train_day = "DAY3"

n_train_per_subj = 500
n_test_per_subj  = 85
n_reps = 500
base_seed = 42
n_chans = 19
sfreq = 300
max_epochs = 2000
batch_size = 64

channel_names = ['P3','C3','F3','Fz','F4','C4','P4','Cz','Pz','Fp1','Fp2','T3','T5','O1','O2','F7','F8','T6','T4']

weights = pd.read_excel('letter_weights.xlsx')
letter_weights = {row['Letter']: row['Weight'] for _, row in weights.iterrows()}


data_by_day = {d: [] for d in days}
for p in glob.glob(data_all_days):
    fname = os.path.basename(p)
    subj, day = fname.replace(".npz", "").split("_")
    z = np.load(p, allow_pickle=True)
    data_by_day[day].append({
        "subject": subj,
        "trials": z["trials"],
        "labels": z["labels"],
        "time": z["time"]})

subjects = sorted(set(d["subject"] for day_data in data_by_day.values() for d in day_data))

def extract_imagined_segment(trials, time):
    start_index = np.where(time == 0)[0][0]
    return trials[:, :, start_index + 1:]

def collect_subject_day(subject, day):
    X_list, y_list = [], []
    for d in data_by_day[day]:
        if d["subject"] != subject:
            continue
        X = extract_imagined_segment(d["trials"], d["time"])
        y = np.array(d["labels"])
        X_list.append(X)
        y_list.append(y)
    if not X_list:
        return None, None
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


#  13-class mapping from train_day across all subjects
y_source_all = []
for s in subjects:
    _, y = collect_subject_day(s, train_day)
    if y is not None:
        y_source_all.append(y)

y_source_all = np.concatenate(y_source_all, axis=0)
unique_words = np.unique(y_source_all)
n_classes = len(unique_words)
print(f"[{train_day}] global classes: {n_classes}")
assert n_classes == 13, f"Expected 13 classes from {train_day}, got {n_classes}"

word_to_idx = {w: i for i, w in enumerate(unique_words)}
idx_to_word = {i: w for w, i in word_to_idx.items()}


subject_pools = {}
for s in subjects:
    X_src, y_src = collect_subject_day(s, train_day)
    X_ex,  y_ex  = collect_subject_day(s, extra_train_day)

    # Filter DAY3 to the 13 classes
    mask = np.isin(y_ex, unique_words)
    X_ex = X_ex[mask]
    y_ex = y_ex[mask]

    X_pool = np.concatenate([X_src, X_ex], axis=0)
    y_pool_words = np.concatenate([y_src, y_ex], axis=0)
    y_pool_int = np.array([word_to_idx[w] for w in y_pool_words], dtype=np.int64)

    subject_pools[s] = (X_pool, y_pool_int)

print(f"Subjects used: {len(subject_pools)} / {len(subjects)}")


def run_one_repetition(rep_seed):
    # Per-subject splits -> global train/test
    X_train_list, y_train_list = [], []
    X_test_list,  y_test_list  = [], []

    for s, (X_pool, y_pool) in subject_pools.items():
        strf = StratifiedShuffleSplit(n_splits=1,train_size=n_train_per_subj, test_size=n_test_per_subj, random_state=rep_seed)
        tr_idx, te_idx = next(strf.split(np.zeros(len(y_pool)), y_pool))

        X_train_list.append(X_pool[tr_idx])
        y_train_list.append(y_pool[tr_idx])
        X_test_list.append(X_pool[te_idx])
        y_test_list.append(y_pool[te_idx])

    X_train_all = np.concatenate(X_train_list, axis=0)
    y_train_all = np.concatenate(y_train_list, axis=0)
    X_test_all  = np.concatenate(X_test_list, axis=0)
    y_test_all  = np.concatenate(y_test_list, axis=0)

    X_train_bp = utils.cheby_bandpass(X_train_all, sfreq=sfreq)
    X_test_bp  = utils.cheby_bandpass(X_test_all,  sfreq=sfreq)
    X_train, X_test = utils.zscore(X_train_bp, X_test_bp)

    window_size_samples = X_train.shape[-1]
    input_window_seconds = window_size_samples / sfreq

    train_ds = create_from_X_y(X_train, y_train_all, drop_last_window=False, sfreq=sfreq,
        ch_names=channel_names, window_size_samples=window_size_samples,window_stride_samples=window_size_samples)
    
    test_ds = create_from_X_y(X_test, y_test_all, drop_last_window=False, sfreq=sfreq, ch_names=channel_names, 
                              window_size_samples=window_size_samples, window_stride_samples=window_size_samples)


    model = EEGConformer(n_outputs=n_classes, n_chans=n_chans, n_times=int(input_window_seconds*sfreq),
        sfreq=sfreq, n_filters_time=40, filter_time_length=25, pool_time_length=75,      
        pool_time_stride=15, drop_prob=0.5,num_layers=4, num_heads=8,  att_drop_prob=0.5,final_fc_length="auto")

    clf = EEGClassifier(model, criterion=nn.CrossEntropyLoss, optimizer=torch.optim.AdamW,
        batch_size=batch_size, train_split=ValidSplit(0.2),
        callbacks=["accuracy", EarlyStopping(monitor="valid_loss", patience=10)],
        device='cuda' if torch.cuda.is_available() else 'cpu', classes=list(range(n_classes)),
        max_epochs=max_epochs)

    clf.fit(train_ds, y=None)

    y_true = np.array([y for y in SliceDataset(test_ds, idx=1)])
    y_pred = clf.predict(test_ds)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    
    per_class_counts = cm.sum(axis=1)
    per_class_correct = cm.diagonal()
    per_class_acc = per_class_correct / per_class_counts
    
    per_class_weighted_acc = []
    for i in range(n_classes):
        word = idx_to_word[i]
        first_letter = word[0].upper() if word[0].isalpha() else 'default'
        weight = letter_weights.get(first_letter, 0.0)
        class_acc = per_class_acc[i]
        weighted_class_acc = weight * class_acc
        per_class_weighted_acc.append(weighted_class_acc)

    total_weighted_acc = sum(per_class_weighted_acc) / sum([letter_weights.get(idx_to_word[i][0].upper(), 1.0) for i in range(n_classes)])
    report=classification_report(y_true, y_pred, target_names=[idx_to_word[i] for i in range(n_classes)], digits=3)
    
    return {
        "acc": float(acc),
        "per_class_acc": per_class_acc,
        "total_weighted_acc": float(total_weighted_acc),
        "per_class_weighted_acc": per_class_weighted_acc,
         "cm": cm,
         "report": report}



results = []
for r in range(n_reps):
    rep_seed = base_seed + r
    print(f"\n-Repetition {r+1}/{n_reps}")
    res = run_one_repetition(rep_seed)
    print(f"acc={res['acc']*100:.2f}%")
    results.append(res)
    
    
  
accs = np.array([d["acc"] for d in results])
print("\n SUMMARY")
print(f"Accuracy: {accs.mean()*100:.2f}% ± {accs.std()*100:.2f}%")


# Save the results list to a pickle file
with open('results_Conformer_MC_day2.pkl', 'wb') as f:
    pickle.dump(results, f)







