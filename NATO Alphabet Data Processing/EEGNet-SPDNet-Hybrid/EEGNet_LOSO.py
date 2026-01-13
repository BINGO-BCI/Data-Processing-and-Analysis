import os
import glob
import numpy as np
import torch
from torch import nn
from braindecode.datasets import create_from_X_y
from braindecode.models import EEGNet
from braindecode import EEGClassifier
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from skorch.helper import SliceDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import utils
import matplotlib.pyplot as plt
import pandas as pd

channel_names = ['P3','C3','F3','Fz','F4','C4','P4','Cz','Pz','Fp1','Fp2','T3','T5','O1','O2','F7','F8','T6','T4']

days = ["DAY1", "DAY2", "DAY3"]
data_by_day = {d: [] for d in days}


for p in glob.glob("/data/*.npz"):
    fname = os.path.basename(p)
    subj, day = fname.replace(".npz", "").split("_")
    z = np.load(p, allow_pickle=True)
    data_by_day[day].append(
        {
            "subject": subj,
            "trials": z["trials"],   # (trials, chans, time)
            "labels": z["labels"],   # (trials,)
            "time":   z["time"]})
    
subjects = sorted(set([d["subject"] for day_data in data_by_day.values() for d in day_data]))

n_chans = 19
input_window_seconds = 1.5
sfreq=300
loso_results = []

# iterate over each subject (leave-one-subject-out)
for test_subject in subjects:
    print(f"test_subject {test_subject}")

    train_data = {d: [] for d in days}
    test_data = {d: [] for d in days}

    # Split the data based on subjects - use data from all three days
    for day in days:
        for d in data_by_day[day]:
            if d["subject"] == test_subject:
                test_data[day].append(d)  # Add data to test set
            else:
                train_data[day].append(d)  # Add data to train set

    X_train_all = []
    y_train_all = []
    X_test_all = []
    y_test_all = []

    for day in days:
        for d in train_data[day]:
            time = d["time"]
            trials = d["trials"]  
            labels = d["labels"]  

            # onset of imagined speech 
            start_index = np.where(time == 0)[0][0]
            X_train_all.append(trials[:, :, start_index + 1:])  
            y_train_all.extend(labels)

        for d in test_data[day]:
            time = d["time"]
            trials = d["trials"]  
            labels = d["labels"]  
            start_index = np.where(time == 0)[0][0]
            X_test_all.append(trials[:, :, start_index + 1:])  
            y_test_all.extend(labels)  


    X_train_all = np.concatenate(X_train_all, axis=0)  # Combine all trials for training
    X_test_all = np.concatenate(X_test_all, axis=0)  # Combine all trials for testing
    y_train_all = np.array(y_train_all)  
    y_test_all = np.array(y_test_all)

    unique_words = np.unique(y_train_all)
    n_classes = len(unique_words)
    print(f"Number of classes: {n_classes}")
    assert n_classes == 26, f"Expected 26 classes, got {n_classes}"
    # word -> index
    word_to_idx = {w: i for i, w in enumerate(unique_words)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    y_train_int = np.array([word_to_idx[w] for w in y_train_all], dtype=np.int64)
    y_test_int = np.array([word_to_idx[w] for w in y_test_all], dtype=np.int64)
    
    window_size_samples = X_train_all.shape[-1]  
    window_stride_samples = window_size_samples
    
    # Band-pass filter the data
    X_train_all = utils.cheby_bandpass(X_train_all, l_freq=1.0, h_freq=145.0, sfreq=sfreq)
    X_test_all = utils.cheby_bandpass(X_test_all, l_freq=1.0, h_freq=145.0, sfreq=sfreq)

    # Z-score normalization
    X_train_all, X_test_all = utils.zscore(X_train_all, X_test_all)

    windows_dataset_train = create_from_X_y(X_train_all, y_train_int, drop_last_window=False,  sfreq=sfreq,
        ch_names=channel_names, window_size_samples=window_size_samples, window_stride_samples=window_stride_samples)

    windows_dataset_test = create_from_X_y(X_test_all, y_test_int, drop_last_window=False, sfreq=sfreq,
        ch_names=channel_names, window_size_samples=window_size_samples, window_stride_samples=window_stride_samples)


    model = EEGNet(n_chans=n_chans, n_outputs=n_classes, input_window_seconds=input_window_seconds, sfreq=sfreq)

    clf = EEGClassifier(model,criterion=nn.CrossEntropyLoss, optimizer=torch.optim.AdamW,
        batch_size=64, train_split=ValidSplit(0.2),
        callbacks=["accuracy", EarlyStopping(monitor="valid_loss", patience=10)],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        classes=list(range(n_classes)), max_epochs=2000)

    clf.fit(windows_dataset_train, y=None)

    # evaluation on held-out test set
    print("\n=== Evaluation on held-out test set ===")
    y_test = np.array([y for y in SliceDataset(windows_dataset_test, idx=1)])
    test_acc = clf.score(windows_dataset_test, y=y_test)
    print(f"Final test accuracy: {test_acc * 100:.2f}%")
    utils.save_results(day, test_subject, "test_acc", file_name_text=str(test_acc))

    # Confusion Matrix
    y_pred = clf.predict(windows_dataset_test)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[idx_to_word[i] for i in range(n_classes)])
    disp.plot(include_values=True, cmap="viridis", ax=plt.gca(), xticks_rotation="vertical")
    plt.title("Confusion Matrix on Held-Out Test Set")
    #plt.tight_layout()
    utils.save_results(day, test_subject, "confusion_matrix", plot=plt.gcf(), confusion_matrix=cm)  

    # Classification Report
    report = classification_report(y_test, y_pred, target_names=[idx_to_word[i] for i in range(n_classes)], digits=3)
    print("\nClassification Report:")
    print(report)
    utils.save_results(day, test_subject, "classification_report", file_name_text=report)
    
    # Per-class accuracy
    per_class_counts = cm.sum(axis=1)
    per_class_correct = cm.diagonal()
    per_class_acc = per_class_correct / per_class_counts
    utils.save_results(day, test_subject, "per_class_acc", file_name_text=per_class_acc)
    
    # Weighted Accuracy Calculation
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
    total_weighted_acc = sum(per_class_weighted_acc) / sum([letter_weights.get(idx_to_word[i][0].upper(), 1.0) 
                                                            for i in range(n_classes)])
    print(f"Total Weighted Accuracy: {total_weighted_acc * 100:.2f}%")
    utils.save_results(day, test_subject, "total_weighted_acc", file_name_text=str(total_weighted_acc))
    utils.save_results(day, test_subject,"per_class_weighted_acc", file_name_text=per_class_weighted_acc)

    # Save the results for this fold
    loso_results.append({"subject": test_subject,"test_acc": test_acc})

    print(f"Completed LOSO fold for test_subject {test_subject}")
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
