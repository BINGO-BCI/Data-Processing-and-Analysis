from scipy.signal import cheby1, filtfilt
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import seaborn as sns


def cheby_bandpass(X, sfreq, l_freq=1.0, h_freq=145.0, order=6, ripple=0.5):
    """
    X: (trials, channels, samples)
    sfreq: sampling rate
    l_freq/h_freq: band limits
    """
    nyq = sfreq / 2.0
    wp = [l_freq / nyq, h_freq / nyq]

    # 6th-order Chebyshev filter
    b, a = cheby1(N=order, rp=ripple, Wn=wp, btype='bandpass')

    return filtfilt(b, a, X, axis=-1)



def zscore(X_train, X_test):
    """
    X_train: (Ntr, channels, samples)
    X_test:  (Nte, channels, samples)

    Returns Z-normalized data using only train μ, σ
    """

    # mean & var over *ime and trials, per channel
    mu = X_train.mean(axis=(0, -1), keepdims=True)
    var = X_train.var(axis=(0, -1), keepdims=True)

    X_train_norm = (X_train - mu) / np.sqrt(var)
    X_test_norm  = (X_test  - mu) / np.sqrt(var)

    return X_train_norm, X_test_norm




def save_results(day, subject, file_name, plot=None, file_name_text=None, confusion_matrix=None):
    
    # Create the directory path (based on day and subject)
    dir_path = f"./EEGNET_MC/{day}/{subject}/"
    os.makedirs(dir_path, exist_ok=True)

    if plot:
        plot.savefig(f"{dir_path}{file_name}.png", dpi=300)
        plt.close()  
    
    if confusion_matrix is not None:
        np.save(f"{dir_path}{file_name}.npy", confusion_matrix)
    
    elif file_name_text is not None:
        if isinstance(file_name_text, np.ndarray):
            file_name_text = json.dumps(file_name_text.tolist(), indent=4) 
        
        elif isinstance(file_name_text, (list, dict)):
            file_name_text = json.dumps(file_name_text, indent=4)  
        
        with open(f"{dir_path}{file_name}.txt", "w") as f:
            f.write(file_name_text)  
            
            


def _get_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _stack_with_padding(arrs, pad_value=np.nan):
    max_len = max(len(a) for a in arrs)
    out = np.full((len(arrs), max_len), pad_value, dtype=float)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a
    return out

def plot_cv_curves(cv_history, title_prefix=" Best CV", show_mean=True):

    loss_train_col = _get_existing_col(cv_history[0], ["train_loss", "loss"])
    acc_train_col  = _get_existing_col(cv_history[0], ["train_acc", "train_accuracy", "acc"])

    plt.figure()
    train_losses, valid_losses = [], []

    for i, h in enumerate(cv_history, start=1):

        if loss_train_col is not None and loss_train_col in h:
            y = h[loss_train_col].astype(float).values
            train_losses.append(y)

        if "valid_loss" in h:
            yv = h["valid_loss"].astype(float).values
            valid_losses.append(yv)

    if show_mean:
        if len(train_losses) > 0:
            M = _stack_with_padding(train_losses)
            mean_curve = np.nanmean(M, axis=0)
            plt.plot(np.arange(1, len(mean_curve)+1), mean_curve, linewidth=3, label="Mean train")
        if len(valid_losses) > 0:
            M = _stack_with_padding(valid_losses)
            mean_curve = np.nanmean(M, axis=0)
            plt.plot(np.arange(1, len(mean_curve)+1), mean_curve, linewidth=3, label="Mean valid")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} - Loss (Best Config)")
    plt.legend()
    plt.grid(True)
    plt.show()

    #  ACCURACY PLOT
    plt.figure()
    train_accs, valid_accs = [], []

    for i, h in enumerate(cv_history, start=1):

        if acc_train_col is not None and acc_train_col in h:
            ya = h[acc_train_col].astype(float).values
            train_accs.append(ya)

        if "valid_acc" in h:
            yav = h["valid_acc"].astype(float).values
            valid_accs.append(yav)

    if show_mean:
        if len(train_accs) > 0:
            M = _stack_with_padding(train_accs)
            mean_curve = np.nanmean(M, axis=0)
            plt.plot(np.arange(1, len(mean_curve)+1), mean_curve, linewidth=3, label="Mean train")
        if len(valid_accs) > 0:
            M = _stack_with_padding(valid_accs)
            mean_curve = np.nanmean(M, axis=0)
            plt.plot(np.arange(1, len(mean_curve)+1), mean_curve, linewidth=3, label="Mean valid")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} - Accuracy (Best Config)")
    plt.legend()
    plt.grid(True)
    plt.show()





" --- PLOTS ---"


"Read accuracy or weighted accuracy per class"
def read_accuracy_files_from_subfolders(folder_path):
    accuracies = []

    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        
        if os.path.isdir(subfolder_path):
            file_path = os.path.join(subfolder_path, 'per_class_acc.txt')
            if os.path.isfile(file_path):  
                with open(file_path, 'r') as file:
                    subject_accuracies = []
                    for line in file.readlines():
                        try:
                            stripped_line = line.strip().strip(',').strip('[]')
                            if stripped_line:  
                                subject_accuracies.append(float(stripped_line))
                        except ValueError as e:
                            print(f"Error processing line in {file_path}: {e}")
                    if subject_accuracies:
                        accuracies.append(subject_accuracies)


    return np.array(accuracies)


"Accuracy or weighted accuracy per class"
def plot_accuracy(accuracies):
    if accuracies.ndim == 1:
        accuracies = accuracies.reshape(1, -1)

    mean_accuracy_per_class = np.nanmean(accuracies, axis=0)  
    std_accuracy_per_class = np.nanstd(accuracies, axis=0)    
    overall_mean_accuracy = np.nanmean(mean_accuracy_per_class)

    class_labels = [
         'Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo', 'Foxtrot', 'Golf', 
         'Hotel', 'India', 'Juliett', 'Kilo', 'Lima', 'Mike', 
        
        'November','Oscar','Papa','Quebec','Romeo','Sierra',
        'Tango','Uniform','Victor','Whiskey','X-Ray','Yankee','Zulu'] 

    # Create a bar plot 
    plt.figure(figsize=(12, 6))
    bars = plt.bar(np.arange(len(mean_accuracy_per_class)), mean_accuracy_per_class, yerr=std_accuracy_per_class, 
                   color='skyblue', edgecolor='gray', width=0.7, capsize=5)

    plt.axhline(y=overall_mean_accuracy, color='red', linestyle='--', label=f'Overall Mean: {overall_mean_accuracy:.2f}%')

    for bar in bars:
        bar.set_edgecolor('black')
        bar.set_linewidth(1.2)

    plt.ylabel('Accuracy (%)')
    plt.xlabel('Classes')
    plt.title('Classification Accuracy Across all Participants')
    plt.xticks(np.arange(len(mean_accuracy_per_class)), class_labels, rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Usage
folder_path = '/Conformer_LOSO/DAY3' 
accuracies = read_accuracy_files_from_subfolders(folder_path)
plot_accuracy(accuracies * 100)
                 

"Accuracy or weighted accuracy per participant"
def plot_participant_accuracy(accuracies):

    if accuracies.ndim == 1:
        accuracies = accuracies.reshape(1, -1)
    mean_accuracy_per_subject = np.nanmean(accuracies, axis=1)  
    overall_mean_accuracy = np.nanmean(mean_accuracy_per_subject)

    participant_labels = [f'S{i+1}' for i in range(len(mean_accuracy_per_subject))]
    sorted_indices = sorted(range(len(participant_labels)), key=lambda i: int(participant_labels[i][1:]))
    sorted_participant_labels = [participant_labels[i] for i in sorted_indices]
    sorted_accuracies = mean_accuracy_per_subject[sorted_indices]

    # Create a bar plot 
    plt.figure(figsize=(14, 6))
    bars = plt.bar(np.arange(len(sorted_accuracies)), sorted_accuracies, color='skyblue', edgecolor='gray', capsize=5)
    plt.axhline(y=overall_mean_accuracy, color='red', linestyle='--', label=f'Overall Mean: {overall_mean_accuracy:.2f}%')


    for bar in bars:
        bar.set_edgecolor('black')
        bar.set_linewidth(1.2)

    plt.ylabel('Accuracy (%)')
    plt.xlabel('Participant ID')
    plt.title('Classification Accuracy per Subject')
    plt.xticks(np.arange(len(sorted_accuracies)), sorted_participant_labels, rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


folder_path = '/Conformer_LOSO2/DAY3'  
accuracies = read_accuracy_files_from_subfolders(folder_path)
plot_participant_accuracy(accuracies * 100)



"Confusion matrix"
main_folder = '/Conformer_LOSO/DAY3'
confusion_matrices = []

for subject_folder in os.listdir(main_folder):
    subject_folder_path = os.path.join(main_folder, subject_folder)
    
    if os.path.isdir(subject_folder_path):
        confusion_matrix_path = os.path.join(subject_folder_path, 'confusion_matrix.npy')
        
        if os.path.exists(confusion_matrix_path):
            confusion_matrix = np.load(confusion_matrix_path)
            confusion_matrices.append(confusion_matrix)

confusion_matrices = np.array(confusion_matrices)
average_confusion_matrix = np.mean(confusion_matrices, axis=0)

class_labels = [
     'Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo', 'Foxtrot', 'Golf', 
      'Hotel', 'India', 'Juliett', 'Kilo', 'Lima', 'Mike',
      
      'November','Oscar','Papa','Quebec','Romeo','Sierra', 'Tango',
      'Uniform','Victor','Whiskey','X-Ray','Yankee','Zulu']


plt.figure(figsize=(8, 6))
sns.heatmap(average_confusion_matrix,annot=True,fmt='.2f',cmap='Blues',xticklabels=class_labels,yticklabels=class_labels)


plt.title('Average Confusion Matrix')
plt.xlabel('Predicted class')
plt.ylabel('True class')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()  
plt.show()










