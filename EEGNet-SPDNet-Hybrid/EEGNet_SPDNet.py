# Temporal filters

import torch
import torch.nn as nn
import torch.nn.functional as F
from spdnet.spd import SPDTransform, SPDTangentSpace, SPDRectified
from spdnet.optimizer import StiefelMetaOptimizer
import matplotlib.pyplot as plt
plt.use("Agg")
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim import Adam, SGD
from torch import nn
import numpy as np
import argparse
import time
from scipy import signal
from sklearn.model_selection import train_test_split
import keras
from torch.utils.data import Dataset, TensorDataset, DataLoader

class TemporalFilt(nn.Module):
  def __init__(self, F1 = 4, sample_freq = 1000, num_channels = 60):
    super(TemporalFilt, self).__init__()
    self.conv2d = nn.Conv2d(1, F1, (1, sample_freq//2), padding= "same")
    self.batchnorm = nn.BatchNorm2d(F1, False)
    self.pooling = nn.AvgPool2d(kernel_size = (1, 4))
    self.dropout = nn.Dropout(p = 0.5)

  def forward(self, x):
    x = torch.unsqueeze(x, 1)
    x = self.conv2d(x)
    x = self.batchnorm(x)
    x = F.elu(x)
    x = self.pooling(x)
    #x = self.dropout(x)
    return x

# SPDNet with 1 SPD transformation

class SPDNet(nn.Module):
  def __init__(self, num_classes = 11, num_channels = 60, input_chns = 4, output_size = 20):
    super(SPDNet, self).__init__()
    self.trans = SPDTransform(num_channels, output_size)
    self.rect  = SPDRectified()
    self.tangent = SPDTangentSpace(output_size)
    self.linear = nn.Linear(int((output_size*(output_size + 1)) / 2) * input_chns, num_classes, bias=True)
        # self.dropout = nn.Dropout(p=0.5)

  def forward(self, x):
    size = x.size()
    x = torch.reshape(x,(-1, size[-2], size[-1]))
    x = self.trans(x)
    x = self.rect(x)
    x = self.tangent(x)
        # x = self.dropout(x)
    x = torch.reshape(x, (size[0], size[1], -1))
    x = torch.reshape(x, (size[0], -1))
    x = self.linear(x)
    return x

class SPDNet_2(nn.Module):
  def __init__(self, num_classes = 11, num_channels = 60, input_chns = 4, output_size = 20):
    super(SPDNet_2, self).__init__()
    self.trans_1 = SPDTransform(num_channels, output_size)
    self.trans_2 = SPDTransform(num_channels, output_size)
    self.trans_3 = SPDTransform(num_channels, output_size)
    self.trans_4 = SPDTransform(num_channels, output_size)
    self.rect_1  = SPDRectified()
    self.rect_2  = SPDRectified()
    self.rect_3  = SPDRectified()
    self.rect_4  = SPDRectified()
    self.tangent_1 = SPDTangentSpace(output_size)
    self.tangent_2 = SPDTangentSpace(output_size)
    self.tangent_3 = SPDTangentSpace(output_size)
    self.tangent_4 = SPDTangentSpace(output_size)
    self.linear = nn.Linear(int((output_size*(output_size + 1)) / 2) * input_chns, num_classes, bias=True)
        # self.dropout = nn.Dropout(p=0.5)

  def forward(self, x):
    size = x.size()
    x_1 = self.trans_1(x[:,0,:,:])
    x_1 = self.rect_1(x_1)
    x_1 = self.tangent_1(x_1)
    x_2 = self.trans_1(x[:,1,:,:])
    x_2 = self.rect_1(x_2)
    x_2 = self.tangent_1(x_2)
    x_3 = self.trans_1(x[:,2,:,:])
    x_3 = self.rect_1(x_3)
    x_3 = self.tangent_1(x_3)
    x_4 = self.trans_1(x[:,3,:,:])
    x_4 = self.rect_1(x_4)
    x_4 = self.tangent_1(x_4)
        # x = self.dropout(x)
    x = torch.cat((x_1, x_2, x_3, x_4),1)
    x = self.linear(x)
    return x

# Converts signals to covariance matrices

class Covariance(nn.Module):
  def __init__(self):
    super(Covariance, self).__init__()
  def forward(self, x):
    D = x.shape[-1]
    mean = torch.mean(x, dim=-1).unsqueeze(-1)
    x = x - mean
    return 1/(D-1) * x @ x.transpose(-1, -2)

# NewNet: EEGNet-SPDNet

class NewNet(nn.Module):
  def __init__(self, num_classes = 11, num_channels = 60, sample_freq = 1000, output_size = 20):
    super(NewNet, self).__init__()
    self.model_1 = TemporalFilt(F1 = 4, sample_freq = sample_freq, num_channels = num_channels)
    self.cov = Covariance()
    self.model_2 = SPDNet(num_classes = num_classes, num_channels = num_channels, output_size = output_size)
  def forward(self, x):
    x = self.model_1(x)
    x = torch.squeeze(x)
    x = self.cov(x)
    x = self.model_2(x)
    return x

# Train function

def train(model = None, loss = None, opt_1 = None, opt_2 = None, dataloader = None, epochs = 1, device = None):
  for e in range(0, epochs):
    if (e+1)%100 == 0:
      print("Epoch : ", e, "\n")
    model.train()
    for (x, y) in dataloader:
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = loss(pred, y)
        opt_1.zero_grad()
        opt_2.zero_grad()
        loss.backward()
        opt_1.step()
        opt_2.step()
  return None

# Evaluation function

def val(model = None, dataloader = None, device = None):
  with torch.no_grad():
    model.eval()
    preds = []
    # loop over the test set
    valCorrect = 0
    for (x, y) in dataloader:
        # send the input to the device
        x = x.to(device)
        y = y.to(device)
        # make the predictions and add them to the list
        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())
        valCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
    valCorrect = valCorrect / len(dataloader.dataset)
    return preds, valCorrect

def filepath_to_labels(file_path, labels_dict):
    with open(file_path) as file:
        line_list = file.readlines()
        line_list = [item.rstrip() for item in line_list]
    labels = np.zeros(len(line_list))
    for i in range(len(line_list)):
        labels[i] = labels_dict[line_list[i]]
    labels = np.int_(labels)
    return labels


# Initialize hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 20
EPOCHS = 500
# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train and evaluate the model for each subject

subjects = ['09','10','11','12','14','15', '16', '18', '19', '20', '21']

labels_dict = {
    '/iy/' : 0,
    '/uw/' : 1,
    '/piy/' : 2,
    '/tiy/' : 3,
    '/diy/' : 4,
    '/m/' : 5,
    '/n/' : 6,
    'pat' : 7,
    'pot' : 8,
    'knew' : 9,
    'gnaw' : 10
  }

accs_subjects = []
y_all = np.zeros((11, 33))

for i,s in enumerate(subjects):

  print("Subject :", s, "\n")

  # Prepare data
  y = filepath_to_labels('drive/My Drive/subject_' + s + '/labels_' + s + '.txt', labels_dict)
  data = np.load('drive/My Drive/subject_' + s + '/cleaned_data_' + s + '.npy')
  x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.25, random_state=42, stratify = y)
  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 1/9, random_state = 89, stratify = y_train)

  x_train = torch.Tensor(x_train)
  y_train = torch.Tensor(y_train).type(torch.LongTensor)
  x_val = torch.Tensor(x_val)
  y_val = torch.Tensor(y_val).type(torch.LongTensor)
  x_test = torch.Tensor(x_test)
  y_test = torch.Tensor(y_test).type(torch.LongTensor)

  train_dataset = TensorDataset(x_train, y_train)
  val_dataset = TensorDataset(x_val, y_val)
  test_dataset = TensorDataset(x_test, y_test)

  numTrainSamples = int(len(train_dataset) )
  numValSamples = int(len(val_dataset))

  trainDataLoader = DataLoader(train_dataset, shuffle=True,
	batch_size=BATCH_SIZE)
  valDataLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
  testDataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
  # calculate steps per epoch for training and validation set
  trainSteps = (len(trainDataLoader.dataset) // BATCH_SIZE) + 1
  valSteps = (len(valDataLoader.dataset) // BATCH_SIZE) + 1

  # Initialize model, optimizers and loss function
  model = None
  model = NewNet(output_size = 20)
  model.cuda()

  opt_1 = Adam(model.model_1.parameters(), lr=INIT_LR)
  opt_2 = Adam(model.model_2.parameters(), lr=INIT_LR)
  opt_2 = StiefelMetaOptimizer(opt_2)
  lossFn = nn.CrossEntropyLoss()

  # Train model
  train(model, lossFn, opt_1, opt_2, trainDataLoader, epochs = EPOCHS, device = device)

  # Evaluate model
  preds, acc = val(model, testDataLoader, device = device)

  y_all[i,:] = np.array(preds)
  accs_subjects.append(acc)

  np.save("./drive/My Drive/y_preds.npy", y_all)
  np.save("./drive/My Drive/ko_accs.npy", accs_subjects)
  print("Subject :", s, "\n", "Accuracy : ", acc, "\n")


# Print accuracies for each subject and mean accuracy
print("ACCURACIES :", accs_subjects)
avg = sum(accs_subjects) / 11
print("Average :", avg)