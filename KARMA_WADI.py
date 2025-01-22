import tensorflow as tf
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import random
import time
from sklearn.model_selection import train_test_split

import joblib
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from scipy.stats import kurtosis, skew
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

import optuna
import torch.nn as nn

from thop import profile

import torch
from torchsummary import summary
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import ks_2samp

import shap

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler

from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

import umap
from scipy.optimize import minimize
from scipy.optimize import nnls

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.metrics import mean_squared_error
import json
import torch.nn.functional as F

from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial import Voronoi, voronoi_plot_2d

from eval_utils import *

from scipy.stats import gaussian_kde

from sklearn.metrics import matthews_corrcoef

torch.manual_seed(42)
tf.config.experimental_run_functions_eagerly(True)

def calc_p2p(predict, actual):
    tp = np.sum(predict * actual)
    tn = np.sum((1-predict) * (1-actual))
    fp = np.sum(predict * (1-actual))
    fn = np.sum((1-predict) * actual)
    
    precision = tp / (tp + fp + 0.000001)
    recall = tp / (tp + fn + 0.000001)
    f1 = 2 * precision * recall / (precision + recall + 0.000001)
    
    return f1, precision, recall, tp, tn, fp, fn

def get_trad_f1(score, label):
    maxx = float(score.max())
    minn = float(score.min())
    
    actual = label > 0.1
    grain = 1000
    max_f1 = 0.0
    max_f1_thres = 0.0
    p = 0
    r = 0
    for i in range(grain):
        thres = (maxx-minn)/grain * i + minn
        predict = score > thres
        f1, precision, recall, tp, tn, fp, fn = calc_p2p(predict, actual)
        if f1 > max_f1:
            max_f1 = f1
            max_f1_thres = thres
            p = precision
            r = recall
            
            
    print("max f1 score is %f and threshold is %f\n" %(max_f1, max_f1_thres))
    return max_f1, max_f1_thres, p, r


# ============== Load Data ==============
Training_WADI_RAW = pd.read_csv("WADI_train.csv")
TEST_WADI_RAW = pd.read_csv("WADI_test.csv")


# ============== Test Dataset ==============
C_TEST_WADI_RAW=TEST_WADI_RAW.drop(['attack'], axis = 1)
WADI_label = TEST_WADI_RAW['attack']

# ============== Training Dataset ==============

WADI_train_label = Training_WADI_RAW['attack']  # train label

C_TRAIN_WADI_RAW = Training_WADI_RAW.drop(['attack'], axis=1)


C_TRAIN_WADI_RAW.fillna(0, inplace=True)

# ============== Original Test Results ==============
MTS_cad_WADI_1 = pd.read_csv("1_WADI_MTS_CAD_prediction_score.csv")
MTAD_gat_2 = pd.read_csv("2_WADI_mtad_gat_prediction_score.csv")
GANF_3 = pd.read_csv("3_WADI_ganf_prediction_score.csv")
ANOMALY_transformer_4 = pd.read_csv("4_WADI_anomaly_transformer_prediction_score.csv")
RANSynCoder_5 = pd.read_csv("5_WADI_RANSyn_prediction_score.csv")
Autoencoder_6 = pd.read_csv("6_WADI_Autoencoder_prediction_score.csv")
USAD_7 = pd.read_csv("7_WADI_USAD_prediction_score.csv")
GDN_8 = pd.read_csv("8_WADI_GDN_w_prediction_scores.csv")
LSTM_9 = pd.read_csv("9_WADI_lstm_prediction_score.csv")
MSCRED_10 =pd.read_csv("10_WADI_mscred_prediction_score.csv")

# ===============================================================================================================================================================

# ============== Rename Columns =================================================================================================================================
MTS_cad_WADI_1 = MTS_cad_WADI_1.rename(columns={'score': 'MTS_score'})
MTAD_gat_2 = MTAD_gat_2.rename(columns={'score': 'MTAD_score'})
GANF_3 = GANF_3.rename(columns={'score': 'GANF_score'})
ANOMALY_transformer_4 = ANOMALY_transformer_4.rename(columns={'score': 'AT_score'})
RANSynCoder_5 = RANSynCoder_5.rename(columns={'score': 'RANSynCoder_score'})
Autoencoder_6 = Autoencoder_6.rename(columns={'score': 'Autoencoder_score'})
USAD_7 = USAD_7.rename(columns={'score': 'USAD_score'})
GDN_8 = GDN_8.rename(columns={'score': 'GDN_score'})
LSTM_9 = LSTM_9.rename(columns={'score': 'LSTM_score'})
MSCRED_10 = MSCRED_10.rename(columns={'score': 'MSCRED_score'})

# ===============================================================================================================================================================


# ============== Training Data Prediction Results ==============
MTS_cad_WADI_1_train = pd.read_csv("1_WADI_MTS_cad_train_prediction_score.csv")
MTAD_gat_2_train = pd.read_csv("2_WADI_MTAD_GAT_train_prediction_score.csv")
GANF_3_train = pd.read_csv("3_WADI_GANF_train_prediction_score.csv")
ANOMALY_transformer_4_train = pd.read_csv("4_WADI_anomaly_transformer_train_prediction_score.csv")
RANSynCoder_5_train = pd.read_csv("5_WADI_RANSYNCoder_train_prediction_score.csv")
Autoencoder_6_train = pd.read_csv("6_WADI_Autoencoder_train_prediction_score.csv")
USAD_7_train = pd.read_csv("7_WADI_USAD_train_prediction_score.csv")
GDN_8_train = pd.read_csv("8_WADI_GDN_train_prediction_scores.csv")
LSTM_9_train = pd.read_csv("9_WADI_LSTM_train_prediction_score.csv")
MSCRED_10_train = pd.read_csv("10_WADI_MScred_train_prediction_score.csv")


# ============== Rename Columns (Training Data) ==============
MTS_cad_WADI_1_train = MTS_cad_WADI_1_train.rename(columns={'score': 'MTS_score'})
MTAD_gat_2_train = MTAD_gat_2_train.rename(columns={'0': 'MTAD_score'})
GANF_3_train = GANF_3_train.rename(columns={'0': 'GANF_score'})
ANOMALY_transformer_4_train = ANOMALY_transformer_4_train.rename(columns={'0': 'AT_score'})
RANSynCoder_5_train = RANSynCoder_5_train.rename(columns={'score': 'RANSynCoder_score'})
Autoencoder_6_train = Autoencoder_6_train.rename(columns={'0': 'Autoencoder_score'})
USAD_7_train = USAD_7_train.rename(columns={'0': 'USAD_score'})
GDN_8_train = GDN_8_train.rename(columns={'score': 'GDN_score'})
LSTM_9_train = LSTM_9_train.rename(columns={'0': 'LSTM_score'})
MSCRED_10_train = MSCRED_10_train.rename(columns={'0': 'MSCRED_score'})


# ============== Use Test Prediction Results ==============
list_WADI_model = [
    MTS_cad_WADI_1['MTS_score'],
    MTAD_gat_2['MTAD_score'],
    GANF_3['GANF_score'],
    ANOMALY_transformer_4['AT_score'],
    RANSynCoder_5['RANSynCoder_score'],
    Autoencoder_6['Autoencoder_score'],
    USAD_7['USAD_score'],
    GDN_8['GDN_score'],
    LSTM_9['LSTM_score'],
    MSCRED_10['MSCRED_score']
]

WADI_anomaly_score_concate = pd.concat(list_WADI_model, axis=1)

# ============== Use Training Prediction Results ==============
list_train_WADI_model = [
    MTS_cad_WADI_1_train['MTS_score'],
    MTAD_gat_2_train['MTAD_score'],
    GANF_3_train['GANF_score'],
    ANOMALY_transformer_4_train['AT_score'],
    RANSynCoder_5_train['RANSynCoder_score'],
    Autoencoder_6_train['Autoencoder_score'],
    USAD_7_train['USAD_score'],
    GDN_8_train['GDN_score'],
    LSTM_9_train['LSTM_score'],
    MSCRED_10_train['MSCRED_score']
]

train_WADI_anomaly_score_concate = pd.concat(list_train_WADI_model, axis=1)


# ============== Data Alignment ==============
C_TEST_WADI_RAW.columns = C_TEST_WADI_RAW.columns.str.strip()
C_TEST_WADI_RAW = C_TEST_WADI_RAW.reindex(columns=C_TEST_WADI_RAW.columns)

# ============== Min-Max Normalization ==============
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(C_TRAIN_WADI_RAW)
X_test_scaled = scaler.transform(C_TEST_WADI_RAW)

df_X_train_scaled = pd.DataFrame(X_train_scaled, columns=C_TEST_WADI_RAW.columns)
df_X_test_scaled = pd.DataFrame(X_test_scaled, columns=C_TEST_WADI_RAW.columns)

# ============== Scaling Training Predictions ==============
scaler_2 = MinMaxScaler()
X_train_prediction_scaled = scaler_2.fit_transform(train_WADI_anomaly_score_concate)
df_X_train_prediction_scaled = pd.DataFrame(X_train_prediction_scaled, columns=train_WADI_anomaly_score_concate.columns)

# ============== Scaling Test Predictions ==============
scaler_3 = MinMaxScaler()
X_test_prediction_scaled = scaler_3.fit_transform(WADI_anomaly_score_concate)
df_X_test_prediction_scaled = pd.DataFrame(X_test_prediction_scaled, columns=WADI_anomaly_score_concate.columns)


#=======================================================Teacher model=========================================================

# Define the RBF Network class
class RBFNetwork(nn.Module):
    def __init__(self, centers, gamma=1.0):
        super(RBFNetwork, self).__init__()
        self.centers = torch.tensor(centers, dtype=torch.float32)
        self.gamma = gamma
        self.linear = nn.Linear(self.centers.shape[0], 1)  # Output layer
    
    def forward(self, x):
        rbf_outputs = []
        for i in range(self.centers.shape[0]):
            mse_distance = torch.mean((x - self.centers[i]) ** 2, dim=1)
            rbf = torch.exp(-self.gamma * mse_distance)  # Gaussian RBF
            rbf_outputs.append(rbf)
        rbf_outputs = torch.stack(rbf_outputs, dim=1)
        output = self.linear(rbf_outputs)
        return output

concate_X_train = pd.concat((df_X_train_scaled, df_X_train_prediction_scaled), axis = 1)
concate_X_test = pd.concat((df_X_test_scaled, df_X_test_prediction_scaled), axis = 1)

X_train_tensor = torch.tensor(concate_X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(WADI_train_label.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(concate_X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(WADI_label.values, dtype=torch.float32).view(-1, 1)


student_train_tensor = torch.tensor(df_X_train_scaled.values, dtype=torch.float32)
student_test_tensor = torch.tensor(df_X_test_scaled.values, dtype=torch.float32)



# Extract the parameters
gamma = 4.7665163962213235 
lr = 0.001077045874701812 
n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_train_tensor)#X_train_scaled)
centroids = kmeans.cluster_centers_

# Step 2: Initialize RBF network with loaded 'gamma'
model = RBFNetwork(centroids, gamma)

# Step 3: Define loss and optimizer with loaded 'lr'
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
'''
# Step 4: Train the RBF network
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
'''

model = torch.load('WADI_teacher_model.pth')

# Step 5: Evaluate the model on the test set
model.eval()

with torch.no_grad():
    y_pred_train = model(X_train_tensor)
    
    mse_train = mean_squared_error(y_train_tensor.numpy(), y_pred_train.numpy())  
    
    std = np.std(y_train_tensor.numpy() - y_pred_train.numpy())
    
     
T_threshold = mse_train  # Use train MSE as threshold

with torch.no_grad():
    y_pred = model(X_test_tensor)
    mse_test_values = (y_test_tensor.numpy() - y_pred.numpy()) ** 2
    y_pred_labels = (mse_test_values > T_threshold + 3 * std).astype(int)

# Step 5: Calculate the F1 Score
y_test_np = y_test_tensor.numpy().astype(int)

Teacher_f1 = f1_score(y_test_np, y_pred_labels)
Teacher_precision = precision_score(y_test_np, y_pred_labels)
Teacher_recall  = recall_score(y_test_np, y_pred_labels)

print("Best F1 Score in SWaT teacher model: ", Teacher_f1, Teacher_precision, Teacher_recall)


input = X_train_tensor[0].unsqueeze(0)

flops, params = profile(model, inputs=(input,))
print(f"Total FLOPs: {flops}")
print(f"Total Parameters: {params}")


#=================================================================================================================================================

class RBF(nn.Module):
    def __init__(self, centroids, gamma, centroid_dim, input_dim):
        super(RBF, self).__init__()
        
       
        self.centroids = nn.Parameter(torch.tensor(centroids, dtype=torch.float32))
        
        
        self.projection_layer = nn.Linear(centroid_dim, input_dim)
        
        
        self.gamma = gamma

    def forward(self, x):
        
        centroids_projected = self.projection_layer(self.centroids)  
        
       
        diff = x.unsqueeze(1) - centroids_projected.unsqueeze(0)  
        
       
        mse = (diff ** 2).mean(dim=2)  
        
       
        return torch.exp(-self.gamma * mse)


class StudentRBFNetwork(nn.Module):
    def __init__(self, centroids, gamma, input_features, centroid_dim):
        super(StudentRBFNetwork, self).__init__()
        
        
        self.rbf_layer = RBF(centroids, gamma, centroid_dim, input_features)
        
        
        self.linear_layer = nn.Linear(centroids.shape[0], 1)

    def forward(self, x):
       
        rbf_output = self.rbf_layer(x)
        output = self.linear_layer(rbf_output)
        return output

# Initialize student model with same centroids and gamma, but fewer input features
input_features = student_train_tensor.shape[1] 
centroid_dim = X_train_tensor.shape[1] 

student_model = StudentRBFNetwork(centroids, gamma, input_features, centroid_dim)


# Parameters for distillation
s_epochs=100#50

alpha = 0
temperature = 1

student_optimizer = optim.Adam(student_model.parameters(), lr=0.001077045874701812)#0.0003898678505538315 #0.001
distillation_loss_fn = nn.MSELoss()
ground_truth_loss_fn = nn.MSELoss()
'''
# Function to save the student model and optimizer
def save_model(student_model, optimizer, epoch, file_path="WADI_student_model.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': student_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, file_path)
    print(f"Model saved at epoch {epoch}")

for epoch in range(s_epochs):
    student_model.train()
    
    with tqdm(total=len(X_train_tensor), desc=f"Epoch {epoch + 1}/{s_epochs}", unit="sample") as progress_bar:
       
       # Teacher predictions (soft targets)
       with torch.no_grad():
           teacher_outputs = model(X_train_tensor) / temperature
       
       # Student predictions
       student_outputs = student_model(student_train_tensor)
       
       # Compute losses
       distillation_loss = distillation_loss_fn(student_outputs, teacher_outputs) * (1 - alpha)
       
       total_loss = distillation_loss
       
       # Backpropagation and optimization
       student_optimizer.zero_grad()
       total_loss.backward()
       student_optimizer.step()
       
       if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{s_epochs}], Loss: {total_loss.item():.10f}")
            # Save the model every 10 epochs
            save_model(student_model, student_optimizer, epoch + 1)

       progress_bar.update(len(X_train_tensor))
'''
def load_model(student_model, optimizer, file_path="WADI_student_model.pth"):
    checkpoint = torch.load(file_path)
    student_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    
    return start_epoch

# Example of loading the model before training
start_epoch = load_model(student_model, student_optimizer)


# Evaluate student model
student_model.eval()

with torch.no_grad():
    student_pred_test = student_model(student_test_tensor)
    
    mse_student_test_values = (y_test_tensor.numpy() - student_pred_test.numpy()) ** 2
    
    #mse_student_test_values = mse_student_test_values.mean(axis=1)
    student_pred_labels = (mse_student_test_values > T_threshold + 3 * std).astype(int) #0.43).astype(int)


f1_student = f1_score(y_test_np, student_pred_labels)
precision_student  = precision_score(y_test_np, student_pred_labels)
recall_student   = recall_score(y_test_np, student_pred_labels)


print("F1 Score for WADI Student Model: ", f1_student, precision_student, recall_student)

input = student_train_tensor[0].unsqueeze(0)

student_flops, student_params = profile(student_model, inputs=(input,))
print(f"Total FLOPs: {student_flops}")
print(f"Total Parameters: {student_params}")