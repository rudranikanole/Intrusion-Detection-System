import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset and specify column names
column_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
                "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
                "num_compromised", "root_shell", "su_attempted", "num_root",
                "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
                "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
                "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
                "dst_host_srv_rerror_rate", "label"]

data = pd.read_csv("data/kddcup.data_10_percent", header=None, names=column_names)

# Encoding symbolic features (categorical data) to numeric values
label_encoder = LabelEncoder()

# Label encode 'protocol_type', 'service', and 'flag' categorical columns
data['protocol_type'] = label_encoder.fit_transform(data['protocol_type'])
data['service'] = label_encoder.fit_transform(data['service'])
data['flag'] = label_encoder.fit_transform(data['flag'])

# Handle any potential missing values by dropping them
data = data.dropna()

# Normalize the data using Z-score normalization for all features except the target column 'label'
scaler = StandardScaler()
features = data.iloc[:, :-1]  # All features except the last column (which is the 'label')
normalized_features = scaler.fit_transform(features)

# Encode the target labels
target = label_encoder.fit_transform(data['label'])

# Determine the correct number of classes
num_classes = len(np.unique(target))
print(f"Number of unique classes: {num_classes}")

# Convert to PyTorch tensors
X = torch.tensor(normalized_features, dtype=torch.float32)
y = torch.tensor(target, dtype=torch.long)  # Correctly assign target labels as the tensor

# Create a DataLoader
dataset = TensorDataset(X, X)  # Autoencoder targets are the same as inputs
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(X.shape[1], 80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 10)  # Bottleneck layer
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, 80),
            nn.ReLU(),
            nn.Linear(80, X.shape[1]),
            nn.Sigmoid()  # Reconstruction output
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize the autoencoder model, define the loss function and optimizer
autoencoder = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Train the autoencoder
num_epochs = 10  # Set number of epochs to 10
for epoch in range(num_epochs):
    for data in dataloader:
        inputs, _ = data
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Extract the reduced features from the bottleneck layer
with torch.no_grad():
    reduced_features = autoencoder.encoder(X).numpy()

# Check the shape of the reduced features
print("Reduced Features Shape:", reduced_features.shape)

from sklearn.model_selection import train_test_split

# Assuming `reduced_features` contains your feature set and `target` contains labels
X_train, X_test, y_train, y_test = train_test_split(reduced_features, target, test_size=0.2, random_state=42)

# Define the MLP model using PyTorch
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(10, 20)  # Input layer (10 reduced features from Autoencoder)
        self.layer2 = nn.Linear(20, 15)
        self.layer3 = nn.Linear(15, 20)
        self.output_layer = nn.Linear(20, num_classes)  # Output layer (Adjusted to the correct number of classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.softmax(self.output_layer(x))
        return x

# Initialize the MLP model 
mlp_model = MLP()

# Define the loss function and optimizer for MLP
mlp_criterion = nn.CrossEntropyLoss()
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

# Train the MLP model
num_epochs = 10  # Set number of epochs to 10
batch_size = 256

# Convert the target to PyTorch tensor
target_tensor = torch.tensor(target, dtype=torch.long)

# Create a DataLoader for the reduced features
mlp_dataset = TensorDataset(torch.tensor(reduced_features, dtype=torch.float32), target_tensor)
mlp_dataloader = DataLoader(mlp_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for features, labels in mlp_dataloader:
        # Zero the gradients
        mlp_optimizer.zero_grad()

        # Forward pass
        outputs = mlp_model(features)
        loss = mlp_criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        mlp_optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    
mlp_model.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient calculation
    y_pred_mlp = mlp_model(torch.tensor(X_test, dtype=torch.float32))
    y_pred_mlp = torch.argmax(y_pred_mlp, dim=1).numpy()  # Convert probabilities to class labels

# K-Nearest Neighbors (K-NN) using Scikit-learn
X_train, X_test, y_train, y_test = train_test_split(reduced_features, target, test_size=0.2)

# Define and train the K-NN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)  # Make predictions using K-NN

# Make predictions
y_pred = knn.predict(X_test)

# Calculate accuracy
knn_accuracy = accuracy_score(y_test, y_pred)
print(f"K-NN Accuracy: {knn_accuracy}")

# Now, we will define a CNN model for classification

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Define a simple CNN architecture
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # 1D convolution
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)  # Max pooling layer
        self.fc1 = nn.Linear(64 * 2, 128)  # Fully connected layer (adjust size based on dynamic computation)
        self.fc2 = nn.Linear(128, num_classes)  # Output layer (num_classes as output)

    def forward(self, x):
      
        x = self.pool(F.relu(self.conv1(x)))  # First convolutional layer + ReLU + Pooling
       
        x = self.pool(F.relu(self.conv2(x)))  # Second convolutional layer + ReLU + Pooling
       
        x = x.view(x.size(0), -1)  # Flatten dynamically
      
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = self.fc2(x)  # Output layer
        return x

# Initialize CNN model
cnn_model = CNN()

# Define the loss function and optimizer for CNN
cnn_criterion = nn.CrossEntropyLoss()
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Reshape the input data for CNN (1D convolution expects a 3D input)
X_cnn = reduced_features.reshape(-1, 1, reduced_features.shape[1])

# Convert the reshaped features to PyTorch tensors
X_cnn_tensor = torch.tensor(X_cnn, dtype=torch.float32)
target_tensor = torch.tensor(target, dtype=torch.long)

# Create DataLoader for CNN
cnn_dataset = TensorDataset(X_cnn_tensor, target_tensor)
cnn_dataloader = DataLoader(cnn_dataset, batch_size=256, shuffle=True)

# Train the CNN model
num_epochs = 10  # Set number of epochs to 10
for epoch in range(num_epochs):
    for features, labels in cnn_dataloader:
        # Zero the gradients
        cnn_optimizer.zero_grad()

        # Forward pass
        outputs = cnn_model(features)
        loss = cnn_criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        cnn_optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

import torch

device = torch.device("cpu")  # Ensure CPU usage

cnn_model.to(device)  # Move model to CPU
cnn_model.eval()  # Set model to evaluation mode

with torch.no_grad():  # Disable gradient computation
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)  # Convert and move to CPU

    # Ensure batch dimension
    if len(X_test.shape) == 2:  # If shape is (samples, features)
        X_test = X_test.unsqueeze(1)  # Add a channel dimension → (samples, 1, features)

    y_pred_cnn = cnn_model(X_test)  # Make predictions

    # ✅ Convert probabilities to class labels
    y_pred_cnn = torch.argmax(y_pred_cnn, dim=1).cpu().numpy()  # Get predicted class indices


# After training the MLP and CNN models, we can evaluate their performances on the test data

# Evaluate MLP
mlp_model.eval()
with torch.no_grad():
    mlp_outputs = mlp_model(torch.tensor(reduced_features, dtype=torch.float32))
    _, predicted_mlp = torch.max(mlp_outputs, 1)
    mlp_accuracy = accuracy_score(target, predicted_mlp.numpy())
    print(f"MLP Accuracy: {mlp_accuracy:.4f}")

# Evaluate CNN
cnn_model.eval()
with torch.no_grad():
    cnn_outputs = cnn_model(X_cnn_tensor)
    _, predicted_cnn = torch.max(cnn_outputs, 1)
    cnn_accuracy = accuracy_score(target, predicted_cnn.numpy())
    print(f"CNN Accuracy: {cnn_accuracy:.4f}")

# Assuming `y_test` is your true labels and `y_pred_knn`, `y_pred_mlp`, `y_pred_cnn` are your model predictions.

def print_metrics(y_test, y_pred, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Check for intrusion detection
    if any(y_pred):  # If any prediction indicates an attack
        print(f"INTRUSION DETECTED by {model_name}")
    else:
        print(f"INTRUSION NOT DETECTED by {model_name}")

# Print metrics and intrusion detection for each model
print_metrics(y_test, y_pred_knn, "K-NN")
print_metrics(y_test, y_pred_mlp, "MLP")
print_metrics(y_test, y_pred_cnn, "CNN")
