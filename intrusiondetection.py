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
from scapy.all import sniff, IP, TCP, UDP
import joblib
import smtplib
from email.message import EmailMessage 

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

# Encoding categorical data to numeric values
label_encoder = LabelEncoder()
data['protocol_type'] = label_encoder.fit_transform(data['protocol_type'])
data['service'] = label_encoder.fit_transform(data['service'])
data['flag'] = label_encoder.fit_transform(data['flag'])

# Normalize the training data
scaler = StandardScaler()
features = data.iloc[:, :-1]
normalized_features = scaler.fit_transform(features)

# Encode target labels
target = label_encoder.fit_transform(data['label'])

# Function to send email alerts
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Mailtrap SMTP credentials
SMTP_SERVER = "sandbox.smtp.mailtrap.io"
PORT = 2525
USERNAME = ""  # Replace with your actual Mailtrap username
PASSWORD = ""  # Replace with your actual Mailtrap password

def send_email(subject, message):
    sender_email = "your_email@example.com"
    recipient_email = ""  # Your Mailtrap Inbox Email

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject

    msg.attach(MIMEText(message, "plain"))

    try:
        with smtplib.SMTP(SMTP_SERVER, PORT) as server:
            server.starttls()  # Secure the connection
            server.login(USERNAME, PASSWORD)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error: {e}")
        
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(normalized_features, target, test_size=0.2, random_state=42)

# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Evaluate the model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"KNN Model Accuracy: {accuracy:.4f}")

# Save the trained KNN model
joblib.dump(knn, "knn_model.pkl")
print("KNN model saved as 'knn_model.pkl'")

# Check class distribution
print("Class Distribution in Dataset:")
print(pd.Series(target).value_counts(normalize=True))

# Apply SMOTE if needed
from imblearn.over_sampling import SMOTE

# Adjust k_neighbors based on the smallest class size
smote = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42) # Reduce from default (5) to 2
X_train, y_train = smote.fit_resample(X_train, y_train)

# Define MLP Model
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize MLP Model
input_size = X_train.shape[1]
hidden_size = 64
output_size = len(np.unique(y_train))

mlp_model = MLPModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

# Convert Data to PyTorch Tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).long()

# Train MLP Model
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = mlp_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Save model correctly
torch.save(mlp_model.state_dict(), "mlp_model.pth")
print("MLP model saved as 'mlp_model.pth'")

# Capture Live Network Packets and Extract Features
def extract_features(packet):
    if IP in packet:
        return {
            'duration': 1,
            'protocol_type': 1 if TCP in packet else (2 if UDP in packet else 0),
            'src_bytes': len(packet[TCP].payload) if TCP in packet else 0,
            'dst_bytes': len(packet[UDP].payload) if UDP in packet else 0,
            'flag': 1,
            'count': 1,
            'srv_count': 1
        }
    return None

print("Capturing live network packets...")
packets = sniff(count=50, filter="ip")
data_live = [extract_features(pkt) for pkt in packets if extract_features(pkt) is not None]
df_live = pd.DataFrame(data_live)
df_live.fillna(0, inplace=True)

# Ensure all training features exist in live data
missing_features = set(features.columns) - set(df_live.columns)
for feature in missing_features:
    df_live[feature] = 0

df_live = df_live[features.columns]
df_live_scaled = scaler.transform(df_live)
X_live = torch.from_numpy(df_live_scaled).float()
print("Preprocessed Live Data Ready:", X_live.shape)

# Load Models and Predict Intrusions
print("Loading trained models...")
knn = joblib.load("knn_model.pkl")
mlp_model = MLPModel(input_size, hidden_size, output_size)
mlp_model.load_state_dict(torch.load("mlp_model.pth", weights_only=True))  # Fix warning
mlp_model.eval()

# Predict Using KNN
y_pred_knn = knn.predict(df_live_scaled)
print("Unique KNN Predictions:", np.unique(y_pred_knn))

# Predict Using MLP
with torch.no_grad():
    y_pred_mlp = mlp_model(X_live)
    y_pred_mlp = torch.argmax(y_pred_mlp, dim=1).numpy()
print("Unique MLP Predictions:", np.unique(y_pred_mlp))

# Check for Intrusions & Alert
if any(y_pred_knn) or any(y_pred_mlp):
    print("INTRUSION DETECTED!")
    email_subject = "Security Alert: Intrusion Detected!"
    email_message = "An intrusion has been detected in the network. Please take immediate action!"
else:
    print("No Intrusion Detected")
    email_subject = "Security Alert: No Intrusion Detected"
    email_message = "No intrusion detected in the network. The system is secure."
    
    # Send email after intrusion check
send_email(email_subject, email_message)
