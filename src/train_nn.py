import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
from model import TextNN

# Load your features CSV
df = pd.read_csv("/Users/susmitapaudel/projects/ai-text-detector/data/raw/features_output.csv")

# Separate features and labels
X = df.drop(columns=['text', 'generated']).values  # all columns except text/label
y = df['generated'].values                        # label column

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for later
joblib.dump(scaler, 'scaler.pkl')

# PyTorch dataset
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = FeatureDataset(X_scaled, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model
input_dim = X_scaled.shape[1]
model = TextNN(input_dim)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(20):
    total_loss = 0
    for features, labels in loader:
        labels = labels.unsqueeze(1)  # shape [batch,1]
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features.size(0)
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader.dataset):.4f}")

# Save trained model
torch.save(model.state_dict(), 'text_feature_nn.pth')
print("✓ Model trained and saved as 'text_feature_nn.pth'")
