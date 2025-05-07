import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from preprocess_data import *
from model import SessionModel
from sklearn.model_selection import train_test_split

df = get_df()
feature_mapping = get_feature_mapping(df, features)
out_mapping = get_feature_mapping(df, out)
out_mapping = out_mapping['event_type']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SessionModel(feature_sizes=[len(feature_mapping[f]) for f in features], out_size=len(out_mapping)).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params / 1e6:.2f}M")

X = df[features]
X[features] = X[features].astype(str)
X = X.apply(lambda x: get_mapping(x, feature_mapping), axis=1)
y = df['event_type'].astype(str)
y = y.map(lambda x: out_mapping[x])
X = X.to_numpy(np.float32)
y = y.to_numpy(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)

X_train = torch.from_numpy(X_train.astype(np.float32)).to(device)
y_train = torch.from_numpy(y_train.astype(np.float32)).to(device)
X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)
y_test = torch.from_numpy(y_test.astype(np.float32)).to(device)

batch_size = 256
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1000

for epoch in range(num_epochs):
    model.train()
    for i, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save({'checkpoint': model.state_dict(), 
                'feature_mapping': feature_mapping, 
                'out_mapping': out_mapping}, 'model.pth')

# Validate
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test set: {100 * correct / total}%')