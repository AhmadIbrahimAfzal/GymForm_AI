import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv('dataset_fullbody.csv')

labels_map = {'Bad Curl': 0, 'Good Curl': 1, 'Bad Squat': 2, 'Good Squat': 3}
df['label_encoded'] = df['label'].map(labels_map)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df.drop(['label', 'label_encoded'], axis=1).values
y = df['label_encoded'].values

split_idx = int(len(X) * 0.8)
X_train, X_test = torch.FloatTensor(X[:split_idx]), torch.FloatTensor(X[split_idx:])
y_train, y_test = torch.LongTensor(y[:split_idx]), torch.LongTensor(y[split_idx:])

class GymModel(nn.Module):
    def __init__(self):
        super(GymModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )
    def forward(self, x):
        return self.network(x)

model = GymModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

epochs = 150
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    predictions = model(X_test)
    _, predicted_classes = torch.max(predictions, 1)
    correct = (predicted_classes == y_test).sum().item()
    accuracy = correct / len(y_test)

print(f"Accuracy: {accuracy * 100:.2f}%")

torch.save(model.state_dict(), 'gym_model_fullbody.pt')