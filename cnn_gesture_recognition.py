import os
import json
import torch
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ------------------- CONFIGURAÇÕES GERAIS -------------------
GESTURE_DATASET_PATH = r"C:\Users\Aline\Desktop\gestures_dataset"

NUM_EPOCHS = 10
BATCH_SIZE = 2
LEARNING_RATE = 0.001
MAX_SEQ_LEN = 5  # reduzido para teste mais rápido
CNN_OUTPUT_SIZE = 32 * 56 * 56
HIDDEN_SIZE = 128

# ------------------- DATASET PERSONALIZADO -------------------
class GestureFrameDataset(Dataset):
    def __init__(self, data_list, class_map, max_len=30):
        self.data = data_list
        self.class_map = class_map
        self.max_len = max_len
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence_path, class_name = self.data[idx]
        label = self.class_map[class_name]

        frames = sorted([
            os.path.join(sequence_path, f) for f in os.listdir(sequence_path)
            if f.endswith(".jpg") and "roi" in f
        ])

        frame_tensors = []
        for f in frames[:self.max_len]:
            img = Image.open(f).convert("RGB")
            tensor = self.transform(img)
            frame_tensors.append(tensor)

        while len(frame_tensors) < self.max_len:
            frame_tensors.append(torch.zeros_like(frame_tensors[0]))

        tensor_seq = torch.stack(frame_tensors)  # [seq_len, 3, 224, 224]
        return tensor_seq, torch.tensor(label)

# ------------------- MODELO CNN + LSTM -------------------
class CNNLSTMModel(nn.Module):
    def __init__(self, cnn_output_size, hidden_size, num_classes):
        super(CNNLSTMModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        cnn_out_seq = []
        for t in range(seq_len):
            cnn_out = self.cnn(x[:, t])
            cnn_out = cnn_out.view(batch_size, -1)
            cnn_out_seq.append(cnn_out)
        cnn_out_seq = torch.stack(cnn_out_seq, dim=1)
        lstm_out, _ = self.lstm(cnn_out_seq)
        out = self.fc(lstm_out[:, -1])
        return out

# ------------------- PREPARAÇÃO DOS DADOS -------------------
print("Preparando os dados a partir dos frames...")

class_names = sorted([d for d in os.listdir(GESTURE_DATASET_PATH) if os.path.isdir(os.path.join(GESTURE_DATASET_PATH, d))])
class_map = {name: i for i, name in enumerate(class_names)}
NUM_CLASSES = len(class_names)

with open("class_map.json", "w") as f:
    json.dump(class_map, f)

all_sequences = []
for class_name in class_names:
    class_path = os.path.join(GESTURE_DATASET_PATH, class_name)
    for seq_name in os.listdir(class_path):
        seq_path = os.path.join(class_path, seq_name)
        if os.path.isdir(seq_path):
            all_sequences.append((seq_path, class_name))

train_files, val_files = train_test_split(
    all_sequences, test_size=0.2, random_state=42, stratify=[item[1] for item in all_sequences]
)

train_dataset = GestureFrameDataset(train_files, class_map, max_len=MAX_SEQ_LEN)
val_dataset = GestureFrameDataset(val_files, class_map, max_len=MAX_SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Classes encontradas: {class_map}")
print(f"Treino: {len(train_dataset)} | Validação: {len(val_dataset)}")

# ------------------- INICIALIZAÇÃO DO MODELO -------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

model = CNNLSTMModel(CNN_OUTPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ------------------- TREINAMENTO COM MÉTRICAS -------------------
best_val_loss = float('inf')
epochs_no_improve = 0
patience = 10

history = {
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': [],
    'val_precision': [],
    'val_recall': [],
    'val_f1': []
}

print("\nIniciando o treinamento com validação e métricas...\n")

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    for i, (sequences, labels) in enumerate(train_loader):
        print(f"Treinando batch {i+1}/{len(train_loader)}")  # Depuração
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * sequences.size(0)

    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * sequences.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    epoch_train_loss = train_loss / len(train_dataset)
    epoch_val_loss = val_loss / len(val_dataset)

    history['epoch'].append(epoch + 1)
    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)
    history['val_accuracy'].append(acc)
    history['val_precision'].append(prec)
    history['val_recall'].append(rec)
    history['val_f1'].append(f1)

    print(f"Época {epoch+1}/{NUM_EPOCHS} | "
          f"Loss Treino: {epoch_train_loss:.4f} | "
          f"Loss Val: {epoch_val_loss:.4f} | "
          f"Acurácia: {acc:.4f} | Precisão: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "best_libras_model.pth")
        epochs_no_improve = 0
        print(f"Novo melhor modelo salvo com val_loss = {best_val_loss:.4f}")
    else:
        epochs_no_improve += 1

    if epochs_no_improve == patience:
        print(f"Early stopping: sem melhora por {patience} épocas.")
        break

# ------------------- SALVAMENTO FINAL -------------------
torch.save(model.state_dict(), "cnn_lstm_model.pth")
print("Modelo final salvo como 'cnn_lstm_model.pth'")

df = pd.DataFrame(history)
df.to_excel("training_metrics.xlsx", index=False)
print("Métricas salvas em 'training_metrics.xlsx'")
