# ===================================================================
# SCRIPT DE TREINAMENTO OFICIAL COM VALIDAÇÃO
# TCC - Aline e Gabi
#
# Este script treina o modelo separando os dados em treino e validação
# para avaliar a real capacidade de generalização do modelo.
# ===================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from glob import glob
import json
# --- DESCOMENTADO PARA USAR A VALIDAÇÃO ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------- 1. CONFIGURAÇÕES -------------------
# Caminho para a pasta com os arquivos .npy que geramos
LANDMARKS_PATH = r"G:\Meu Drive\TCC - Aline e Gabi\TCC_Dataset_Landmarks"

# Parâmetros de Treinamento
NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001
MAX_LEN = 150

# Parâmetros do Modelo
INPUT_SIZE = 1662
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.4

# ------------------- 2. CLASSES DE DATASET E MODELO -------------------
class LandmarkDataset(Dataset):
    def __init__(self, data_list, class_map, max_len=100):
        self.data_list = data_list
        self.class_map = class_map
        self.max_len = max_len
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        npy_path, gesture_name = self.data_list[idx]
        landmarks = np.load(npy_path)
        label = self.class_map[gesture_name]
        seq_len, num_features = landmarks.shape
        processed_landmarks = np.zeros((self.max_len, num_features))
        if seq_len > self.max_len:
            processed_landmarks = landmarks[:self.max_len, :]
        else:
            processed_landmarks[:seq_len, :] = landmarks
        return torch.FloatTensor(processed_landmarks), torch.tensor(label, dtype=torch.long)

class SignClassifierLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout):
        super(SignClassifierLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ------------------- 3. PREPARAÇÃO DOS DADOS -------------------
print("Preparando os dados para o treino oficial com validação...")

class_names = sorted([d for d in os.listdir(LANDMARKS_PATH) if os.path.isdir(os.path.join(LANDMARKS_PATH, d))])
class_map = {name: i for i, name in enumerate(class_names)}
NUM_CLASSES = len(class_names)
print(f"{NUM_CLASSES} classes encontradas: {class_map}")

with open("class_map.json", "w") as f:
    json.dump(class_map, f)
print("Mapa de classes salvo em 'class_map.json'")

all_files = []
for gesture_name in class_names:
    files = glob(os.path.join(LANDMARKS_PATH, gesture_name, "*.npy"))
    for f in files:
        all_files.append((f, gesture_name))

# --- VALIDAÇÃO REATIVADA ---
# Agora dividimos os dados em 80% para treino e 20% para validação.
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42, stratify=[item[1] for item in all_files])

print(f"\nTotal de amostras: {len(all_files)}")
print(f"Amostras de treino: {len(train_files)}")
print(f"Amostras de validação: {len(val_files)}")

# Cria os objetos Dataset e DataLoader para treino e validação
train_dataset = LandmarkDataset(train_files, class_map, max_len=MAX_LEN)
val_dataset = LandmarkDataset(val_files, class_map, max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------- 4. INICIALIZAÇÃO E TREINAMENTO -------------------

# ------------------- 4. INICIALIZAÇÃO E TREINAMENTO -------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

model = SignClassifierLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, dropout=DROPOUT).to(device)
criterion = nn.CrossEntropyLoss() # Se quiser, pode adicionar o 'weight=weights_tensor' aqui
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- NOVO: Variáveis para o Early Stopping ---
best_val_loss = float('inf')  # Começa com um valor infinito
epochs_no_improve = 0         # Contador de épocas sem melhora
patience = 10                 # Número de épocas a esperar antes de parar

print("\nIniciando o treinamento com Early Stopping...")
for epoch in range(NUM_EPOCHS):
    # --- Fase de Treino (continua igual) ---
    model.train()
    train_loss = 0.0
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * sequences.size(0)

    # --- Fase de Validação (continua igual) ---
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
    
    accuracy = accuracy_score(all_labels, all_preds)
    epoch_train_loss = train_loss / len(train_dataset)
    epoch_val_loss = val_loss / len(val_dataset)
    
    print(f"Época {epoch+1}/{NUM_EPOCHS} | "
          f"Loss Treino: {epoch_train_loss:.4f} | "
          f"Loss Validação: {epoch_val_loss:.4f} | "
          f"Acurácia Validação: {accuracy:.4f}")

    # --- NOVO: Lógica do Early Stopping ---
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), "best_libras_model.pth") # Salva o melhor modelo
        epochs_no_improve = 0
        print(f"  -> Novo melhor modelo salvo com Loss Validação: {best_val_loss:.4f}")
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve == patience:
        print(f"\nParada antecipada! O loss de validação não melhora há {patience} épocas.")
        break

print("\nTreinamento concluído!")

# ------------------- 5. SALVAR O MODELO TREINADO -------------------
torch.save(model.state_dict(), "libras_classifier_model.pth")
print("\nTreinamento concluído! Modelo salvo como 'libras_classifier_model.pth'")