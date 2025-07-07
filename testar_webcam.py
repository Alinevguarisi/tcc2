# ===================================================================
# SCRIPT DE TESTE EM TEMPO REAL COM A WEBCAM
# ===================================================================

import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import json

# ------------------- 1. CONFIGURAÇÕES -------------------
# Caminho para o modelo treinado e para o mapa de classes
MODEL_PATH = "best_libras_model.pth"
CLASS_MAP_PATH = "class_map.json"

# Parâmetros que DEVEM ser iguais aos do treinamento
MAX_LEN = 150
INPUT_SIZE = 1662
HIDDEN_SIZE = 128 # Use o mesmo valor do seu script de treino (128 ou 256)
NUM_LAYERS = 2

# ------------------- 2. CARREGAR MODELO E CLASSES -------------------

# Carrega o mapa de classes (ex: {'oi': 1} -> {1: 'oi'})
with open(CLASS_MAP_PATH, 'r') as f:
    class_map = json.load(f)
# Inverte o dicionário para mapear de índice para nome
idx_to_class = {v: k for k, v in class_map.items()}
NUM_CLASSES = len(class_map)

# Recria a arquitetura do modelo
# (É necessário ter a definição da classe do modelo para carregar os pesos)
class SignClassifierLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SignClassifierLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Carrega o modelo e os pesos treinados
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SignClassifierLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() # Coloca o modelo em modo de avaliação

print("Modelo e mapa de classes carregados com sucesso.")

# ------------------- 3. INICIALIZAÇÃO DO MEDIAPIPE E WEBCAM -------------------

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh]), results

# Inicia a captura da webcam
cap = cv2.VideoCapture(0)

sequence = []
current_prediction = ""
confidence_threshold = 0.6 # Limiar de confiança para mostrar a previsão

# ------------------- 4. LOOP DE RECONHECIMENTO EM TEMPO REAL -------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extrai landmarks
    landmarks, results = extract_landmarks(frame)
    sequence.append(landmarks)
    
    # Mantém a sequência com o tamanho máximo
    sequence = sequence[-MAX_LEN:]
    
    # Faz a previsão apenas quando a sequência está cheia
    if len(sequence) == MAX_LEN:
        # Prepara os dados para o modelo
        input_tensor = torch.FloatTensor(np.array(sequence)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            # Usa softmax para obter probabilidades
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Atualiza a previsão se a confiança for alta o suficiente
            if confidence.item() > confidence_threshold:
                current_prediction = idx_to_class[predicted_idx.item()]
            else:
                current_prediction = "..."

    # Desenha os landmarks na imagem
    if results.face_landmarks:
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Mostra a previsão na tela
    cv2.putText(frame, current_prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Reconhecimento de Libras em Tempo Real', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()