# ===================================================================
# SCRIPT PARA PROCESSAMENTO DE VÍDEOS DE LIBRAS PARA LANDMARKS
# TCC - Aline e Gabi
#
# O que este script faz:
# 1. Lê os vídeos da pasta do Google Drive.
# 2. Usa o MediaPipe para extrair pontos-chave (landmarks) do rosto,
#    mãos e corpo em cada frame do vídeo.
# 3. Salva a sequência de landmarks de cada vídeo como um arquivo .npy
#    em uma nova pasta de dataset, mantendo a organização por gestos.
# ===================================================================

import cv2
import os
import numpy as np
import mediapipe as mp
from glob import glob
import time

# ------------------- 1. CONFIGURAÇÃO DE CAMINHOS -------------------

BASE_DRIVE_PATH = r"G:\Meu Drive\TCC - Aline e Gabi"

# O script vai procurar seus vídeos nesta pasta:
VIDEOS_BASE_PATH = os.path.join(BASE_DRIVE_PATH, "videos_libras")

# O novo dataset com os arquivos .npy será salvo aqui:
LANDMARKS_OUTPUT_PATH = os.path.join(BASE_DRIVE_PATH, "TCC_Dataset_Landmarks")

# ------------------- 2. DESCOBERTA AUTOMÁTICA DOS VÍDEOS -------------------

print("="*50)
print("INICIANDO SCRIPT DE PROCESSAMENTO DE DADOS")
print("="*50)

# Verificação de segurança para garantir que o caminho principal existe
if not os.path.exists(VIDEOS_BASE_PATH):
    print(f"\n[ERRO CRÍTICO]")
    print(f"O caminho para os vídeos não foi encontrado: '{VIDEOS_BASE_PATH}'")
    print("Por favor, verifique se a letra do drive na variável 'BASE_DRIVE_PATH' está correta.")
    exit() # Encerra o script se o caminho estiver errado

print(f"\nProcurando vídeos em: '{VIDEOS_BASE_PATH}'...")

video_files = []
# Lista todas as subpastas (que são os nomes dos gestos)
class_names = sorted([d for d in os.listdir(VIDEOS_BASE_PATH) if os.path.isdir(os.path.join(VIDEOS_BASE_PATH, d))])

if not class_names:
    print("\n[ERRO] Nenhuma pasta de gesto foi encontrada dentro de 'videos_libras'.")
    exit()

print(f"\n{len(class_names)} classes (gestos) encontradas: {class_names}")

for gesture_name in class_names:
    # Procura por todos os arquivos .mp4 dentro de cada pasta de gesto
    search_path = os.path.join(VIDEOS_BASE_PATH, gesture_name, "*.mp4")
    videos_in_folder = glob(search_path)
    for vid_path in videos_in_folder:
        video_files.append((vid_path, gesture_name))

print(f"\nTotal de {len(video_files)} vídeos encontrados para processamento.\n")
print("-"*50)


# ------------------- 3. INICIALIZAÇÃO DO MEDIAPIPE E FUNÇÕES -------------------

# Inicializa o modelo Holístico do MediaPipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_landmarks(frame):
    """Processa um frame e extrai os landmarks, retornando um vetor numpy."""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    
    # Extrai e achata os landmarks. Se não detectar, cria um vetor de zeros.
    # Isso garante que todos os vetores de saída tenham o mesmo tamanho.
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    
    return np.concatenate([pose, face, lh, rh])

# ------------------- 4. LOOP PRINCIPAL DE PROCESSAMENTO -------------------

start_time = time.time()
for i, (vid_path, gesture_name) in enumerate(video_files):
    
    # Define onde o arquivo .npy será salvo
    gesture_output_folder = os.path.join(LANDMARKS_OUTPUT_PATH, gesture_name)
    os.makedirs(gesture_output_folder, exist_ok=True)
    
    video_basename = os.path.splitext(os.path.basename(vid_path))[0]
    output_filename = os.path.join(gesture_output_folder, f"{video_basename}.npy")
    
    # Pula o vídeo se ele já foi processado anteriormente
    if os.path.exists(output_filename):
        print(f"({i+1}/{len(video_files)}) VÍDEO JÁ PROCESSADO: '{vid_path}'... PULANDO")
        continue
        
    print(f"({i+1}/{len(video_files)}) PROCESSANDO VÍDEO: '{vid_path}'...")
    
    video = cv2.VideoCapture(vid_path)
    sequence_landmarks = [] # Lista para guardar os landmarks de todos os frames do vídeo
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        # Extrai os landmarks do frame atual e adiciona à nossa sequência
        landmarks = extract_landmarks(frame)
        sequence_landmarks.append(landmarks)
        
    video.release()
    
    # Salva a sequência completa de landmarks do vídeo em um único arquivo .npy
    if sequence_landmarks:
        np.save(output_filename, np.array(sequence_landmarks))
        print(f"  -> SUCESSO! Sequência salva em '{output_filename}' com {len(sequence_landmarks)} frames.")
    else:
        print(f"  -> AVISO! Nenhum frame processado para o vídeo '{vid_path}'.")

# ------------------- 5. FINALIZAÇÃO -------------------

holistic.close() # Libera os recursos do MediaPipe
end_time = time.time()

print("\n" + "="*50)
print("PROCESSAMENTO DE TODOS OS VÍDEOS CONCLUÍDO!")
print(f"Tempo total de execução: {((end_time - start_time) / 60):.2f} minutos.")
print(f"Seu novo dataset está pronto em: '{LANDMARKS_OUTPUT_PATH}'")
print("="*50)