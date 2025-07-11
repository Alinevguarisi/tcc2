import cv2
import os
import numpy as np
import random
from glob import glob
import mediapipe as mp

# Inicializa o MediaPipe Holistic
mp_holistic = mp.solutions.holistic

def normalize_frame(frame, size=(224, 224)):
    return cv2.resize(frame, size)

def get_dynamic_roi(frame):
    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        landmarks = []
        if results.face_landmarks:
            landmarks.extend([(lm.x, lm.y) for lm in results.face_landmarks.landmark])
        if results.left_hand_landmarks:
            landmarks.extend([(lm.x, lm.y) for lm in results.left_hand_landmarks.landmark])
        if results.right_hand_landmarks:
            landmarks.extend([(lm.x, lm.y) for lm in results.right_hand_landmarks.landmark])
        if results.pose_landmarks:
            landmarks.extend([(lm.x, lm.y) for lm in results.pose_landmarks.landmark])

        if not landmarks:
            return None

        h, w, _ = frame.shape
        coords = np.array([[int(x * w), int(y * h)] for x, y in landmarks])

        x_min = max(np.min(coords[:, 0]) - 20, 0)
        y_min = max(np.min(coords[:, 1]) - 20, 0)
        x_max = min(np.max(coords[:, 0]) + 20, w)
        y_max = min(np.max(coords[:, 1]) + 20, h)

        roi = frame[y_min:y_max, x_min:x_max]
        return roi

def apply_augmentation(image):
    # Flip horizontal com 30% de chance
    if random.random() < 0.3:
        image = cv2.flip(image, 1)

    # Rotação leve entre -10 e +10 graus
    angle = random.uniform(-10, 10)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    image = cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Ajuste de brilho/saturação com variação menor
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= random.uniform(0.95, 1.05)  # saturação
    hsv[..., 2] *= random.uniform(0.95, 1.1)   # brilho
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Desfoque leve com 20% de chance
    if random.random() < 0.2:
        ksize = 3
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    # Ruído Gaussiano suave
    noise = np.random.normal(0, 2, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image

def create_output_directory(base_path, gesture):
    gesture_path = os.path.join(base_path, gesture)
    os.makedirs(gesture_path, exist_ok=True)
    existing_sequences = glob(os.path.join(gesture_path, "sequence_*"))
    sequence_number = len(existing_sequences)
    out_dir = os.path.join(gesture_path, f'sequence_{sequence_number}')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

# === CAMINHO DO DRIVE ===
BASE_DRIVE_PATH = r"G:\Meu Drive\TCC - Aline e Gabi"
out_base_path = r'C:\Users\Aline\Desktop\gestures_dataset'

# Busca todos os arquivos mp4 dentro da pasta e subpastas
video_files = glob(os.path.join(BASE_DRIVE_PATH, '**', '*.mp4'), recursive=True)

for vid_path in video_files:
    gesture_name = os.path.basename(os.path.dirname(vid_path))
    vid_name = os.path.basename(vid_path)
    print(f"Processando vídeo: {vid_name} para o gesto: {gesture_name}")

    video = cv2.VideoCapture(vid_path)
    if not video.isOpened():
        print(f"Erro ao abrir o vídeo: {vid_name}")
        continue

    out_dir = create_output_directory(out_base_path, gesture_name)
    print(f"Pasta de saída criada: {out_dir}")

    i = 1
    while video.isOpened():
        flag, frame = video.read()
        if not flag:
            break

        normalized_frame = normalize_frame(frame)
        roi_frame = get_dynamic_roi(normalized_frame)

        if roi_frame is None:
            print(f"Frame {i}: Nenhum ROI detectado, salvando apenas o frame completo.")
            roi_frame = normalized_frame

        # Salva o ROI original (sem augmentation)
        roi_original_path = os.path.join(out_dir, f"frame_{i}_roi_raw.jpg")
        cv2.imwrite(roi_original_path, roi_frame)

        # Aplica data augmentation no ROI
        roi_augmented = apply_augmentation(roi_frame)

        # Caminhos para salvar
        normalized_frame_path = os.path.join(out_dir, f"frame_{i}.jpg")
        roi_augmented_path = os.path.join(out_dir, f"frame_{i}_roi.jpg")

        # Salva arquivos
        cv2.imwrite(normalized_frame_path, normalized_frame)
        cv2.imwrite(roi_augmented_path, roi_augmented)

        print(f'Imagem salva: {normalized_frame_path}')
        print(f'ROI original salva: {roi_original_path}')
        print(f'ROI aumentada salva: {roi_augmented_path}')
        i += 1

    video.release()
    print(f'✅ Conversão concluída para: {vid_name}\n')

print("🏁 Processamento de vídeos finalizado.")
