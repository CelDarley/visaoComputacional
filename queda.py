import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# Inicialização do MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Buffer para armazenar posições anteriores do CG
cg_positions = deque(maxlen=10)
cg_timestamps = deque(maxlen=10)

def calculate_angle(a, b, c):
    """
    Calcula o ângulo entre três pontos
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)

    return angle

def calculate_vertical_angle(point1, point2):
    """
    Calcula o ângulo em relação à vertical
    """
    vertical = np.array([0, -1])  # Vetor vertical para referência
    vector = np.array([point2[0] - point1[0], point2[1] - point1[1]])

    # Normaliza o vetor
    vector = vector / np.linalg.norm(vector)

    # Calcula o ângulo com a vertical
    angle = np.arccos(np.dot(vertical, vector))
    angle = np.degrees(angle)

    return angle

def detect_fall(landmarks):
    """
    Detecta se houve queda baseado nos ângulos do corpo
    """
    # Extrai coordenadas dos pontos-chave
    left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
    right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
    left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
    right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
    left_ankle = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])
    right_ankle = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])

    # Calcula o ponto médio entre os ombros e quadris
    mid_shoulder = (left_shoulder + right_shoulder) / 2
    mid_hip = (left_hip + right_hip) / 2
    mid_ankle = (left_ankle + right_ankle) / 2

    # Calcula ângulos
    spine_angle = calculate_vertical_angle(mid_hip, mid_shoulder)
    leg_angle = calculate_vertical_angle(mid_ankle, mid_hip)
    body_angle = calculate_angle(mid_shoulder, mid_hip, mid_ankle)

    # Retorna os ângulos para visualização
    return {
        'spine_angle': spine_angle,
        'leg_angle': leg_angle,
        'body_angle': body_angle,
        'is_fallen': (spine_angle > 60 and leg_angle > 60) or body_angle < 110
    }

# Inicializa a câmera
cap = cv2.VideoCapture(0)

# Variáveis para controle do alerta
fall_detected = False
fall_start_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converte a imagem para RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Detecta pose
    results = pose.process(image)

    # Converte de volta para BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        # Desenha os pontos da pose
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        # Verifica se houve queda
        angles = detect_fall(results.pose_landmarks.landmark)

        # Exibe os ângulos na tela
        cv2.putText(image, f'Angulo Coluna: {angles["spine_angle"]:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, f'Angulo Pernas: {angles["leg_angle"]:.1f}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(image, f'Angulo Corpo: {angles["body_angle"]:.1f}', (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        if angles['is_fallen']:
            if not fall_detected:
                fall_detected = True
                fall_start_time = time.time()

            # Exibe mensagem de alerta por 4 segundos
            if time.time() - fall_start_time < 4:
                cv2.putText(image, 'PESSOA CAIDA DETECTADA!', (50,120),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            else:
                fall_detected = False

    # Mostra a imagem
    cv2.imshow('MediaPipe Pose', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()