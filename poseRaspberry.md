~~~bash 
sudo su 
source raspPose/bin/actvate  
sudo nano treinar_modelo.py  
~~~
~~~python

pip install pandas
pip install scikit-learn
python3 treinar_modelo.py
sudo nano aplicacao_tempo_real.py

~~~python

import cv2
import mediapipe as mp
import numpy as np
from sklearn.linear_model import SGDClassifier
import joblib
import os
import math

# Número de features esperado (coordenadas brutas + features derivadas)
NUM_FEATURES = 109

# Verificar se existe um modelo salvo
if os.path.exists("pose_classifier_incremental.pkl"):
    # Carregar o modelo incremental existente
    model = joblib.load("pose_classifier_incremental.pkl")
    print("Modelo carregado com sucesso!")

    # Verificar se o número de features é compatível
    if model.coef_.shape[1] != NUM_FEATURES:
        print("Número de features incompatível. Reinicializando o modelo.")
        model = SGDClassifier()
else:
    # Inicializar um modelo incremental (SGDClassifier)
    model = SGDClassifier()
    print("Novo modelo inicializado!")

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Função para calcular a distância entre dois pontos
def calculate_distance(p1, p2):
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)

# Função para calcular o ângulo entre três pontos
def calculate_angle(p1, p2, p3):
    # Vetores
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    # Produto escalar e norma
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    # Ângulo em graus
    angle = math.degrees(math.acos(dot_product / (norm_v1 * norm_v2)))
    return angle

# Função para extrair features derivadas
def extract_features(landmarks):
    features = []

    # Distâncias
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    # Distâncias entre ombros e quadris
    features.append(calculate_distance(left_shoulder, left_hip))
    features.append(calculate_distance(right_shoulder, right_hip))

    # Distâncias entre quadris e joelhos
    features.append(calculate_distance(left_hip, left_knee))
    features.append(calculate_distance(right_hip, right_knee))

    # Distâncias entre joelhos e tornozelos
    features.append(calculate_distance(left_knee, left_ankle))
    features.append(calculate_distance(right_knee, right_ankle))

    # Ângulos
    # Ângulo do tronco (ombro-quadril-joelho)
    features.append(calculate_angle(left_shoulder, left_hip, left_knee))
    features.append(calculate_angle(right_shoulder, right_hip, right_knee))

    # Ângulo das pernas (quadril-joelho-tornozelo)
    features.append(calculate_angle(left_hip, left_knee, left_ankle))
    features.append(calculate_angle(right_hip, right_knee, right_ankle))

    return features

# Função para prever a posição com base nos landmarks
def predict_pose(landmarks):
    data = []

    # Adicionar coordenadas brutas (x, y, z)
    for lm in landmarks:
        data.extend([lm.x, lm.y, lm.z])

    # Adicionar features derivadas
    data.extend(extract_features(landmarks))

    data = np.array([data])  # Formatar como uma matriz 2D
    return model.predict(data)[0]  # Retorna a classe prevista

# Função para treinar o modelo incrementalmente
def train_incremental(landmarks, label):
    data = []

    # Adicionar coordenadas brutas (x, y, z)
    for lm in landmarks:
        data.extend([lm.x, lm.y, lm.z])

    # Adicionar features derivadas
    data.extend(extract_features(landmarks))

    data = np.array([data])  # Formatar como uma matriz 2D
    label = np.array([label])  # Formatar o rótulo como uma matriz 1D

    # Log dos dados usados para treinamento
    print("Treinando com os seguintes dados:")
    print("Features:", data)
    print("Classe:", label)

    # Treinamento incremental
    model.partial_fit(data, label, classes=np.array(["De Pé", "Sentado", "Deitado"]))
    print(f"Modelo atualizado com a classe: {label[0]}")

# Capturar vídeo da webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converter imagem para RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Processar a imagem com MediaPipe
    results = pose.process(image)

    # Converter de volta para BGR para exibição
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Verificar se landmarks foram detectados
    if results.pose_landmarks:
        # Desenhar os landmarks na imagem
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Prever a posição
        landmarks = results.pose_landmarks.landmark
        try:
            pose_class = predict_pose(landmarks)
        except:
            pose_class = "Desconhecido"  # Caso o modelo ainda não tenha sido treinado

        # Exibir a classificação na imagem
        cv2.putText(
            image,
            f"Posicao: {pose_class}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Mostrar a imagem
        cv2.imshow('Pose Detection', image)

        # Permitir correção manual
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):  # Sair com a tecla 'q'
            break
        elif key == ord('s'):  # Corrigir para "Sentado"
            train_incremental(landmarks, "Sentado")
            print("Corrigido para: Sentado")
        elif key == ord('p'):  # Corrigir para "De Pé"
            train_incremental(landmarks, "De Pé")
            print("Corrigido para: De Pé")
        elif key == ord('d'):  # Corrigir para "Deitado"
            train_incremental(landmarks, "Deitado")
            print("Corrigido para: Deitado")

# Salvar o modelo atualizado
joblib.dump(model, "pose_classifier_incremental.pkl")
print("Modelo incremental salvo como 'pose_classifier_incremental.pkl'")

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

~~

python aplicacao_tempo_real.py
rm pose_classifier_incremental.pkl
python3 treinar_modelo.py 
python3 aplicacao_tempo_real.py
python3 aplicacao_tempo_real.py
nano pose_classifier_incremental.pkl 
