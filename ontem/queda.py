import cv2
import mediapipe as mp
import numpy as np
import time

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def calculate_center_of_gravity(landmarks):
  """Calcula o centro de gravidade baseado nos pontos do quadril e ombros"""
  try:
      left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])
      right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
      left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
      right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])

      hip_center = (left_hip + right_hip) / 2
      shoulder_center = (left_shoulder + right_shoulder) / 2
      cog = (hip_center + shoulder_center) / 2
      return cog
  except:
      return None

def detect_fall(landmarks, frame_height):
  """Detecta queda baseado na posição vertical do centro de gravidade"""
  if landmarks is None:
      return False

  cog = calculate_center_of_gravity(landmarks)
  if cog is None:
      return False

  cog_y_pixels = cog[1] * frame_height
  fall_threshold = frame_height * 0.65  # Ajustado para ser mais sensível

  return cog_y_pixels > fall_threshold

def main():
  cap = cv2.VideoCapture(0)

  # Variáveis para controle
  fall_start_time = None
  fall_detected = False
  debug_mode = True  # Ativar modo debug

  # Ajuste de sensibilidade
  FALL_DURATION_THRESHOLD = 1.0  # Reduzido para 1 segundo

  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break

      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = pose.process(frame_rgb)

      # Adicionar informações de debug
      if debug_mode:
          cv2.putText(frame, "Status:", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

      if results.pose_landmarks:
          mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

          current_fall_state = detect_fall(results.pose_landmarks.landmark, frame.shape[0])

          # Debug info
          if debug_mode:
              status = "Em pé"
              color = (0, 255, 0)  # Verde

              if current_fall_state:
                  status = "Possível queda detectada"
                  color = (0, 255, 255)  # Amarelo
                  if fall_start_time:
                      time_elapsed = time.time() - fall_start_time
                      status = f"Verificando queda: {time_elapsed:.1f}s"

              if fall_detected:
                  status = "MOVIMENTO SUSPEITO!"
                  color = (0, 0, 255)  # Vermelho

              cv2.putText(frame, status, (100, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

          # Lógica de detecção de queda
          if current_fall_state:
              if fall_start_time is None:
                  fall_start_time = time.time()
              elif time.time() - fall_start_time >= FALL_DURATION_THRESHOLD:
                  fall_detected = True

                  # Adiciona retângulo semi-transparente
                  overlay = frame.copy()
                  cv2.rectangle(overlay, (30, 50), (400, 120), (0, 0, 0), -1)
                  cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

                  # Adiciona o texto em vermelho (maior e mais visível)
                  cv2.putText(frame, "QUEDA DETECTADA!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
          else:
              fall_start_time = None
              fall_detected = False
      else:
          if debug_mode:
              cv2.putText(frame, "Nenhuma pessoa detectada", (100, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

      # Mostrar frame
      cv2.imshow('Fall Detection', frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()