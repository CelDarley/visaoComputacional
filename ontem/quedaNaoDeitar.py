import cv2
import mediapipe as mp
import numpy as np
import time

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

class MovementTracker:
  def __init__(self, buffer_size=10):
      self.positions = []
      self.timestamps = []
      self.buffer_size = buffer_size
      self.person_down = False

  def add_position(self, position):
      current_time = time.time()
      self.positions.append(position)
      self.timestamps.append(current_time)

      if len(self.positions) > self.buffer_size:
          self.positions.pop(0)
          self.timestamps.pop(0)

  def calculate_velocity(self):
      if len(self.positions) < 2:
          return 0

      dy = self.positions[-1] - self.positions[-2]
      dt = self.timestamps[-1] - self.timestamps[-2]

      if dt == 0:
          return 0

      velocity = abs(dy / dt)
      return velocity

def calculate_center_of_gravity(landmarks):
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

def detect_movement_type(tracker, frame_height):
  velocity = tracker.calculate_velocity()
  FALL_VELOCITY_THRESHOLD = frame_height * 0.3

  if velocity > FALL_VELOCITY_THRESHOLD:
      return "QUEDA"
  elif velocity > 0:
      return "MOVIMENTO_LENTO"
  return "PARADO"

def main():
  cap = cv2.VideoCapture(0)
  movement_tracker = MovementTracker()

  # Variáveis de estado
  fall_detected = False
  person_down = False
  fall_start_time = None

  # Configurações
  FALL_DURATION_THRESHOLD = 0.5
  POSITION_THRESHOLD = 0.65  # Limiar para considerar pessoa caída

  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break

      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = pose.process(frame_rgb)

      # Adicionar overlay para debug
      debug_overlay = frame.copy()
      cv2.putText(debug_overlay, "Status:", (10, 30),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

      if results.pose_landmarks:
          mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

          cog = calculate_center_of_gravity(results.pose_landmarks.landmark)
          if cog is not None:
              # Atualizar posição e calcular velocidade
              current_y_pos = cog[1] * frame.shape[0]
              movement_tracker.add_position(current_y_pos)

              # Verificar se pessoa está caída
              is_down = current_y_pos > frame.shape[0] * POSITION_THRESHOLD
              movement_type = detect_movement_type(movement_tracker, frame.shape[0])

              # Lógica de detecção de queda
              if is_down and movement_type == "QUEDA" and not fall_detected:
                  if fall_start_time is None:
                      fall_start_time = time.time()
                  elif time.time() - fall_start_time >= FALL_DURATION_THRESHOLD:
                      fall_detected = True
                      person_down = True

              # Verificar se pessoa levantou
              if not is_down and person_down:
                  fall_detected = False
                  person_down = False
                  fall_start_time = None

              # Mostrar mensagens e debug info
              if fall_detected or person_down:
                  # Adiciona retângulo semi-transparente
                  overlay = frame.copy()
                  cv2.rectangle(overlay, (30, 50), (400, 120), (0, 0, 0), -1)
                  cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

                  # Adiciona o texto em vermelho
                  cv2.putText(frame, "QUEDA DETECTADA!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

              # Informações de debug
              velocity = movement_tracker.calculate_velocity()
              cv2.putText(frame, f"Velocidade: {velocity:.1f} px/s", (10, 60),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
              cv2.putText(frame, f"Movimento: {movement_type}", (10, 90),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
              cv2.putText(frame, f"Pessoa caída: {'Sim' if person_down else 'Não'}", (10, 120),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

      cv2.imshow('Fall Detection', frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()