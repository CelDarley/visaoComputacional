import cv2
import mediapipe as mp
import numpy as np
import math

class PoseDetector:
  def __init__(self):
      self.mp_pose = mp.solutions.pose
      self.pose = self.mp_pose.Pose()
      self.mp_draw = mp.solutions.drawing_utils

  def calculate_angle(self, a, b, c):
      # Calcula o ângulo entre três pontos
      a = np.array([a.x, a.y])
      b = np.array([b.x, b.y])
      c = np.array([c.x, c.y])

      radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
      angle = np.abs(radians * 180.0 / np.pi)

      if angle > 180.0:
          angle = 360 - angle

      return angle

  def calculate_body_orientation(self, shoulder, hip):
      # Calcula o ângulo do corpo em relação à vertical
      dx = shoulder.x - hip.x
      dy = shoulder.y - hip.y
      angle = math.degrees(math.atan2(dx, dy))
      return abs(angle)

  def detect_pose(self, frame):
      # Converter a imagem para RGB
      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = self.pose.process(rgb_frame)

      if results.pose_landmarks:
          # Desenhar os pontos do corpo
          self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

          # Obter landmarks
          landmarks = results.pose_landmarks.landmark

          # Pontos importantes
          left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
          right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
          left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
          right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
          knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]
          ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]

          # Calcular ângulos
          knee_angle = self.calculate_angle(left_hip, knee, ankle)
          hip_angle = self.calculate_angle(left_shoulder, left_hip, knee)
          body_orientation = self.calculate_body_orientation(left_shoulder, left_hip)

          # Calcular altura relativa dos pontos
          shoulder_height = (left_shoulder.y + right_shoulder.y) / 2
          hip_height = (left_hip.y + right_hip.y) / 2

          # Determinar a pose baseada nos ângulos e orientação
          pose_status = self.determine_pose(knee_angle, hip_angle, body_orientation, 
                                         shoulder_height, hip_height)

          # Adicionar informações na imagem
          cv2.putText(frame, f"Pose: {pose_status}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
          cv2.putText(frame, f"Joelho: {int(knee_angle)}", (10, 60),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
          cv2.putText(frame, f"Quadril: {int(hip_angle)}", (10, 90),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
          cv2.putText(frame, f"Orientacao: {int(body_orientation)}", (10, 120),
                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

      return frame

  def determine_pose(self, knee_angle, hip_angle, body_orientation, shoulder_height, hip_height):
      # Verificar primeiro se está deitado
      height_diff = abs(shoulder_height - hip_height)
      is_horizontal = 45 <= body_orientation <= 135
      points_same_height = height_diff < 0.1

      if is_horizontal or points_same_height:
          if shoulder_height > hip_height:
              return "Deitado (Cabeça à Direita)"
          else:
              return "Deitado (Cabeça à Esquerda)"

      # Se não estiver deitado, verificar outras poses
      if knee_angle > 160 and hip_angle > 160:  # Pernas e tronco retos
          return "Em Pe"
      elif knee_angle < 90 and hip_angle < 90:  # Joelhos e quadril muito dobrados
          return "Agachado"
      elif knee_angle > 90 and hip_angle < 120:  # Quadril dobrado, joelhos menos dobrados
          return "Sentado"
      elif 90 <= hip_angle <= 160 and 90 <= knee_angle <= 160:  # Posição intermediária
          return "Semi-agachado"
      else:
          return "Indefinido"

def main():
  # Inicializar a câmera
  cap = cv2.VideoCapture(0)
  detector = PoseDetector()

  while True:
      success, frame = cap.read()
      if not success:
          break

      # Detectar a pose
      frame = detector.detect_pose(frame)

      # Mostrar o resultado
      cv2.imshow("Pose Detection", frame)

      # Pressione 'q' para sair
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()