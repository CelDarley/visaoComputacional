import cv2
import mediapipe as mp
import numpy as np

class FallDetector:
  def __init__(self):
      self.mp_pose = mp.solutions.pose
      self.pose = self.mp_pose.Pose()
      self.mp_draw = mp.solutions.drawing_utils

  def detect_fall(self, frame):
      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = self.pose.process(rgb_frame)

      if results.pose_landmarks:
          self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

          # Obter landmarks
          landmarks = results.pose_landmarks.landmark
          left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
          left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
          left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value]

          # Calcular a altura do centro de massa
          center_of_mass = (left_shoulder.y + left_hip.y + left_knee.y) / 3

          # Verificar se a altura do centro de massa caiu abaixo de um certo limiar
          if center_of_mass > 0.5:  # Ajuste o limiar conforme necessário
              cv2.putText(frame, "Possível Queda Detectada!", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

      return frame

def main():
  cap = cv2.VideoCapture(0)
  detector = FallDetector()

  while True:
      success, frame = cap.read()
      if not success:
          break

      frame = detector.detect_fall(frame)
      cv2.imshow("Detecção de Queda", frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()