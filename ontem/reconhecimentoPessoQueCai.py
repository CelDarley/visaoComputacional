import cv2
import mediapipe as mp
import numpy as np
import time

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh  # ou mp_face_detection para detecção mais simples
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicializar detectores
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(
  max_num_faces=1,
  min_detection_confidence=0.5,
  min_tracking_confidence=0.5
)

class PersonTracker:
  def __init__(self, buffer_size=10):
      self.positions = []
      self.timestamps = []
      self.buffer_size = buffer_size
      self.person_down = False
      self.alert_start_time = None
      self.face_landmarks = None
      self.known_faces = {}  # Dicionário para armazenar faces conhecidas

  def add_position(self, position):
      current_time = time.time()
      self.positions.append(position)
      self.timestamps.append(current_time)

      if len(self.positions) > self.buffer_size:
          self.positions.pop(0)
          self.timestamps.pop(0)

  def calculate_velocity_with_direction(self):
      if len(self.positions) < 2:
          return 0

      dy = self.positions[-1] - self.positions[-2]
      dt = self.timestamps[-1] - self.timestamps[-2]

      if dt == 0:
          return 0

      velocity = dy / dt
      return velocity

  def start_alert(self):
      self.alert_start_time = time.time()

  def should_show_alert(self):
      if self.alert_start_time is None:
          return False
      return time.time() - self.alert_start_time < 3.0

  def update_face_landmarks(self, landmarks):
      self.face_landmarks = landmarks

  def get_face_features(self):
      """Extrai características faciais para reconhecimento"""
      if self.face_landmarks is None:
          return None

      # Simplificado - usar pontos chave do rosto como características
      features = []
      for landmark in self.face_landmarks:
          features.extend([landmark.x, landmark.y, landmark.z])
      return np.array(features)

  def compare_faces(self, features1, features2, threshold=0.5):
      """Compara duas faces usando distância euclidiana"""
      if features1 is None or features2 is None:
          return False
      distance = np.linalg.norm(features1 - features2)
      return distance < threshold

  def register_face(self, name, features):
      """Registra uma nova face conhecida"""
      self.known_faces[name] = features

  def identify_face(self, features):
      """Identifica uma face entre as faces conhecidas"""
      if features is None:
          return "Desconhecido"

      for name, known_features in self.known_faces.items():
          if self.compare_faces(features, known_features):
              return name
      return "Desconhecido"

def main():
  cap = cv2.VideoCapture(0)
  tracker = PersonTracker()

  # Variáveis de estado
  fall_detected = False
  person_down = False
  fall_start_time = None

  # Configurações
  FALL_DURATION_THRESHOLD = 1.0
  POSITION_THRESHOLD = 0.65

  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break

      frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      # Processar pose
      pose_results = pose.process(frame_rgb)

      # Processar face
      face_results = face_mesh.process(frame_rgb)

      # Desenhar resultados da face
      if face_results.multi_face_landmarks:
          for face_landmarks in face_results.multi_face_landmarks:
              mp_drawing.draw_landmarks(
                  image=frame,
                  landmark_list=face_landmarks,
                  connections=mp_face_mesh.FACEMESH_TESSELATION,
                  landmark_drawing_spec=None,
                  connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
              )

              # Atualizar landmarks da face
              tracker.update_face_landmarks(face_landmarks.landmark)

              # Extrair e identificar face
              face_features = tracker.get_face_features()
              person_name = tracker.identify_face(face_features)

              # Mostrar nome da pessoa
              cv2.putText(frame, f"Pessoa: {person_name}", (10, 150),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

      # Processar pose e detecção de queda
      if pose_results.pose_landmarks:
          mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

          # Calcular centro de gravidade e detectar queda
          cog = calculate_center_of_gravity(pose_results.pose_landmarks.landmark)
          if cog is not None:
              current_y_pos = cog[1] * frame.shape[0]
              tracker.add_position(current_y_pos)

              is_down = current_y_pos > frame.shape[0] * POSITION_THRESHOLD
              movement_type = detect_movement_type(tracker, frame.shape[0])

              # Lógica de detecção de queda
              if is_down and movement_type == "QUEDA" and not fall_detected:
                  if fall_start_time is None:
                      fall_start_time = time.time()
                  elif time.time() - fall_start_time >= FALL_DURATION_THRESHOLD:
                      fall_detected = True
                      person_down = True
                      tracker.start_alert()

              # Verificar se pessoa levantou
              if movement_type == "LEVANTANDO" or not is_down:
                  fall_detected = False
                  person_down = False
                  fall_start_time = None

              # Mostrar informações
              velocity = tracker.calculate_velocity_with_direction()

              # Definir cor baseado no movimento
              color = (0, 0, 255) if movement_type == "QUEDA" else \
                     (0, 255, 0) if movement_type == "LEVANTANDO" else \
                     (255, 255, 255)

              # Informações na tela
              cv2.putText(frame, f"Velocidade: {velocity:.1f} px/s", (10, 60),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
              cv2.putText(frame, f"Movimento: {movement_type}", (10, 90),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
              cv2.putText(frame, f"Pessoa caída: {'Sim' if person_down else 'Não'}", (10, 120),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

              # Alerta de queda
              if fall_detected or person_down or tracker.should_show_alert():
                  overlay = frame.copy()
                  cv2.rectangle(overlay, (30, 50), (400, 120), (0, 0, 0), -1)
                  cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                  person_name = tracker.identify_face(tracker.get_face_features())
                  alert_text = f"QUEDA DETECTADA! ({person_name})"
                  cv2.putText(frame, alert_text, (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

      cv2.imshow('Fall Detection with Face Recognition', frame)

      key = cv2.waitKey(1) & 0xFF
      if key == ord('q'):
          break
      elif key == ord('r'):  # Registrar nova face
          name = input("Digite o nome da pessoa: ")
          face_features = tracker.get_face_features()
          if face_features is not None:
              tracker.register_face(name, face_features)
              print(f"Face registrada para: {name}")

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()