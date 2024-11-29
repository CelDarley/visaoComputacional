import cv2
import numpy as np
import os
from datetime import datetime
import json
import pickle
from tqdm import tqdm
import time

class FaceRecognitionSystem:
  def __init__(self):
      self.face_cascade = cv2.CascadeClassifier(
          cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
      )
      self.recognizer = cv2.face.LBPHFaceRecognizer_create()

      self.data_dir = "face_data"
      self.model_file = "face_model.yml"
      self.labels_file = "face_labels.pkl"

      self.label_dict = {}
      self.next_id = 0
      self.is_model_trained = False

      if not os.path.exists(self.data_dir):
          os.makedirs(self.data_dir)

      self.load_model()

  def load_model(self):
      """Carrega o modelo se existir"""
      if os.path.exists(self.model_file) and os.path.exists(self.labels_file):
          try:
              self.recognizer.read(self.model_file)
              with open(self.labels_file, 'rb') as f:
                  data = pickle.load(f)
                  self.label_dict = data['labels']
                  self.next_id = data['next_id']
              self.is_model_trained = True
              print("✅ Modelo carregado com sucesso")
          except Exception as e:
              print(f"❌ Erro ao carregar modelo: {str(e)}")
              self.is_model_trained = False
      else:
          print("ℹ️ Nenhum modelo encontrado. Por favor, adicione faces e treine o modelo.")
          self.is_model_trained = False

  def save_model(self):
      """Salva o modelo treinado"""
      if self.is_model_trained:
          try:
              self.recognizer.write(self.model_file)
              with open(self.labels_file, 'wb') as f:
                  pickle.dump({
                      'labels': self.label_dict,
                      'next_id': self.next_id
                  }, f)
              print("✅ Modelo salvo com sucesso")
          except Exception as e:
              print(f"❌ Erro ao salvar modelo: {str(e)}")
      else:
          print("⚠️ Não há modelo treinado para salvar. Treine o modelo primeiro.")

  def train_model(self):
      """Treina o modelo com as faces cadastradas"""
      print("\n🔄 Iniciando processo de treinamento...")

      if not os.listdir(self.data_dir):
          print("❌ Nenhuma pessoa cadastrada. Adicione faces primeiro.")
          return False

      faces = []
      labels = []
      total_images = 0

      # Conta o total de imagens
      for person_name in os.listdir(self.data_dir):
          person_dir = os.path.join(self.data_dir, person_name)
          if os.path.isdir(person_dir):
              image_files = [f for f in os.listdir(person_dir) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))]
              total_images += len(image_files)

      print(f"📊 Total de imagens encontradas: {total_images}")

      # Barra de progresso principal
      with tqdm(total=total_images, desc="Processando imagens", unit="img") as pbar:
          for person_name in os.listdir(self.data_dir):
              person_dir = os.path.join(self.data_dir, person_name)
              if not os.path.isdir(person_dir):
                  continue

              print(f"\n👤 Processando: {person_name}")

              image_files = [f for f in os.listdir(person_dir) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))]

              if not image_files:
                  print(f"⚠️ Nenhuma imagem encontrada para {person_name}")
                  continue

              if person_name not in self.label_dict:
                  self.label_dict[person_name] = self.next_id
                  self.next_id += 1

              person_id = self.label_dict[person_name]

              for img_name in image_files:
                  img_path = os.path.join(person_dir, img_name)
                  try:
                      image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                      if image is not None:
                          faces.append(image)
                          labels.append(person_id)
                          pbar.update(1)
                      else:
                          print(f"❌ Erro ao ler imagem: {img_path}")
                  except Exception as e:
                      print(f"❌ Erro ao processar {img_path}: {str(e)}")

      if faces and labels:
          print("\n🔄 Iniciando treinamento final...")
          try:
              self.recognizer.train(faces, np.array(labels))
              self.is_model_trained = True
              self.save_model()
              print("✅ Treinamento concluído com sucesso!")
              print(f"📊 Total de faces processadas: {len(faces)}")
              print(f"👥 Pessoas cadastradas: {', '.join(self.label_dict.keys())}")
              return True
          except Exception as e:
              print(f"❌ Erro durante o treinamento: {str(e)}")
              return False
      else:
          print("❌ Nenhuma face válida encontrada para treinar")
          return False

  def process_frame(self, frame):
      """Processa cada frame do vídeo"""
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

      for (x, y, w, h) in faces:
          face = gray[y:y+h, x:x+w]

          color = (0, 0, 255)  # Vermelho para desconhecido
          name = "Desconhecido"
          confidence_text = ""

          if self.is_model_trained:
              try:
                  label_id, confidence = self.recognizer.predict(face)

                  for person_name, person_id in self.label_dict.items():
                      if person_id == label_id and confidence < 100:
                          name = person_name
                          color = (0, 255, 0)  # Verde para reconhecido
                          break

                  confidence_text = f" ({confidence:.1f})"
              except Exception as e:
                  print(f"❌ Erro no reconhecimento: {str(e)}")

          cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
          cv2.putText(frame, name + confidence_text,
                     (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                     0.9, color, 2)

      return frame

def show_menu():
  """Mostra o menu de opções"""
  print("\n📋 Menu de Controles:")
  print("'q' - Sair")
  print("'a' - Adicionar nova face pela webcam")
  print("'d' - Adicionar faces de um diretório")
  print("'t' - Treinar modelo")
  print("'s' - Salvar modelo")
  print("'h' - Mostrar este menu")

def main():
  face_system = FaceRecognitionSystem()
  cap = cv2.VideoCapture(0)

  if not cap.isOpened():
      print("❌ Erro ao abrir a câmera")
      return

  show_menu()
  training_in_progress = False
  last_key_time = time.time()
  key_timeout = 0.5  # Tempo mínimo entre teclas (em segundos)

  while True:
      ret, frame = cap.read()
      if not ret:
          print("❌ Erro ao capturar frame da câmera")
          break

      if not training_in_progress:
          frame = face_system.process_frame(frame)
          cv2.imshow('Reconhecimento Facial', frame)

          key = cv2.waitKey(1) & 0xFF
          current_time = time.time()

          if current_time - last_key_time >= key_timeout:
              if key == ord('q'):
                  print("\n👋 Encerrando programa...")
                  break
              elif key == ord('h'):
                  show_menu()
              elif key == ord('a'):
                  name = input("👤 Digite o nome da pessoa: ")
                  if name:
                      print("📸 Capturando face... Olhe para a câmera.")
                      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                      faces = face_system.face_cascade.detectMultiScale(gray, 1.3, 5)

                      if len(faces) > 0:
                          person_dir = os.path.join(face_system.data_dir, name)
                          if not os.path.exists(person_dir):
                              os.makedirs(person_dir)

                          timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                          cv2.imwrite(os.path.join(person_dir, f"{timestamp}.jpg"), gray)
                          print(f"✅ Face adicionada para {name}")
                          print("ℹ️ Pressione 't' para treinar o modelo com a nova face")
                      else:
                          print("⚠️ Nenhuma face detectada. Tente novamente.")

              elif key == ord('d'):
                  name = input("👤 Digite o nome da pessoa: ")
                  if name:
                      directory = input("📁 Digite o caminho do diretório com as fotos: ")
                      if os.path.exists(directory):
                          files_processed = 0
                          for filename in os.listdir(directory):
                              if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                                  filepath = os.path.join(directory, filename)
                                  if face_system.add_face_from_file(name, filepath):
                                      files_processed += 1
                          print(f"✅ {files_processed} faces processadas para {name}")
                          print("ℹ️ Pressione 't' para treinar o modelo com as novas faces")
                      else:
                          print("❌ Diretório não encontrado")

              elif key == ord('t'):
                  print("🔄 Iniciando treinamento...")
                  training_in_progress = True
                  face_system.train_model()
                  training_in_progress = False
                  print("ℹ️ Pressione 's' para salvar o modelo.")

              elif key == ord('s'):
                  face_system.save_model()

              last_key_time = current_time

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  try:
      main()
  except KeyboardInterrupt:
      print("\n👋 Programa encerrado pelo usuário")
  except Exception as e:
      print(f"❌ Erro inesperado: {str(e)}")