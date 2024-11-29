import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

class FaceRecognitionSystem:
    def __init__(self):
        print("Inicializando sistema...")
        self.known_faces_dir = "known_faces"
        if not os.path.exists(self.known_faces_dir):
            os.makedirs(self.known_faces_dir)
            print(f"Diretório {self.known_faces_dir} criado")

        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def load_known_faces(self):
        print("Carregando faces conhecidas...")
        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(self.known_faces_dir, filename)
                name = os.path.splitext(filename)[0]
                if self.add_face_from_file(path, name):
                    print(f"Face carregada: {name}")
        print(f"Total de faces carregadas: {len(self.known_face_encodings)}")

    def add_face_from_file(self, image_path, name):
        try:
            print(f"Tentando carregar imagem: {image_path}")
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            if face_encodings:
                self.known_face_encodings.append(face_encodings[0])
                self.known_face_names.append(name)
                print(f"Face de {name} adicionada com sucesso")
                return True
            print(f"Nenhuma face detectada em {image_path}")
            return False
        except Exception as e:
            print(f"Erro ao adicionar face do arquivo {image_path}: {str(e)}")
            return False

    def add_face_from_webcam(self, name):
        print("Iniciando captura da webcam...")
        try:
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                print("Erro: Não foi possível acessar a webcam!")
                return

            # Configurações da câmera
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            print("Câmera iniciada com sucesso")
            cv2.waitKey(1000)  # Aguardar a câmera inicializar

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Erro na captura do frame!")
                    break

                # Mostrar guia de posicionamento
                height, width = frame.shape[:2]
                center_x, center_y = width // 2, height // 2
                cv2.rectangle(frame, 
                            (center_x - 100, center_y - 100),
                            (center_x + 100, center_y + 100),
                            (0, 255, 0), 2)

                cv2.putText(frame, "Posicione o rosto no quadrado", 
                           (width//2 - 150, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "SPACE: Capturar | Q: Sair", 
                           (width//2 - 150, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Captura de Face', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    # Salvar imagem
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{name}_{timestamp}.jpg"
                    filepath = os.path.join(self.known_faces_dir, filename)
                    cv2.imwrite(filepath, frame)
                    print(f"Imagem salva em: {filepath}")

                    if self.add_face_from_file(filepath, name):
                        print(f"Face de {name} adicionada com sucesso!")
                        break
                    else:
                        print("Nenhuma face detectada. Tente novamente.")
                        os.remove(filepath)

                elif key == ord('q'):
                    print("Captura cancelada pelo usuário")
                    break

        except Exception as e:
            print(f"Erro durante a captura: {str(e)}")

        finally:
            print("Finalizando captura...")
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()

    def start_recognition(self):
        print("Iniciando reconhecimento facial...")

        if len(self.known_face_encodings) == 0:
            print("Nenhuma face cadastrada! Por favor, cadastre uma face primeiro.")
            return

        try:
            # Inicializa a câmera
            print("Tentando abrir a câmera...")
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                print("Erro: Não foi possível acessar a câmera!")
                return

            # Configurações da câmera
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            print("Câmera iniciada com sucesso!")
            print(f"Faces conhecidas carregadas: {len(self.known_face_encodings)}")

            # Aguarda a câmera inicializar
            cv2.waitKey(2000)

            frame_count = 0
            last_detection_time = None
            detection_message = ""
            show_message_duration = 5  # duração em segundos

            while True:
                try:
                    ret, frame = cap.read()
                    if not ret:
                        print("Erro ao capturar frame!")
                        break

                    frame_count += 1
                    current_time = datetime.now()

                    # Processa apenas a cada 3 frames para melhor performance
                    if frame_count % 3 != 0:
                        # Mostra o frame com a mensagem, se houver
                        if last_detection_time and (current_time - last_detection_time).total_seconds() < show_message_duration:
                            # Adiciona a mensagem de detecção
                            cv2.putText(frame, detection_message, (10, 60), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        cv2.putText(frame, "Pressione 'q' para sair", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow('Reconhecimento Facial (Q para sair)', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        continue

                    # Redimensiona o frame para processamento mais rápido
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

                    if face_locations:
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                            name = "Desconhecido"

                            if self.known_face_encodings:
                                matches = face_recognition.compare_faces(
                                    self.known_face_encodings, 
                                    face_encoding,
                                    tolerance=0.6
                                )

                                if True in matches:
                                    first_match_index = matches.index(True)
                                    name = self.known_face_names[first_match_index]
                                    # Atualiza a mensagem e o tempo quando encontra uma face conhecida
                                    if name != "Desconhecido":
                                        last_detection_time = current_time
                                        detection_message = f"Pessoa reconhecida: {name}!"

                                # Ajusta as coordenadas para o tamanho original
                                top *= 4
                                right *= 4
                                bottom *= 4
                                left *= 4

                                # Desenha o retângulo e o nome
                                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                                cv2.putText(frame, name, (left + 6, bottom - 6), 
                                          cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

                    # Adiciona a mensagem de detecção se estiver dentro do período de exibição
                    if last_detection_time and (current_time - last_detection_time).total_seconds() < show_message_duration:
                        cv2.putText(frame, detection_message, (10, 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Adiciona informações no frame
                    cv2.putText(frame, "Pressione 'q' para sair", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Mostra o frame
                    cv2.imshow('Reconhecimento Facial (Q para sair)', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Tecla Q pressionada. Encerrando...")
                        break

                except Exception as e:
                    print(f"Erro durante o processamento do frame: {str(e)}")
                    continue

        except Exception as e:
            print(f"Erro durante o reconhecimento: {str(e)}")

        finally:
            print("Liberando recursos...")
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            print("Reconhecimento encerrado!")

if __name__ == "__main__":
    try:
        print("Iniciando sistema...")
        print("Verificando dependências...")

        # Verifica se as bibliotecas necessárias estão disponíveis
        print(f"OpenCV versão: {cv2.__version__}")
        print("face_recognition disponível")
        print("numpy disponível")

        # Cria instância do sistema
        print("Inicializando sistema de reconhecimento facial...")
        face_system = FaceRecognitionSystem()

        while True:
            try:
                print("\n=== Sistema de Reconhecimento Facial ===")
                print("1. Adicionar face da webcam")
                print("2. Iniciar reconhecimento")
                print("3. Sair")

                choice = input("Escolha uma opção: ")

                if choice == "1":
                    name = input("Digite o nome da pessoa: ")
                    face_system.add_face_from_webcam(name)
                elif choice == "2":
                    print("Iniciando modo de reconhecimento...")
                    face_system.start_recognition()
                elif choice == "3":
                    print("Encerrando programa...")
                    break
                else:
                    print("Opção inválida!")

            except Exception as e:
                print(f"Erro durante a execução: {str(e)}")
                print("Tentando continuar...")
                continue

    except Exception as e:
        print(f"Erro fatal durante a inicialização: {str(e)}")

    finally:
        print("Programa encerrado.")