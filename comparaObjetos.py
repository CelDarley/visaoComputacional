import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

class ObjectRecognizer:
    def __init__(self, model_name="object_model"):
        self.model_name = model_name
        self.model = None
        self.image_size = (224, 224)

    def create_model(self, num_classes):
        """Cria o modelo CNN para reconhecimento de objetos"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.image_size, 3)),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        return model

    def train_model(self, train_dir, validation_dir, epochs=10, batch_size=32):
        """Treina o modelo com as imagens fornecidas"""
        # Configuração do data generator para aumentação de dados
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        validation_datagen = ImageDataGenerator(rescale=1./255)

        # Geradores de dados de treinamento e validação
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical'
        )

        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=self.image_size,
            batch_size=batch_size,
            class_mode='categorical'
        )

        # Treinamento do modelo
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size
        )

        # Salva o modelo treinado
        self.model.save(f"{self.model_name}.h5")
        return history

    def detect_object(self, image_path):
        """Detecta objetos em uma imagem"""
        # Carrega e pré-processa a imagem
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.image_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Faz a predição
        prediction = self.model.predict(img)
        return prediction

    def real_time_detection(self):
        """Realiza detecção em tempo real usando a webcam"""
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Pré-processa o frame
            processed_frame = cv2.resize(frame, self.image_size)
            processed_frame = processed_frame / 255.0
            processed_frame = np.expand_dims(processed_frame, axis=0)

            # Faz a predição
            prediction = self.model.predict(processed_frame)

            # Obtém a classe com maior probabilidade
            class_idx = np.argmax(prediction[0])
            confidence = prediction[0][class_idx]

            # Desenha o resultado no frame
            cv2.putText(frame, f"Class: {class_idx}, Conf: {confidence:.2f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Object Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Exemplo de uso
def main():
    # Cria diretórios para treinamento se não existirem
    os.makedirs("dataset/train", exist_ok=True)
    os.makedirs("dataset/validation", exist_ok=True)

    # Inicializa o reconhecedor
    recognizer = ObjectRecognizer()

    # Cria o modelo (ajuste num_classes de acordo com suas necessidades)
    num_classes = 1  # exemplo: 2 classes diferentes de objetos
    recognizer.create_model(num_classes)

    # Treina o modelo
    train_dir = "dataset/train"
    validation_dir = "dataset/validation"
    history = recognizer.train_model(train_dir, validation_dir)

    # Detecta objetos em tempo real
    recognizer.real_time_detection()

if __name__ == "__main__":
    main()

# Created/Modified files during execution:
# - object_model.h5 (modelo treinado)