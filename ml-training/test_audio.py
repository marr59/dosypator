import numpy as np
import librosa
import matplotlib.pyplot as plt
import os

# --- Параметры ---
# Замените этот путь на путь к вашему аудиофайлу (например, mp3 или wav)
# Если у вас нет файла, вы можете найти короткий пример (например, .wav) в интернете.
# Для простоты можно использовать любой небольшой аудиофайл.
AUDIO_FILE_PATH = "example_cry.wav" # <--- ЗАМЕНИТЕ НА ВАШ ФАЙЛ

# Загружаем аудиофайл с помощью Librosa
print(f"Загрузка аудиофайла: {AUDIO_FILE_PATH}...")
try:
    # sr=None - загрузить с исходной частотой дискретизации
    y, sr = librosa.load(AUDIO_FILE_PATH, sr=None)
    print(f"Аудио загружено. Длительность: {len(y)/sr:.2f} сек, Частота дискретизации (sr): {sr} Гц")

    # --- Создание Спектрограммы (MFCC) ---
    # Мы используем MFCCs (Мел-частотные кепстральные коэффициенты), 
    # которые обычно используются для распознавания звука и речи.
    print("Расчет MFCC...")
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    # Визуализация (опционально, но полезно для проверки Matplotlib)
    print("Визуализация...")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    # plt.show() # Пока закомментируем, чтобы не было проблем с отображением в некоторых средах

    # --- Проверка TensorFlow (простое действие) ---
    import tensorflow as tf
    # Простейшее действие с тензором
    hello = tf.constant('Проверка TensorFlow: Успешно!')
    print(tf.constant('Проверка TensorFlow: Успешно!'))

except FileNotFoundError:
    print(f"\n❌ Ошибка: Файл '{AUDIO_FILE_PATH}' не найден.")
    print("Пожалуйста, замените 'example_cry.wav' на фактический путь к вашему аудиофайлу.")
except Exception as e:
    print(f"\n❌ Произошла ошибка при загрузке или обработке аудио: {e}")
