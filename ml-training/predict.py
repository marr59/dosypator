import librosa
import numpy as np
import tensorflow as tf
import os
import warnings

# Игнорируем предупреждения
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Настройка путей и параметров ---
MODEL_PATH = 'dosypator_model.h5'
AUDIO_PATH = 'test_cry_check.wav' # Используем ваш тестовый WAV-файл

# Параметры должны соответствовать параметрам обучения!
SAMPLE_RATE = 16000
SEGMENT_DURATION = 3
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SEGMENT_DURATION
N_MFCC = 40

# --- 1. Загрузка модели ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Модель успешно загружена: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Ошибка при загрузке модели: {e}")
    exit()

# --- 2. Функция для предобработки аудио ---
def preprocess_audio(file_path):
    # Загружаем аудио (должно быть 16000 Гц, как при конвертации)
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Берем только первый сегмент для простоты тестирования
    # В реальном боте мы бы анализировали несколько сегментов
    segment = y[0:SAMPLES_PER_SEGMENT]

    if len(segment) < SAMPLES_PER_SEGMENT:
        print(f"❌ Файл {os.path.basename(file_path)} слишком короткий (менее {SEGMENT_DURATION} сек). Пропускаем.")
        return None

    # Извлечение MFCC (аналогично процессу обучения)
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)

    # Добавляем оси для соответствия форме CNN: (1, 94, 40, 1)
    mfccs = mfccs.T[np.newaxis, ..., np.newaxis] 

    return mfccs

# --- 3. Выполнение предсказания ---
print(f"--- Тестирование аудиофайла: {AUDIO_PATH} ---")
features = preprocess_audio(AUDIO_PATH)

if features is not None:
    # Делаем предсказание
    prediction = model.predict(features)

    # Получаем вероятность класса "Плач"
    probability = prediction[0][0]

    # Устанавливаем порог (Threshold) 0.5: если больше 0.5, то это плач
    is_cry = probability > 0.5

    print("\nРезультат:")
    print(f"Вероятность 'Плача': {probability:.4f}")

    if is_cry:
        print("👶 Классифицировано: ПЛАЧ (POSITIVE)")
    else:
        print("🎧 Классифицировано: ШУМ (NEGATIVE)")
else:
    print("Не удалось выполнить предсказание.")
