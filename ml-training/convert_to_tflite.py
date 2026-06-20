import tensorflow as tf
import numpy as np
import os

# --- Настройка путей ---
H5_MODEL_PATH = 'dosypator_model.h5'
TFLITE_MODEL_PATH = 'dosypator_model_quantized.tflite'
DATA_PATH = 'dataset/extracted_features.npy' # Путь к вашим признакам

print(f"--- Загрузка модели Keras и данных для калибровки ---")

# 1. Загрузка Keras модели
model = tf.keras.models.load_model(H5_MODEL_PATH)
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 2. Загрузка признаков (для Representative Dataset)
data = np.load(DATA_PATH, allow_pickle=True).item()
X = data['features']
# Форма X: (N, 94, 40, 1) - должна быть такой, как в обучении, но TFLite требует float32
X = X[..., np.newaxis].astype(np.float32)

# --- Генератор Representative Dataset ---
def representative_dataset_gen():
    # Используем небольшую часть данных (например, 100 примеров) для калибровки
    num_samples = min(100, X.shape[0])
    
    for i in range(num_samples):
        # Преобразование данных к нужному входному формату TFLite
        yield [X[i:i+1]]

# 3. Оптимизация и квантование для ESP32-S3
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Добавляем генератор для калибровки
converter.representative_dataset = representative_dataset_gen

# Целочисленное квантование (Full Integer Quantization)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Указываем, что входные/выходные данные также должны быть преобразованы в int8
converter.inference_input_type = tf.int8  
converter.inference_output_type = tf.int8


# 4. Конвертация
tflite_model = converter.convert()

# 5. Сохранение
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

# Получение размера для оценки
file_size = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024) # МБ

print(f"\n✅ Модель успешно конвертирована и полностью оптимизирована (Квантование).")
print(f"Имя файла: {TFLITE_MODEL_PATH}")
print(f"Размер файла: {file_size:.2f} МБ")
print("---")