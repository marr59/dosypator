import os
import librosa
import numpy as np
import warnings
from tqdm import tqdm # Для красивой полосы прогресса

# Игнорируем предупреждения Librosa о старых функциях
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Глобальные параметры ---
# Сколько признаков (MFCC) извлекаем
N_MFCC = 40
# Частота дискретизации, с которой мы конвертировали файлы
SAMPLE_RATE = 16000 
# Продолжительность сегмента в секундах (стандарт для аудио ML)
SEGMENT_DURATION = 3 

# Количество сэмплов в одном сегменте
SAMPLES_PER_SEGMENT = SAMPLE_RATE * SEGMENT_DURATION

# --- Настройка путей ---
# Корневая папка с WAV-файлами
AUDIO_ROOT = 'dataset'
# Файл для сохранения итоговых признаков
OUTPUT_FILE = 'dataset/extracted_features.npy' 

# Список папок, которые нужно обработать: (имя папки, метка класса)
# Метка 1 = Плач, Метка 0 = Шум
FOLDERS_TO_PROCESS = [
    ('wav_cries', 1),
    ('wav_noise', 0)
]

all_features = [] # Список для хранения всех извлеченных MFCC-признаков
all_labels = []   # Список для хранения всех меток (1 или 0)

print("--- Начат процесс извлечения признаков (MFCC) ---")

for folder_name, label in FOLDERS_TO_PROCESS:
    folder_path = os.path.join(AUDIO_ROOT, folder_name)
    print(f"\nОбработка папки: {folder_path} (Метка: {label})")
    
    # tqdm делает красивую полосу прогресса
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    
    for filename in tqdm(file_list, desc=f"Конвертация {folder_name}"):
        file_path = os.path.join(folder_path, filename)
        
        try:
            # 1. Загрузка аудио
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            
            # 2. Сегментация (разрезаем длинный файл)
            num_segments = len(y) // SAMPLES_PER_SEGMENT
            
            for i in range(num_segments):
                start = i * SAMPLES_PER_SEGMENT
                end = start + SAMPLES_PER_SEGMENT
                
                # Извлекаем сегмент
                segment = y[start:end]
                
                # 3. Извлечение MFCC
                # N_MFCC=40 - это богатое описание звука
                mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
                
                # mfccs имеет форму (N_MFCC, num_frames). 
                # Мы транспонируем ее и сохраняем
                all_features.append(mfccs.T)
                all_labels.append(label)

        except Exception as e:
            print(f"\n❌ Ошибка обработки файла {filename}: {e}")
            
print("\n--- Сохранение данных ---")
# Преобразуем списки в массивы NumPy
X = np.array(all_features)
y = np.array(all_labels)

# Сохраняем данные в файл .npy
np.save(OUTPUT_FILE, {'features': X, 'labels': y})

print(f"✅ Данные сохранены в: {OUTPUT_FILE}")
print(f"Итоговое количество обучающих примеров (сегментов): {len(X)}")
print(f"Форма данных X (Признаки): {X.shape}") 
print("---")
