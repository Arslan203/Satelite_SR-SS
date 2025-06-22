# Репозиторий для обучения моделей сегментации снимков (SS) и повышения разрешения (SR)
## Описание
Данный репозиторий содержит инструменты для обучения моделей машинного обучения для двух задач:

Сегментация снимков (SS) - сегментация спутниковых/аэрофотоснимков
Повышение разрешения (SR) - улучшение качества изображений
## Требования
Python 3.8+
CUDA-совместимый GPU (рекомендуется)
Hugging Face токен для загрузки моделей
## Установка
```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```
## Сегментация снимков (SS)
# 1. Подготовка данных
Организуйте данные в следующей структуре:

```kotlin
data/
├── subdir1/
│   ├── subdir1_channel_RED.tif
│   ├── subdir1_channel_GRN.tif
│   ├── subdir1_channel_BLU.tif
│   ├── subdir1_class_603.tif
│   └── subdir1_class_604.tif
├── subdir2/
│   ├── subdir2_channel_RED.tif
│   ├── subdir2_channel_GRN.tif
│   ├── subdir2_channel_BLU.tif
│   ├── subdir2_class_603.tif
│   └── subdir2_class_604.tif
└── ...
```
Где:

{subdir}_channel_RED.tif - красный канал снимка
{subdir}_channel_GRN.tif - зеленый канал снимка
{subdir}_channel_BLU.tif - синий канал снимка
{subdir}_class_603.tif - маска сегментации для класса 603
{subdir}_class_604.tif - маска сегментации для класса 604
# 2. Предобработка изображений
Запустите скрипт предобработки с масштабированием:

```bash
python preprocess_img.py --rescale 1
```
Дополнительные параметры:

```bash
# Указать входную директорию
python preprocess_img.py --rescale 1 --input_dir ./data

# Указать выходную директорию
python preprocess_img.py --rescale 1 --output_dir ./processed_data

# Полный набор параметров
python preprocess_img.py --rescale 1 --input_dir ./data --output_dir ./processed_data
```
# 3. Обучение модели SegFormer
Получите токен Hugging Face:

Зарегистрируйтесь на huggingface.co
Создайте токен в настройках аккаунта
Запустите Jupyter Notebook:

```bash
jupyter notebook SegFormer1.ipynb
```
В notebook'е укажите ваш HF токен
Выполните все ячейки notebook'а для запуска обучения
# 4. Валидация
Во время обучения автоматически запускается скрипт reconstructionv3.py, который выполняет сегментацию на валидационной выборке.

Для ручного запуска валидации:
```bash
python reconstructionv3.py --model_path ./checkpoints/best_model --val_dir ./processed_data/val
```

Структура репозитория
```bash
.
├── preprocess_img.py          # Скрипт предобработки изображений
├── SegFormer1.ipynb          # Notebook для обучения SegFormer
├── reconstructionv3.py       # Скрипт валидации и реконструкции
├── requirements.txt          # Зависимости проекта
├── data/                     # Директория с исходными данными
├── processed_data/           # Директория с обработанными данными
├── checkpoints/              # Сохраненные модели
└── results/                  # Результаты сегментации
```
Примечания
1. Убедитесь, что все .tif файлы имеют правильное именование
2. Рекомендуется использовать GPU для ускорения обучения
3. Процесс обучения может занять несколько часов в зависимости от объема данных
4. Результаты валидации сохраняются автоматически в директорию results/
