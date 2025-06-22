import os
import argparse
import numpy as np
from PIL import Image

def to_class_index(gray_2d: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """
    Преобразует 8-битную серую картинку (0..255) в индекс классов (0..num_classes-1).
    Предположим, что у нас ступенчатое кодирование: 0, 128, 255 (для 3 классов).
    Тогда используем: класс = round((pixel_value/255)*(num_classes-1)).
    Если у вас иной способ кодирования, подстройте под него.
    """
    float_map = gray_2d.astype(np.float32) / 255.0
    class_map = np.round(float_map * (num_classes - 1)).astype(np.uint8)
    return class_map

def compute_confusion_matrix(true_mask: np.ndarray, pred_mask: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Считает матрицу неточетностей (confusion matrix) размера (num_classes, num_classes).
    [i, j] = число пикселей, где истинный класс = i, предсказанный = j.
    """
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    t_flat = true_mask.flatten()
    p_flat = pred_mask.flatten()
    valid = (t_flat >= 0) & (t_flat < num_classes)
    t_flat = t_flat[valid]
    p_flat = p_flat[valid]
    for i in range(len(t_flat)):
        confusion[t_flat[i], p_flat[i]] += 1
    return confusion

def compute_iou_from_confusion(confusion: np.ndarray):
    """
    Считает для confusion matrix (num_classes x num_classes) IoU по классам и средний IoU (mIoU).
    IoU(c) = TP(c) / (TP(c) + FP(c) + FN(c)).
    """
    num_classes = confusion.shape[0]
    iou_per_class = []
    for c in range(num_classes):
        tp = confusion[c, c]
        fp = np.sum(confusion[:, c]) - tp
        fn = np.sum(confusion[c, :]) - tp
        denom = (tp + fp + fn)
        iou_c = tp / denom if denom != 0 else 0.0
        iou_per_class.append(iou_c)
    mean_iou = np.mean(iou_per_class)
    return np.array(iou_per_class), mean_iou

def main():
    parser = argparse.ArgumentParser(description="Measure mIoU for side-by-side preds & masks")
    parser.add_argument("--in_dir", type=str, required=True, help="Папка, где лежат склеенные изображения (pred|mask|image).")
    parser.add_argument("--num_classes", type=int, default=3, help="Число классов в сегментации.")
    parser.add_argument("--format", type=str, default="tif", help="Расширение файлов в in_dir (png/tif и т.п.).")
    args = parser.parse_args()
    in_dir = args.in_dir
    num_classes = args.num_classes
    
    # Инициируем общую confusion matrix
    global_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # Перебираем все файлы, оканчивающиеся на указанное расширение
    files = [f for f in os.listdir(in_dir) if f.endswith(f".{args.format}")]
    files.sort()
    
    print(f"Найдено {len(files)} файлов для обработки в {in_dir}...")
    
    for filename in files:
        path = os.path.join(in_dir, filename)
        # Загружаем склеенное изображение
        combined_pil = Image.open(path)
        combined_np = np.array(combined_pil)  # shape (H, 3W, 3) ожидается
    
        if combined_np.ndim != 3 or combined_np.shape[2] not in [1, 3, 4]:
            print(f"Пропускаю {filename}: некорректная форма изображения {combined_np.shape}")
            continue
    
        H, total_W, C = combined_np.shape
        part_W = total_W // 3  # ширина одного блока
    
        # Разрезаем на три части
        #  1) Предсказанная маска (слева)
        pred_rgb = combined_np[:, :part_W, :]  # (H, part_W, C)
        #  2) Истинная маска (по центру)
        true_rgb = combined_np[:, part_W:2*part_W, :]
        #  3) Исходный снимок (справа) - для IoU не нужен, но мы его не трогаем
    
        # Переводим предсказанную маску в градации серого (т.к. она повторялась по каналам)
        # Берём, например, красный канал (или среднее) — т.к. они совпадают.
        pred_gray = pred_rgb[:, :, 0]
        # Аналогично для истинной маски
        true_gray = true_rgb[:, :, 0]
    
        # Превращаем в индексы классов [0..num_classes-1],
        # если у вас действительно 0,128,255 для 3 классов
        pred_mask_classes = to_class_index(pred_gray, num_classes=num_classes)
        true_mask_classes = to_class_index(true_gray, num_classes=num_classes)
    
        # Считаем confusion matrix для этого изображения
        cm = compute_confusion_matrix(true_mask_classes, pred_mask_classes, num_classes)

        # Считаем IoU для этого файла
        iou_per_class, mean_iou = compute_iou_from_confusion(cm)

        # Выводим IoU для данного изображения
        print(f"\nФайл: {filename}")
        for c in range(num_classes):
            print(f"  Класс {c} IoU: {iou_per_class[c]:.4f}")
        print(f"  Mean IoU: {mean_iou:.4f}")
        global_confusion += cm
    
    print("\n=== Результаты ===")
    iou_per_class, mIoU = compute_iou_from_confusion(global_confusion)
    for c in range(num_classes):
        print(f"Класс {c} IoU: {iou_per_class[c]:.4f}")
    print(f"Mean IoU: {mIoU:.4f}")


if __name__ == "__main__":
    main()