import numpy as np
import os
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from argparse import ArgumentParser
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    disk,
    opening,
    closing
)

def dummy_model_predict(patches_batch):
    """
    Пример (заглушка) функции, эмулирующей работу модели.
    Предположим, что у нас монохромная сегментация (один канал на выходе).
    patches_batch: numpy-массив формы (N, patch_h, patch_w, C) или (N, patch_h, patch_w)
    где N - количество патчей.
    Возвращаем массив той же формы, что и вход (без учёта канала, если он есть),
    но в коде ниже зададим (N, patch_h, patch_w) для удобства.
    """
    # Для примера вернём просто нули
    # В реальной ситуации здесь вызывается, например, model.predict(...) (Keras/TF/PyTorch и т.д.)
    num_patches = patches_batch.shape[0]
    patch_h = patches_batch.shape[1]
    patch_w = patches_batch.shape[2]
    return np.zeros((num_patches, patch_h, patch_w), dtype=np.float32)


def sliding_window_prepare_patches(image, patch_size=256, overlap=32):
    """
    Разбивает большое изображение image на перекрывающиеся патчи,
    возвращает список самих патчей patches_list и список координат coords_list.
    image: np.array формы (H, W, C) или (H, W)
    patch_size: размер стороны вырезаемого патча
    overlap: перекрытие в пикселях
    
    Возврат:
    patches_list: список np.array патчей
    coords_list: список кортежей (y, x, y_end, x_end) для каждого патча
    """
    H, W = image.shape[:2]
    step = patch_size - overlap
    
    patches_list = []
    coords_list = []
    x_max, y_max = 0, 0
    for y in range(0, H - patch_size, step):
        for x in range(0, W - patch_size, step):
            y_end, x_end = y + patch_size, x + patch_size
            y_max = max(y_end, y_max)
            x_max = max(x_end, x_max)
            
            patch = image[y:y_end, x:x_end]
            patches_list.append(patch)
            coords_list.append((y, x, y_end, x_end))
    
    return patches_list, coords_list, y_max, x_max


def sliding_window_reconstruct_multiclass(predictions, coords_list, out_shape, num_classes=3):
    """
    Собирает итоговую сегментацию из списка многоканальных предсказаний (многоклассных),
    используя усреднение в зоне перекрытия.
    predictions:  np.array формы (N, patch_h, patch_w, num_classes),
                  где N = количество патчей.
    coords_list:  список кортежей (y, x, y_end, x_end),
                  координаты каждого патча в исходном большом изображении.
    out_shape:    (H, W) - желаемый размер итоговой карты (без учёта кол-ва каналов).
    
    Возвращает:
    final_mask:   np.array формы (H, W), где в каждом пикселе - индекс класса (argmax).
    """
    H, W = out_shape
    
    # result_probs накопит суммы вероятностей (или логитов) по всем патчам
    result_probs = np.zeros((H, W, num_classes), dtype=np.float32)
    # count_map скажет, сколько патчей покрывали конкретный пиксель
    count_map = np.zeros((H, W, 1), dtype=np.float32)

    predictions = F.one_hot(torch.as_tensor(predictions, dtype=torch.long), num_classes).numpy()
    
    for i, (y, x, y_end, x_end) in tqdm(enumerate(coords_list)):
        # Суммируем в зонах наложения
        result_probs[y:y_end, x:x_end, :] += predictions[i]
        count_map[y:y_end, x:x_end, 0] += 1.0
    print(result_probs.shape, count_map.shape)
    # Усредняем накопленные вероятности
    avg_probs = result_probs / (count_map + 1e-8)
    # Берём argmax по каналу классов, получая «метку» класса (0 .. num_classes-1)
    final_mask = np.argmax(avg_probs, axis=-1)
    print(np.unique(final_mask, return_counts=True))
    
    return final_mask


def segment_large_image_once(image, predictor, patch_size=256, overlap=32, num_classes=3):
    """
    Основная функция:
    1) Разбивает большое изображение на патчи (overlapping-стратегия).
    2) Формирует из них батч и "одновременно" передаёт в модель.
    3) Собирает предсказания обратно в одно большое изображение.
    Возвращаем итоговую сегментированную карту (одно канальное np.array).
    """
    # 1) Разбиваем на патчи
    patches_list, coords_list, y_max, x_max = sliding_window_prepare_patches(
        image, patch_size=patch_size, overlap=overlap
    )
    W, H = image.shape[:2]
    image = image[0:y_max,0:x_max]
    
    # Преобразуем список патчей в единый батч (например, (N, patch_size, patch_size, C) )
    # NOTE: если исходный image содержит (H, W) - уже монохром, или (H, W, C) - цветное
    # Для простоты примем, что это RGB или одно канальное (тогда делаем одинаковые формы)
    
    # Выясним, есть ли канал:
    if len(image.shape) == 3:  # (H, W, C)
        # Считаем, что C=3 (или другое число)
        batch = np.stack([p for p in patches_list], axis=0)  # (N, h, w, C)
    else:  # (H, W)
        # Одноканальное
        batch = np.stack([p for p in patches_list], axis=0)  # (N, h, w)
    print("Done slicing")
    # 2) "Прогоняем" все патчи через модель одним батчем
    predictions = predictor(batch)
    # Ожидаем, что predictions имеет форму (N, patch_h, patch_w) (если 1 канал)
    print("Done predicting")
    # 3) Делаем обратную сборку
    final_mask = sliding_window_reconstruct_multiclass(
        predictions, coords_list, image.shape[:2], num_classes=num_classes
    ).astype(np.uint8)
    print("Done reconstructing")
    final_mask = postprocess_segmentation(final_mask)
    final_mask = ((final_mask / (num_classes - 1)) * 255).astype(np.uint8)
    
    return final_mask


def postprocess_segmentation(seg_mask,
    min_size=100,
    hole_size=100,
    selem_size=3):
    """
    Выполняет морфологические операции и удаляет мелкие компоненты
    (а при желании и «дыры») для многоклассовой сегментации.
    Параметры:
    -----------
    seg_mask : np.array формы (H, W), 
               значения целочисленные (например, {0, 1, 2}).
    min_size : int, 
               минимальный размер связной компоненты,
               меньше которого объекты будут удалены.
    hole_size : int,
                максимальная площадь дыр (полостей), которые надо заполнять.
    selem_size : int,
                 радиус структурирующего элемента (диска) для морф. операций.
    
    Возвращает:
    -----------
    final_mask : np.array формы (H, W), 
                 постобработанная многоклассовая маска (значения классов).
    """
    # Инициализируем итоговую маску
    final_mask = np.zeros_like(seg_mask)
    
    # Получаем список уникальных классов (например, [0, 1, 2])
    classes = np.unique(seg_mask)
    
    # Для каждого класса проводим морфологическую обработку по отдельности
    for c in classes:
        # Бинарная маска для текущего класса
        bin_mask = (seg_mask == c)
        
        # Удаляем мелкие объекты
        # (связные компоненты, содержащие менее чем `min_size` пикселей)
        bin_mask = remove_small_objects(bin_mask, min_size=min_size)
        
        # При необходимости можно удалять и мелкие "дыры" внутри объектов
        bin_mask = remove_small_holes(bin_mask, area_threshold=hole_size)
        
        # Сглаживание краёв (могут быть opening/closing — открытие/закрытие)
        bin_mask = opening(bin_mask, disk(selem_size))
        bin_mask = closing(bin_mask, disk(selem_size))
        
        # Записываем класс c в final_mask там, где bin_mask=True
        final_mask[bin_mask] = c
    
    return final_mask



from transformers.modeling_outputs import SemanticSegmenterOutput
import safetensors
from typing import Optional, Union, Tuple

class ModelWrapper(torch.nn.Module):
    def __init__(self, segformer, patch_size=256, accepts_loss_kwargs=False):
        super().__init__()
        self.main = segformer
        self.accepts_loss_kwargs = accepts_loss_kwargs
        self.patch_size = patch_size
        # self.dice_loss = DiceLoss()

    # def forward(self, *args, **kwargs):
    #     outputs = self.main(*args, **kwargs)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SemanticSegmenterOutput]:
        outputs = self.main(pixel_values, labels=labels, output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states, return_dict=return_dict)
        loss, logits_val = outputs.loss, outputs.logits
        upsampled_val_logits = torch.nn.functional.interpolate(
        logits_val, size=(self.patch_size, self.patch_size), mode="bilinear", align_corners=False
        )
        # loss += 0.5 * self.dice_loss(upsampled_val_logits, labels)
        return SemanticSegmenterOutput(loss=loss, logits=upsampled_val_logits)

class SegFormerPredictor:
    def __init__(self, checkpoint):
        # checkpoint_hf = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
        checkpoint_hf = "nvidia/segformer-b0-finetuned-ade-512-512"
        
        self.image_processor = SegformerImageProcessor.from_pretrained(checkpoint_hf, size={"height": PATCH_SIZE, "width": PATCH_SIZE}, do_reduce_labels=False)
        
        self.model = ModelWrapper(SegformerForSemanticSegmentation.from_pretrained(checkpoint_hf, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True), PATCH_SIZE)
        self.model.load_state_dict(safetensors.torch.load_file(os.path.join(checkpoint, "model.safetensors")))
        self.model.to(device)
        self.model.eval()

    def __call__(self, batch):
        inputs = self.image_processor([x for x in batch], return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits.cpu().numpy()
        preds = np.argmax(logits, axis=1).astype(np.uint8)
        print(np.unique(preds, return_counts=True))
        return preds


from transformers import MaskFormerImageProcessor, MaskFormerForInstanceSegmentation
import torch
import numpy as np
import safetensors

class MaskformerPredictor:
    def __init__(self, checkpoint):
        checkpoint_hf = "facebook/maskformer-swin-base-ade"
        # Загружаем процессор и модель
        self.image_processor = MaskFormerImageProcessor.from_pretrained(
            checkpoint_hf,
            # Подгоняем входное изображение под PATCH_SIZE
            size={"height": PATCH_SIZE, "width": PATCH_SIZE},
        )
        self.model = MaskFormerForInstanceSegmentation.from_pretrained(
            checkpoint_hf,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
        )
        self.model.load_state_dict(safetensors.torch.load_file(os.path.join(checkpoint, "model.safetensors")))
        self.model.to(device)
        self.model.eval()

    def __call__(self, batch):
        """
        Аналог функции segformer_predictor, но для MaskFormer.
        Здесь добавляем постобработку через image_processor.post_process_instance_segmentation.
        """
    
        
        # Готовим входные данные для модели
        inputs = self.image_processor(batch, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # В MaskFormer также есть метод постобработки,
        # который может вернуть семантическую карту:
        decoded_preds = self.image_processor.post_process_semantic_segmentation(
            outputs=outputs,
            target_sizes=[(PATCH_SIZE, PATCH_SIZE)] * len(batch)
        )
        # decoded_preds – список словарей [{ "segmentation": Tensor(H, W) }, ...]

        print(decoded_preds[0].shape)
        preds = np.stack([
            decoded.cpu().numpy().astype(np.uint8)
            for decoded in decoded_preds
        ], axis=0)
        
        print(np.unique(preds, return_counts=True))
        return preds




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--step", type=int)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--rescale", type=int, default=1)
    parser.add_argument("--format", type=str, default="tif")
    args = parser.parse_args()
    print(os.getenv("CUDA_VISIBLE_DEVICES"))
    NUM_CLASSES = 3
    rescale = args.rescale
    out_dir = args.out_dir
    overlap = args.overlap
    step = args.step
    model_type = args.model.lower()
    
    PATCH_SIZE = 256 // rescale
    overlap = overlap // rescale
    device="cuda"
    checkpoint = f'{out_dir}/checkpoints/checkpoint-{step}'

    assert model_type in ["segformer", "maskformer", "pspnet", "deeplab"]

    if model_type == "segformer":
        model_predictor = SegFormerPredictor(checkpoint)
    elif model_type == "maskformer":
        model_predictor = MaskformerPredictor(checkpoint)

    # Папка с валидационными данными
    # При rescale=1 -> data_entire/val_prep, а иначе data_entire_x{rescale}/val_prep
    images_dir = (
        "data_entire" if rescale == 1 else f"data_entire_x{rescale}"
    )
    val_dir = os.path.join(images_dir, "val_prep")
    
    # Папка, куда складываем наши результаты
    out_subdir = f'{out_dir}/eval_predict/eval-{step}'
    os.makedirs(out_subdir, exist_ok=True)
    
    # Перебираем все подпапки, каждая подпапка соответствует одному снимку: image_name
    for image_name in os.listdir(val_dir):
        full_img_dir = os.path.join(val_dir, image_name, "img")
        full_mask_dir = os.path.join(val_dir, image_name, "mask")
    
        # Файл с исходным снимком: {image_name}/img/{image_name}.png
        img_path = os.path.join(full_img_dir, f"{image_name}.png")
        # Файл с истинной маской: {image_name}/mask/{image_name}.png
        mask_path = os.path.join(full_mask_dir, f"{image_name}.png")
    
        # Проверим, что пути действительно существуют
        if not (os.path.isfile(img_path) and os.path.isfile(mask_path)):
            # Пропускаем, если вдруг не совпадает структура
            continue
    
        # Загружаем исходное изображение (для сегментации)
        img_pil = Image.open(img_path)
        img_np = np.asarray(img_pil)
    
        # Запускаем сегментацию
        final_mask = segment_large_image_once(
            img_np,
            model_predictor,
            PATCH_SIZE,
            overlap=overlap
        )
        print("Final mask shape:", final_mask.shape)
    
        # Загружаем истинную маску (для визуального сравнения)
        mask_pil = Image.open(mask_path)
        mask_np = np.asarray(mask_pil)
        mask_np = ((mask_np / (NUM_CLASSES - 1)) * 255).astype(np.uint8)
    
        # Склеиваем финальную маску, истинную маску и исходное изображение горизонтально.
        # Для удобства переведём все три в RGB (по необходимости).
        # 1) Предсказанная маска (final_mask)
        # Обычно она одноканальная. Сделаем 3 канала одинаковых:
        final_rgb = np.stack([final_mask]*3, axis=-1)
    
        # 2) Истинная маска
        if len(mask_np.shape) == 2:
            # одноканальная
            mask_rgb = np.stack([mask_np]*3, axis=-1)
        else:
            # уже 3 канала
            mask_rgb = mask_np
    
        # 3) Исходный снимок (уже может быть RGB)
        if len(img_np.shape) == 2:
            # одноканальное
            img_rgb = np.stack([img_np]*3, axis=-1)
        else:
            # если 3 канала
            img_rgb = img_np
    
        # Проверяем совместимость по высоте (если нужно - ресайзим один из вариантов, 
        #     но здесь предполагается, что все H совпадают)
        Hf, Wf = final_rgb.shape[:2]
        Hm, Wm = mask_rgb.shape[:2]
        Hi, Wi = img_rgb.shape[:2]
        # Если необходимо, можно добавить логику resize:
        # например, чтобы все имели одну высоту:
        # common_height = min(Hf, Hm, Hi)
        # и далее ресайз PIL-ом или cv2.
        # Но впрямую предполагается одинаковый H.
        mask_rgb = mask_rgb[:Hf, :Wf]
        img_rgb = img_rgb[:Hf, :Wf]
    
        # Склеиваем:
        combined = np.concatenate([final_rgb, mask_rgb, img_rgb], axis=1)
    
        # Сохраняем результат
        result_pil = Image.fromarray(combined)
        out_path = os.path.join(out_subdir, f"pred_{image_name}.{args.format}")
        result_pil.save(out_path)
    
        print(f"Сохранили результат для {image_name} в {out_path}")