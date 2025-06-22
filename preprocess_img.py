import os
import os.path as osp
import numpy as np
import torch
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from argparse import ArgumentParser

from tqdm import tqdm

def preprocess_images(root_dir, output_dir, use_transforms=False, patch_size=256, **kwargs):
    """
    - Ищет в root_dir подпапки вида "21s", "52s" и т.д.
    - Считывает там файлы:
    *_channel_RED.tif, *_channel_GRN.tif, *_channel_BLU.tif
    *class_603.tif, *class_604.tif
    - Склеивает каналы RGB в одну картинку
    - Создаёт единую 2D‑маску: 0 (класс 603), 1 (класс 604), 255 (фон)
    - При use_transforms=True (опционально) можно добавить аугментации (не реализовано в данном примере).
    - Затем режет полученные изображения и маски на патчи 256×256 (или patch_size×patch_size,
    если задать другое значение).
    - Для каждого subdir создаёт подпапку в output_dir, а внутри — папки img и mask.
    - В эти папки сохраняются патчи с именами {subdir}{h}{w}.jpeg,
    где (h,w) — координаты левого верхнего угла патча.
    """
    rescale = kwargs.pop("rescale", 1)
    overlap = kwargs.pop("overlap", 0) // rescale
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Здесь можно объявить любые transforms при use_transforms=True, 
    # но в примере они не реализованы
    transform = None
    
    # Сканируем все подпапки в корне
    for subdir in tqdm(os.listdir(root_dir)):
        full_subdir_path = os.path.join(root_dir, subdir)
    
        if os.path.isdir(full_subdir_path):
            # Составляем ожидаемые пути к файлам
            red_path = os.path.join(full_subdir_path, f"{subdir}_channel_RED.tif")
            grn_path = os.path.join(full_subdir_path, f"{subdir}_channel_GRN.tif")
            blu_path = os.path.join(full_subdir_path, f"{subdir}_channel_BLU.tif")
            c603_path = os.path.join(full_subdir_path, f"{subdir}_class_603.tif")
            c604_path = os.path.join(full_subdir_path, f"{subdir}_class_604.tif")
    
            # Проверяем, что все файлы существуют
            if not all(os.path.exists(p) for p in [red_path, grn_path, blu_path, c603_path, c604_path]):
                print(f"Warning: Not all required files found for {subdir}, skipping.")
                continue
    
            # Считываем каналы изображения
            img_red = np.array(Image.open(red_path))
            img_grn = np.array(Image.open(grn_path))
            img_blu = np.array(Image.open(blu_path))
    
            # Склеиваем в (H, W, 3)
            img = np.stack([img_red, img_grn, img_blu], axis=-1)
    
            # Считываем маски (два канала)
            mask_603 = np.array(Image.open(c603_path))
            mask_604 = np.array(Image.open(c604_path))
    
            # Формируем одну итоговую маску [H, W] (0,1,255)
            H, W, _ = img.shape
            final_mask = np.full((H, W), 0, dtype=np.uint8)  # фон = 255
            final_mask[mask_603 > 0] = 1
            final_mask[mask_604 > 0] = 2
    
            # Здесь можно применить аугментации, если use_transforms=True
            if rescale > 1:
                torch_img = torch.from_numpy(img.astype(float)).moveaxis((0, 1, 2), (1, 2, 0)).unsqueeze(0)
                torch_mask = torch.from_numpy(final_mask.astype(float)).unsqueeze(0).unsqueeze(0)
                aug_img = (F.interpolate(torch_img, size=(img.shape[0] // rescale, img.shape[1] // rescale), mode="bilinear", align_corners=False)).numpy().astype(np.uint8).squeeze(0)
                aug_mask = (F.interpolate(torch_mask, size=(img.shape[0] // rescale, img.shape[1] // rescale), mode="nearest")).numpy().astype(np.uint8).squeeze(0).squeeze(0)
                aug_img = np.moveaxis(aug_img, (0, 1, 2), (2, 0, 1))
            else:
                aug_img = img
                aug_mask = final_mask
            H, W, _ = aug_img.shape
            # Создаём структуру директорий: output_dir/subdir/{img,mask}
            out_subdir = os.path.join(output_dir, subdir)
            img_out_dir = os.path.join(out_subdir, "img")
            mask_out_dir = os.path.join(out_subdir, "mask")
            os.makedirs(img_out_dir, exist_ok=True)
            os.makedirs(mask_out_dir, exist_ok=True)

            # Image.fromarray(aug_img.astype(np.uint8)).save(os.path.join(img_out_dir, f"{subdir}.png"), format="PNG")
            # Image.fromarray(aug_mask.astype(np.uint8)).save(os.path.join(mask_out_dir, f"{subdir}.png"), format="PNG")
    
            # Разрезаем на патчи 256×256
            step = patch_size - overlap
            for i in range(0, H, step):
                for j in range(0, W, step):
                    patch_img = aug_img[i:i+patch_size, j:j+patch_size, :]
                    patch_mask = aug_mask[i:i+patch_size, j:j+patch_size]
    
                    # Если хотите пропускать неполные патчи (в случае, 
                    # когда H или W не кратны 256), то можно сделать проверку:
                    if patch_img.shape[0] < patch_size or patch_img.shape[1] < patch_size:
                        continue
                    # if (patch_mask == 2).mean() >= 0.99 or (patch_mask == 0).mean() >= 0.99 or (patch_mask == 1).mean() >= 0.99:
                    #     continue
    
                    patch_img_name = f"{subdir}_{i}_{j}.png"
                    patch_mask_name = f"{subdir}_{i}_{j}.png"
    
                    patch_img_path = os.path.join(img_out_dir, patch_img_name)
                    patch_mask_path = os.path.join(mask_out_dir, patch_mask_name)
    
                    Image.fromarray(patch_img.astype(np.uint8)).save(patch_img_path, format="PNG")
                    Image.fromarray(patch_mask.astype(np.uint8)).save(patch_mask_path, format="PNG")
    
            print(f"Saved preprocessed patches for {subdir} to:\n  {img_out_dir}\n  {mask_out_dir}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--rescale", default=1, type=int)
    parser.add_argument("--overlap", default=0, type=int)
    args = parser.parse_args()
    out_dir = f"/home/neos/vkr/{args.out_dir}"
    if args.rescale > 1:
        out_dir += f"_x{args.rescale}"
    preprocess_images(root_dir="/home/neos/vkr/val", output_dir=osp.join(out_dir, "val_prep"), use_transforms=False, patch_size=256 // args.rescale, **vars(args))
    preprocess_images(root_dir="/home/neos/vkr/train", output_dir=osp.join(out_dir, "train_prep"), use_transforms=False, patch_size=256 // args.rescale, **vars(args))
