import os
import re
from PIL import Image
from argparse import ArgumentParser

def restore_3_images_from_patches(patches_folder, output_folder, out_format):
    """
    1) Считывает все PNG-файлы в папке patches_folder, которые имеют формат:
    '{base_name}{x}{y}.png'
    где:
    - {base_name} -- имя набора (исходного большого изображения),
    - x, y -- координаты "одиночного" патча (в под-изображении).
    2) Каждый такой файл фактически хранит «тройной» патч по горизонтали,
    из которого нужно вырезать 3 под-патча.
    Для i-го (0..2) под-патча координы в итоговом под-изображении такие же (x, y),
    т.е. мы не делаем никаких сдвигов на i * subW.
    3) Собираем 3 под-изображения (Image #0, #1, #2) для каждого base_name
    и склеиваем их бок о бок в один файл '{base_name}.png'.
    """
    # Регулярка для извлечения base_name, x, y из имени файла
    pattern = re.compile(r'^(.*)_(\d+)_(\d+)\.png$')

    # Структура данных для под‐изображений:
    # images_data[base_name] = [
    #     { 'patches': [], 'max_width': 0, 'max_height': 0 },  # под-изображение #0
    #     { 'patches': [], 'max_width': 0, 'max_height': 0 },  # под-изображение #1
    #     { 'patches': [], 'max_width': 0, 'max_height': 0 }   # под-изображение #2
    # ]
    images_data = {}

    # Шаг 1. Считываем все «тройные» патчи из папки
    for filename in os.listdir(patches_folder):
        if not filename.endswith('.png'):
            continue
        
        match = pattern.match(filename)
        if not match:
            # Если имя файла не подходит под шаблон "{base_name}_{x}_{y}.png", пропустим
            continue
        
        base_name, x_str, y_str = match.groups()
        x, y = int(y_str), int(x_str)

        patch_path = os.path.join(patches_folder, filename)
        patch_img = Image.open(patch_path)
        W, H = patch_img.size

        # Подготовим структуру для base_name, если не создана
        if base_name not in images_data:
            images_data[base_name] = [
                {'patches': [], 'max_width': 0, 'max_height': 0},
                {'patches': [], 'max_width': 0, 'max_height': 0},
                {'patches': [], 'max_width': 0, 'max_height': 0}
            ]
        
        # Ширина каждого «одиночного» под‐патча
        subW = W // 3
        
        # Разрезаем наш тройной патч на 3 равные части
        for i in range(3):
            left   = i * subW
            top    = 0
            right  = (i + 1) * subW
            bottom = H

            sub_patch_img = patch_img.crop((left, top, right, bottom))
            
            # Координаты размещения sub_patch в под-изображении i:
            # Согласно условию, x,y уже заданы для одиночных патчей,
            # поэтому не нужно делать сдвиг на i * subW — оставляем (x, y).
            local_x = x
            local_y = y

            images_data[base_name][i]['patches'].append((local_x, local_y, sub_patch_img))

            # Обновим границы (чтобы понять, какой размер понадобится для сборки)
            images_data[base_name][i]['max_width'] = max(
                images_data[base_name][i]['max_width'],
                local_x + subW
            )
            images_data[base_name][i]['max_height'] = max(
                images_data[base_name][i]['max_height'],
                local_y + H
            )

    # Шаг 2. Для каждого base_name восстанавливаем три под-изображения и склеиваем их
    os.makedirs(output_folder, exist_ok=True)

    for base_name, three_subimages in images_data.items():
        # three_subimages -- это список из 3 словарей (для i=0,1,2)
        final_sub_imgs = []

        # Собираем последовательно три под-изображения
        for i, subimage_data in enumerate(three_subimages):
            max_w = subimage_data['max_width']
            max_h = subimage_data['max_height']

            # Создаём пустой холст нужного размера
            sub_final = Image.new('RGB', (max_w, max_h))

            # «Наклеиваем» все под-патчи
            for (lx, ly, patch) in subimage_data['patches']:
                sub_final.paste(patch, (lx, ly))
            
            final_sub_imgs.append(sub_final)

        # Теперь склеиваем три под-изображения бок о бок
        w0, h0 = final_sub_imgs[0].size
        w1, h1 = final_sub_imgs[1].size
        w2, h2 = final_sub_imgs[2].size

        total_width = w0 + w1 + w2
        total_height = max(h0, h1, h2)  # если высоты ничем не ограничены

        final_image = Image.new('RGB', (total_width, total_height))

        # Вставляем по очереди
        x_offset = 0
        for sub_img in final_sub_imgs:
            final_image.paste(sub_img, (x_offset, 0))
            x_offset += sub_img.width

        # Сохраняем итоговое изображение
        out_path = os.path.join(output_folder, f"{base_name}.{out_format}")
        final_image.save(out_path)
        print(f"Изображение '{base_name}' восстановлено и сохранено в '{out_path}'.")


if __name__ == "__main__":
    # Пример использования функции
    parser = ArgumentParser()
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--step", type=int)
    parser.add_argument("--format", type=str, default="tif")
    args = parser.parse_args()
    patch_folder = f"{args.out_dir}/eval_predict/step_{args.step}"
    output_folder = f"{args.out_dir}/eval_predict_rec_{args.step}"

    restore_3_images_from_patches(patch_folder, output_folder, args.format)
