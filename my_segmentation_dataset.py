import os
import datasets

_CITATION = """
(Текст цитирования, если требуется)
"""

_DESCRIPTION = """
Описание вашего датасета: для чего он нужен и как устроен.
"""

class MySegmentationDataset(datasets.GeneratorBasedBuilder):
    """Датасет для семантической сегментации, читает пары (img, mask)."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),  # Изображение
                    "annotation": datasets.Image(),   # Соответствующая маска
                    "filename": datasets.Value("string"),    # имя файла изображения
                }
            ),
            supervised_keys=("image", "mask"),  # Указываем, что вход — изображение, выход — маска
        )
    
    def _split_generators(self, dl_manager):
        # Ожидаем, что пользователь передаст data_dir, где лежат папки "train" и "val"
        root_dir = self.config.data_dir
    
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"base_dir": os.path.join(root_dir, "train_prep")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"base_dir": os.path.join(root_dir, "val_prep")},
            ),
        ]
    
    def _generate_examples(self, base_dir):
        """
        base_dir -> папка "train" или "val".
        Внутри неё - подпапки с названиями вида Ns, в каждой из которых
        лежат папки img/ и mask/ с нарезанными патчами JPEG 256x256.
        """
        idx = 0
        # Проходим по всем подпапкам (Ns)
        for folder_name in sorted(os.listdir(base_dir)):
            folder_path = os.path.join(base_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
    
            images_dir = os.path.join(folder_path, "img")
            masks_dir = os.path.join(folder_path, "mask")
    
            # Убедимся, что обе директории существуют
            if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
                continue
    
            # Перебираем все файлы в папке с изображениями
            for image_file in sorted(os.listdir(images_dir)):
                # print("keep going")
                if not image_file.lower().endswith(".png"):
                    # Можно расширить под другие форматы, если нужно
                    # print("BUTWTHBRUH")
                    continue
    
                image_path = os.path.join(images_dir, image_file)
                mask_path = os.path.join(masks_dir, image_file)  # Предполагаем, что имена совпадают
    
                if os.path.exists(mask_path):
                    # print("GOGO")
                    yield idx, {
                        "image": image_path,
                        "annotation": mask_path,
                        "filename": image_file,
                    }
                    idx += 1