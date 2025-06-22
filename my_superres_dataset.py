import os
import glob
import datasets

_CITATION = """
(При необходимости добавьте библиографию или описание вашего датасета)
"""

_DESCRIPTION = """
Пример датасета для задачи супер-разрешения, где:

hr_image: путь к изображению высокого разрешения (HR).
lr_image: путь к соответствующему изображению низкого разрешения (LR). """

class SuperResolutionConfig(datasets.BuilderConfig):
    """Дополнительные настройки датасета при желании."""
    def init(self, **kwargs):
        super(SuperResolutionConfig, self).init(**kwargs)

class SuperResolutionDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        SuperResolutionConfig(
        name="default",
        version=datasets.Version("1.0.0"),
        description=_DESCRIPTION,
        ),
    ]
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "hr_image": datasets.Image(),  # Будем грузить как картинку
                    "lr_image": datasets.Image(),  # Тоже картинка
                    "filename": datasets.Value("string"), # имя файла изображения
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )
    
    def _split_generators(self, dl_manager):
        """
        Здесь указываем пути к train и val, исходя из data_dir,
        который будет передан при вызове load_dataset(..., data_dir=...).
        """
        data_dir = os.path.abspath(self.config.data_dir)
    
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "hr_dir": os.path.join(data_dir, "train_prep"),
                    "lr_dir": os.path.join(data_dir + "_x4", "train_prep"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "hr_dir": os.path.join(data_dir, "val_prep"),
                    "lr_dir": os.path.join(data_dir + "_x4", "val_prep"),
                },
            ),
        ]
    
    def _generate_examples(self, hr_dir, lr_dir):
        """
        Проходимся по всем HR изображениям, ищем для каждого соответствующий LR,
        сопоставляя одинаковый относительный путь.
        """
        # Собираем все пути к изображениям высокого разрешения рекурсивно
        hr_paths = glob.glob(os.path.join(hr_dir, "**", "img", "*.png"), recursive=True)
        
        idx = 0
        for hr_path in hr_paths:
            # Относительный путь относительно hr_dir
            hr_dir1, hr_file = os.path.dirname(hr_path), os.path.basename(hr_path)
            hr_filename, resol = hr_file.split(".")
            hr_filename, x, y = hr_filename.rsplit("_", 2)
            lr_file = "_".join([hr_filename, str(int(x) // 4), str(int(y) // 4)]) + "." + resol
            lr_path = os.path.join(hr_dir1, lr_file)
            rel_path = os.path.relpath(lr_path, start=hr_dir)
            lr_path = os.path.join(lr_dir, rel_path)
            
            # Проверяем, что изображение LR существует
            if not os.path.exists(lr_path):
                print("BRUH")
                continue

            yield idx, {
                "hr_image": hr_path,
                "lr_image": lr_path,
                "filename": hr_file,
            }
            idx += 1
