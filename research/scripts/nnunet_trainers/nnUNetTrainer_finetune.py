"""Тренеры для ДООБУЧЕНИЯ nnU-Net (не для обучения с нуля).

Зачем свой тренер. У nnU-Net скорость обучения по умолчанию `1e-2` — это
значение для обучения С НУЛЯ, и для дообучения оно на порядок велико: готовые
веса разрушаются в первых же эпохах, и «дообучение» превращается в обучение
заново, только с худшим стартом.

Замер, ради которого это и написано (28.08, Dataset113 + чекпойнт ResEncL
ToothFairy2): при `lr = 1e-2` на нулевой эпохе живыми остались 2 класса из 76,
хотя анатомия (челюсти, нерв, пазухи) у обоих датасетов ОДИНАКОВАЯ и должна
была стартовать высокой. За 250 эпох модель дала 69 полноценных зубов на трёх
кейсах против 74 у исходного ResEncL — то есть не догнала собственный старт.

Классика дообучения — скорость на порядок ниже. Здесь `1e-3`.

Установка (nnU-Net ищет тренеры внутри своего пакета):
    .venv/Scripts/python.exe research/scripts/nnunet_trainers/install.py

Запуск:
    nnUNetv2_train 113 3d_fullres 0 -p nnUNetResEncUNetLPlans \\
        -tr nnUNetTrainer_750epochs_lr1en3 --c
"""

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_lr1en3(nnUNetTrainer):
    """Базовый: скорость обучения 1e-3 вместо 1e-2."""

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-3


class nnUNetTrainer_500epochs_lr1en3(nnUNetTrainer_lr1en3):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500


class nnUNetTrainer_750epochs_lr1en3(nnUNetTrainer_lr1en3):
    """750 эпох при продолжении с 250-й = 500 новых эпох, около 16.5 часов.

    Расписание у nnU-Net полиномиальное от `initial_lr` к нулю за `num_epochs`,
    поэтому при продолжении с 250-й эпохи из 750 скорость стартует не с 1e-3, а
    с ~7e-4 — то есть мягче, чем при обучении с нуля, как дообучению и положено.
    """

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 750


class nnUNetTrainer_1000epochs_lr1en3(nnUNetTrainer_lr1en3):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
