"""Установка наших тренеров внутрь пакета nnunetv2.

Зачем скрипт, а не «положить руками». nnU-Net ищет класс тренера обходом
СВОЕГО пакета (`recursive_find_python_class` по `nnunetv2.training.nnUNetTrainer`),
указать файл снаружи нельзя. Значит копия обязана лежать в `.venv`, а `.venv`
переустанавливается и в гит не входит — без этого скрипта тренер молча
исчезнет, и обучение упадёт с «unable to locate trainer».

Исходник живёт в репозитории рядом с этим файлом, копия ставится по месту.

Запуск:
    .venv/Scripts/python.exe research/scripts/nnunet_trainers/install.py
    .venv/Scripts/python.exe research/scripts/nnunet_trainers/install.py --check
"""

import argparse
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = ["nnUNetTrainer_finetune.py"]


def target_dir() -> Path:
    import nnunetv2.training.nnUNetTrainer.variants as v
    return Path(v.__file__).resolve().parent / "custom"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="только проверить, что тренеры находятся")
    a = ap.parse_args()

    dst = target_dir()
    if not a.check:
        dst.mkdir(parents=True, exist_ok=True)
        # nnU-Net обходит пакеты, поэтому каталогу нужен __init__.py
        (dst / "__init__.py").touch()
        for name in FILES:
            shutil.copy2(HERE / name, dst / name)
            print(f"установлен: {dst / name}")

    # Проверяем тем же механизмом, каким ищет сам nnU-Net, — иначе «скопировал»
    # ещё не значит «найдётся».
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    import nnunetv2
    base = Path(nnunetv2.__file__).resolve().parent / "training" / "nnUNetTrainer"
    ok = True
    for cls in ("nnUNetTrainer_500epochs_lr1en3", "nnUNetTrainer_750epochs_lr1en3",
                "nnUNetTrainer_1000epochs_lr1en3"):
        found = recursive_find_python_class(
            str(base), cls, "nnunetv2.training.nnUNetTrainer")
        print(f"  {cls}: {'НАЙДЕН' if found else 'НЕ НАЙДЕН'}")
        ok = ok and found is not None
    if not ok:
        sys.exit("тренеры не находятся — обучение не запустится")


if __name__ == "__main__":
    main()
