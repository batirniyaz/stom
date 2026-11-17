"""Однопроцессный инференс nnU-Net (обход multiprocessing-воркеров Windows).

Запуск:
    py research_3d/inference/predict_single.py INPUT.mha OUTPUT.mha [MODEL_DIR]

MODEL_DIR по умолчанию — pretrained ToothFairy2 (ResEncUNetL, fold_0).
Наша модель: research_3d/nnUNet_results/Dataset112_ToothFairy2/nnUNetTrainer__nnUNetPlans__3d_fullres
"""

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "nnUNet_results/Dataset112_ToothFairy2/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres"


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    model_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else MODEL_DIR

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=(0,),
                                                   checkpoint_name="checkpoint_best.pth")

    sitk_img = sitk.ReadImage(str(src))
    arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)[None]  # (1, D, H, W)
    props = {
        "spacing": sitk_img.GetSpacing()[::-1],  # nnUNet ждёт (z, y, x)
        "sitk_stuff": {
            "spacing": sitk_img.GetSpacing(),
            "origin": sitk_img.GetOrigin(),
            "direction": sitk_img.GetDirection(),
        },
    }

    seg = predictor.predict_single_npy_array(
        arr, props, None, None, False)  # (D, H, W) в геометрии входа

    out = sitk.GetImageFromArray(seg.astype(np.uint8))
    out.SetSpacing(sitk_img.GetSpacing())
    out.SetOrigin(sitk_img.GetOrigin())
    out.SetDirection(sitk_img.GetDirection())
    dst.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(dst))

    ids, counts = np.unique(seg, return_counts=True)
    print("классы в маске:", dict(zip(ids.tolist(), counts.tolist())))
    print("saved:", dst)


if __name__ == "__main__":
    main()
