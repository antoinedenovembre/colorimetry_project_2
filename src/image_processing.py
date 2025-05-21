import cv2
import numpy as np
from pathlib import Path
from colorama import Fore
from macbeth import detect_macbeth

REF_PATCH_NUMBER = 21

def linearize(img: np.ndarray) -> np.ndarray:
    """
    Convert an sRGB image (values in [0,1]) to linear RGB using IEC 61966-2-1:
      if C ≤ 0.04045: C_lin = C / 12.92
      else           : C_lin = ((C + 0.055)/1.055) ** 2.4
    """
    # Use vectorized operations for better performance
    mask = img <= 0.04045
    img_lin = np.where(mask, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    return img_lin

def apply_white_balance(img_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    """
    img_bgr : float32 array in [0–1], shape (H,W,3) BGR
    ref_bgr : uint8 or float32 patch color in [0–255] or [0–1] BGR
    """
    mean = np.sum(ref_bgr) / 3

    img_bgr[:, :, 0] /= (ref_bgr[0] + 1)
    img_bgr[:, :, 1] /= (ref_bgr[1] + 1)
    img_bgr[:, :, 2] /= (ref_bgr[2] + 1)
    img_bgr *= mean

    return img_bgr

def average_3x3_patch(img: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """
    Get the average color of a 3x3 patch around (cx, cy) in img.
    Returns a float32 array in [0–1] BGR.
    """
    h, w = img.shape[:2]
    x1 = max(0, cx - 1)
    x2 = min(w - 1, cx + 1)
    y1 = max(0, cy - 1)
    y2 = min(h - 1, cy + 1)

    patch = img[y1:y2+1, x1:x2+1]
    avg_bgr = np.mean(patch, axis=(0, 1))
    return avg_bgr

def process_image(path: Path, chart_enum: int, max_num: int):
    img_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        print(Fore.RED + f"Error: failed to load '{path.name}' – skipping.")
        return

    # linearize
    img_f   = img_bgr.astype(np.float32) / 255.0   # [0-1] BGR
    img_lin = linearize(img_f)                     # [0-1] linearized BGR
    img_lin_8u = np.clip(img_lin * 255.0, 0, 255).astype(np.uint8)

    out_name = "lin_" + path.stem + path.suffix
    out_path = path.parent.parent / "results" / "linearized" / out_name
    cv2.imwrite(str(out_path), img_lin_8u)
    print(Fore.GREEN + f"Saved linearized '{out_name}'.")

    # detect
    annotated, found, centers = detect_macbeth(img_bgr, chart_enum, max_num)
    if not found:
        print(Fore.YELLOW + f"Warning: no chart detected in '{path.name}'.")
        exit(0)

    out_name = "det_" + path.stem + path.suffix
    out_path = path.parent.parent / "results" / "detection" / out_name
    cv2.imwrite(str(out_path), annotated)
    print(Fore.GREEN + f"Saved annotated '{out_name}'.")

    # retrieve the reference patch BGR values
    cx = centers[REF_PATCH_NUMBER][0]
    cy = centers[REF_PATCH_NUMBER][1]
    ref_bgr = average_3x3_patch(img_lin_8u, cx, cy)

    # apply WB on the linear float image
    img_lin_float = img_lin_8u.astype(np.float32)
    wb_lin = apply_white_balance(img_lin_float, ref_bgr)

    # save the white-balanced image
    out_8u = wb_lin.astype(np.uint8)
    out_name = "balanced_" + path.stem + path.suffix
    out_path = path.parent.parent / "results" / "wb" / out_name
    cv2.imwrite(str(out_path), out_8u)
    print(Fore.GREEN + f"Saved adapted '{out_name}'.")