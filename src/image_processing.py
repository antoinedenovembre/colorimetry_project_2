import cv2
import numpy as np
from pathlib import Path
from colorama import Fore
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from macbeth import detect_macbeth

REF_PATCH_NUMBER = 20

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

def apply_gamma_correction(img: np.ndarray) -> np.ndarray:
    """
    Apply gamma correction to an image using the sRGB standard.
    The input image should be in the range [0, 1].
    """
    img = np.clip(img, 0, 1)  # Ensure values are in [0, 1]
    img_corrected = np.where(img <= 0.0031308,
                             img * 12.92,
                             1.055 * (img ** (1 / 2.4)) - 0.055)
    return img_corrected

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

def average_7x7_patch(img: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """
    Get the average color of a 7x7 patch around (cx, cy) in img.
    Returns a float32 array in [0–1] BGR.
    """
    h, w = img.shape[:2]
    x1 = max(0, cx - 3)
    x2 = min(w - 1, cx + 3)
    y1 = max(0, cy - 3)
    y2 = min(h - 1, cy + 3)

    patch = img[y1:y2+1, x1:x2+1]
    avg_bgr = np.mean(patch, axis=(0, 1))
    return avg_bgr

def plot_rg_graph(img_before: np.ndarray, img_after: np.ndarray, centers: list, index: int = 0):
    _, ax = plt.subplots()
    ax.set_title("rg Graph")
    ax.set_xlabel("r")
    ax.set_ylabel("g")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add red dotted lines at coordinates 0.3, 0.3
    ax.axvline(x=0.33, color='red', linestyle='--', label="r=0.33")
    ax.axhline(y=0.33, color='red', linestyle='--', label="g=0.33")

    # Plot the before and after colors
    for i, center in enumerate(centers):
        x = center[0]
        y = center[1]
        bgr_before = average_7x7_patch(img_before, x, y)
        bgr_after = average_7x7_patch(img_after, x, y)

        # Convert BGR to RG
        epsilon = 1e-10  # Small value to prevent division by zero
        r_before = bgr_before[2] / (bgr_before[0] + bgr_before[1] + bgr_before[2] + epsilon)
        g_before = bgr_before[1] / (bgr_before[0] + bgr_before[1] + bgr_before[2] + epsilon)
        r_after = bgr_after[2] / (bgr_after[0] + bgr_after[1] + bgr_after[2] + epsilon)
        g_after = bgr_after[1] / (bgr_after[0] + bgr_after[1] + bgr_after[2] + epsilon)

        # Plot before points in blue
        ax.scatter(r_before, g_before, color='blue', label="Before" if i == 0 else "")

        # Plot after points in green
        ax.scatter(r_after, g_after, color='green', label="After" if i == 0 else "")

    # Avoid duplicate labels in the legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys())

    # Save the plot
    out_name = f"rg_graph_{index}.png"
    out_path = Path("results/graphs") / out_name
    plt.savefig(out_path)
    print(Fore.GREEN + f"Saved rg graph '{out_name}'.")

def process_image(path: Path, chart_enum: int, max_num: int):
    img_bgr = cv2.imread(str(path))    
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

    # apply WB on the linear float image
    img_lin_float = img_lin_8u.astype(np.float32)
    ref_bgr = average_7x7_patch(img_lin_float, cx, cy)
    wb = apply_white_balance(img_lin_float, ref_bgr)
    wb_gamma = apply_gamma_correction(wb / 255.0)  # [0-1] BGR
    wb_8u = np.clip(wb_gamma * 255, 0, 255).astype(np.uint8)

    # rg graph
    index = path.stem.split(".")[0]
    plot_rg_graph(img_lin_8u, wb_8u, centers, index)

    # save the white-balanced image
    out_name = "balanced_" + path.stem + path.suffix
    out_path = path.parent.parent / "results" / "wb" / out_name
    cv2.imwrite(str(out_path), wb_8u)
    print(Fore.GREEN + f"Saved adapted '{out_name}'.")