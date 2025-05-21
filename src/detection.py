import cv2
import numpy as np
import argparse
from pathlib import Path
from colorama import init, Fore

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

def extract_patch_center_coordinates(checker) -> np.ndarray:
    centers_4 = checker.getColorCharts()  # list of (x,y) floats for each patch
    centers_1 = [[0, 0] for _ in range(24)]

    cx, cy = 0, 0
    for i in range(24):
        for j in range(4):
            cx += centers_4[i*4 + j][0]
            cy += centers_4[i*4 + j][1]
        cx /= 4
        cy /= 4
        cx = int(cx)
        cy = int(cy)
        centers_1[i] = [cx, cy]
        cx, cy = 0, 0

    return centers_1

def detect_macbeth(img: np.ndarray, chart_type: int, max_num: int) -> tuple[np.ndarray, bool, np.ndarray]:
    """
    Run the OpenCV mcc detector on `img`. Returns the annotated image,
    a flag indicating whether any chart was found, and the RGB values of all patches.
    """
    detector = cv2.mcc.CCheckerDetector.create()
    found = detector.process(img, chart_type, max_num, useNet=True)
    patch_colors = []
    
    if not found:
        return img, False, patch_colors

    # get the first detected checker
    checker = detector.getListColorChecker()[0]

    # draw the usual chart outlines
    drawer = cv2.mcc.CCheckerDraw.create(checker)
    drawer.draw(img)

    centers = extract_patch_center_coordinates(checker)

    return img, True, centers

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

def process_image(path: Path, chart_enum: int, max_num: int):
    img_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img_bgr is None:
        print(Fore.RED + f"Error: failed to load '{path.name}' – skipping.")
        return

    # linearize
    img_f   = img_bgr.astype(np.float32) / 255.0   # now in [0-1] sRGB
    img_lin = linearize(img_f)                     # still [0-1] linearized  
    img_lin_8u = np.clip(img_lin * 255.0, 0, 255).astype(np.uint8)

    out_name = "lin_" + path.stem + path.suffix
    out_path = path.parent.parent / "results" / out_name
    cv2.imwrite(str(out_path), img_lin_8u)
    print(Fore.CYAN + f"Linearized '{path.name}'.")

    # detect
    annotated, found, centers = detect_macbeth(img_lin_8u, chart_enum, max_num)
    if not found:
        print(Fore.YELLOW + f"Warning: no chart detected in '{path.name}'.")
        exit(0)
    else:
        out_name = "det_" + path.stem + path.suffix
        out_path = path.parent.parent / "results" / out_name
        cv2.imwrite(str(out_path), annotated)
        print(Fore.GREEN + f"Saved annotated '{out_name}'.")

    # retrieve the reference patch RGB values
    num_ref_patch = 21
    cx = centers[num_ref_patch][0]
    cy = centers[num_ref_patch][1]

    # apply WB on the *linear* float image:
    img_lin_new = np.clip(img_lin * 255.0, 0, 255).astype(np.uint8)
    ref_bgr_values = img_lin_new[cy, cx]
    img_lin_float = np.clip(img_lin * 255.0, 0, 255).astype(np.float32)
    wb_lin = apply_white_balance(img_lin_float, ref_bgr_values)

    # apply white balance
    out_8u = np.clip(wb_lin, 0, 255).astype(np.uint8)
    out_name = "adapted_" + path.stem + path.suffix
    out_path = path.parent.parent / "results" / out_name
    cv2.imwrite(str(out_path), out_8u)
    print(Fore.GREEN + f"Saved adapted '{out_name}'.")

def main():
    init(autoreset=True)
    p = argparse.ArgumentParser(
        description="Batch linearize + Macbeth ColorChecker detection"
    )
    p.add_argument("-d", "--dir", required=True,
                   help="Path to folder with JPG/JPEG images")
    p.add_argument("-t", "--type", type=int, choices=[0,1,2], default=0,
                   help="Chart type: 0=MCC24, 1=SG140, 2=VINYL18")
    p.add_argument("-n", "--num",  type=int, default=1,
                   help="Max # of charts to detect per image")
    args = p.parse_args()

    folder = Path(args.dir)
    if not folder.is_dir():
        print(Fore.RED + f"Error: '{folder}' is not a valid directory.")
        return

    if not hasattr(cv2, "mcc"):
        print(Fore.RED + "Error: cv2.mcc module not found. Install opencv-contrib-python.")
        return

    chart_types = {
        0: cv2.mcc.MCC24,
        1: cv2.mcc.SG140,
        2: cv2.mcc.VINYL18
    }
    chart_enum = chart_types[args.type]

    patterns = ["*.jpg", "*.JPG"]
    files = []
    for pat in patterns:
        files.extend(folder.glob(pat))
    files = sorted(files)
    if not files:
        print(Fore.YELLOW + f"No JPG files found in '{folder}'.")
        return

    for img_path in files:
        process_image(img_path, chart_enum, args.num)

if __name__ == "__main__":
    main()
