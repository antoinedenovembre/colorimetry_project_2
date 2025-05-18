"""
Requires:
    pip install opencv-contrib-python numpy colorama
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from colorama import init, Fore, Style

def linearize(img: np.ndarray) -> np.ndarray:
    """
    Convert an sRGB image (values in [0,1]) to linear RGB using IEC 61966-2-1:
      if C ≤ 0.04045: C_lin = C / 12.92
      else           : C_lin = ((C + 0.055)/1.055) ** 2.4
    """
    mask = img > 0.04045
    lin = np.empty_like(img, dtype=np.float32)
    lin[mask]  = ((img[mask] + 0.055) / 1.055) ** 2.4
    lin[~mask] = img[~mask] / 12.92
    return lin

def detect_macbeth(img: np.ndarray, chart_type: int, max_num: int) -> tuple[np.ndarray, bool]:
    """
    Run the OpenCV mcc detector on `img`. Returns the annotated image
    and a flag indicating whether any chart was found.
    """
    detector = cv2.mcc.CCheckerDetector.create()
    found = detector.process(img, chart_type, max_num)
    if not found:
        return img, False

    for checker in detector.getListColorChecker():
        drawer = cv2.mcc.CCheckerDraw.create(checker)
        drawer.draw(img)
    return img, True

def process_image(path: Path, chart_enum: int, max_num: int):
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        print(Fore.RED + f"Error: failed to load '{path.name}' – skipping.")
        return

    # linearize
    img_f = img_bgr.astype(np.float32) / 255.0
    img_lin = linearize(img_f)
    img_lin_8u = np.clip(img_lin * 255.0, 0, 255).astype(np.uint8)
    out_name = "lin_" + path.stem + path.suffix
    out_path = path.parent.parent / "results" / out_name
    cv2.imwrite(str(out_path), img_lin_8u)
    print(Fore.CYAN + f"Linearized '{path.name}'.")

    # detect
    annotated, found = detect_macbeth(img_bgr, chart_enum, max_num)
    if not found:
        print(Fore.YELLOW + f"Warning: no chart detected in '{path.name}'.")
    else:
        out_name = "det_" + path.stem + path.suffix
        out_path = path.parent.parent / "results" / out_name
        cv2.imwrite(str(out_path), annotated)
        print(Fore.GREEN + f"Saved annotated '{out_name}'.")

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
        print(Fore.YELLOW + f"No JPG/JPEG files found in '{folder}'.")
        return

    for img_path in files:
        process_image(img_path, chart_enum, args.num)

if __name__ == "__main__":
    main()
