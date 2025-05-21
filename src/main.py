import cv2
import argparse
from pathlib import Path
from colorama import init, Fore
from image_processing import process_image

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
    
    # Create resuts/linearized, results/detection and results/wb directories if they don't exist
    for subdir in ["linearized", "detection", "wb"]:
        (folder.parent / "results" / subdir).mkdir(parents=True, exist_ok=True)

    for img_path in files:
        process_image(img_path, chart_enum, args.num)

if __name__ == "__main__":
    main()
