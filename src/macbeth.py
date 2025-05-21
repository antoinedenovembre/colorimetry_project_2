import cv2
import numpy as np

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

def extract_patch_center_coordinates(checker) -> np.ndarray:
    """
    From a ColorChecker object, extract the center coordinates of the patches,
    Returns a list of 24 (x,y) coordinates for the patches.
    The coordinates are averaged over the 4 corners of the detected patch.
    """
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