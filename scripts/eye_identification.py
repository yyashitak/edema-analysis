import cv2
import numpy as np
import os

#paths
image_folder = "/Users/yashitak/Desktop/stuff/edema/data/"
roi_output_folder = "/Users/yashitak/Desktop/stuff/edema/data/edema_identification"

os.makedirs(roi_output_folder, exist_ok=True)

images = [f for f in os.listdir(image_folder)
          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

#processing loop
for img_name in images:
    img_path = os.path.join(image_folder, img_name)
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Could not read {img_name}")
        continue

    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    h_img, w_img = img_gray.shape

    #pre-processing
    blurred = cv2.GaussianBlur(img_gray, (9, 9), 0)

    # eye thresholding
    _, eye_thresh = cv2.threshold(
        blurred, 30, 255, cv2.THRESH_BINARY_INV
    )

    # eye contour
    contours, _ = cv2.findContours(
        eye_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_eye = None
    max_x = 0

    for c in contours:
        area = cv2.contourArea(c)
        if 800 < area < 15000:
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > 0.5:
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue

                cX = int(M["m10"] / M["m00"])

                # Right bias
                if cX > max_x:
                    max_x = cX
                    best_eye = c

    # roi definition
    if best_eye is not None:
        x, y, w, h = cv2.boundingRect(best_eye)

        # Blue box definition (same logic as before)
        roi_x = x - int(w * 0.5)
        roi_y = y + h
        roi_w = int(w * 2)
        roi_h = int(h * 2)

        # clip bounds
        roi_x = max(0, roi_x)
        roi_y = max(0, roi_y)
        roi_w = min(roi_w, w_img - roi_x)
        roi_h = min(roi_h, h_img - roi_y)

        # crop
        roi_crop = img_color[roi_y:roi_y + roi_h,
                              roi_x:roi_x + roi_w]

        if roi_crop.size == 0:
            print(f"Empty ROI for {img_name}, skipping.")
            continue

        # save
        save_path = os.path.join(
            roi_output_folder, f"roi_{img_name}"
        )
        cv2.imwrite(save_path, roi_crop)
        print(f"Saved ROI → {save_path}")

        # -draw
        cv2.rectangle(
            img_color,
            (roi_x, roi_y),
            (roi_x + roi_w, roi_y + roi_h),
            (255, 0, 0),
            2
        )

    else:
        print(f"Eye not found in {img_name}")

    # -see
    cv2.imshow("Eye Mask", eye_thresh)
    cv2.imshow("ROI Location", img_color)
    cv2.waitKey(0)

cv2.destroyAllWindows()
print("\nAll images processed.")
