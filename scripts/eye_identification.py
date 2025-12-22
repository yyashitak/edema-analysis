import cv2
import numpy as np
import os

# Paths
image_folder = "/Users/yashitak/Desktop/stuff/edema/data/"
output_folder = "/Users/yashitak/Desktop/stuff/edema/results/"
os.makedirs(output_folder, exist_ok=True)

images = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.tif', '.jpeg'))]

for img_name in images:
    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 1. PRE-PROCESSING
    # Stronger blur helps merge the eye into one solid shape
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    
    # 2. STRICT THRESHOLD (Lowered to 30)
    # If the mask window is still too 'white', lower this to 20.
    # If the mask is all black, raise this to 45.
    _, eye_thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY_INV)

    # 3. FIND CONTOURS
    contours, _ = cv2.findContours(eye_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_eye = None
    max_x = 0 

    for c in contours:
        area = cv2.contourArea(c)
        if 800 < area < 15000: 
            # CIRCULARITY CHECK: 4*pi*Area / Perimeter^2
            # A perfect circle is 1.0. We want > 0.5 to avoid body streaks.
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            if circularity > 0.5:
                M = cv2.moments(c)
                if M["m00"] == 0: continue
                cX = int(M["m10"] / M["m00"])
                
                # RIGHT-SIDE BIAS: The head is on the right
                if cX > max_x:
                    max_x = cX
                    best_eye = c

    # 4. DRAW RESULTS
    if best_eye is not None:
        x, y, w, h = cv2.boundingRect(best_eye)
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Define PE Search Zone (Blue Box)
        # Positioned slightly below and to the left of the eye
        roi_x = x - int(w * 0.5)
        roi_y = y + h
        roi_w = w * 2
        roi_h = h * 2
        cv2.rectangle(img_color, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
        print(f"Success: Eye detected in {img_name}")
    else:
        print(f"Failed: Could not find eye in {img_name}. Try adjusting threshold.")

    # 5. SHOW DEBUG WINDOWS
   
    cv2.imshow("Final Detection", img_color)
    
    # Pauses the script - Press any key to see the next image
    cv2.waitKey(0) 

cv2.destroyAllWindows()