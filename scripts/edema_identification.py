import cv2
import numpy as np
import os

<<<<<<< HEAD
#paths
=======
# ==============================
# 1. SETUP PATHS
# ==============================
>>>>>>> e16fe415e8ef5f1b2d99720338800b9705ef0329
cardiac_roi_folder = "/Users/yashitak/Desktop/stuff/edema/data/edema_identification"
output_folder = "/Users/yashitak/Desktop/stuff/edema/results"

os.makedirs(output_folder, exist_ok=True)
images = [f for f in os.listdir(cardiac_roi_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

for img_name in images:
    img_path = os.path.join(cardiac_roi_folder, img_name)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None: continue
    
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

<<<<<<< HEAD
   #edge intesnity
    edges = cv2.Canny(img_gray, 50, 150)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sticky_zone = cv2.dilate(edges, edge_kernel, iterations=2)

    #thresholding
    mask = cv2.inRange(img_gray, 75, 205)
    sticky_mask = cv2.bitwise_and(mask, sticky_zone)
    
    #close gapes
    m_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    refined_mask = cv2.morphologyEx(sticky_mask, cv2.MORPH_CLOSE, m_kernel)

    #fill holes
=======
    # --- STEP 1: EDGE INTENSITY LOGIC (The "Sticky" Part) ---
    # We find strong gradients in the image (the borders between PE and background)
    edges = cv2.Canny(img_gray, 50, 150)
    
    # We thicken these edges to create a "sticky zone"
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    sticky_zone = cv2.dilate(edges, edge_kernel, iterations=2)

    # --- STEP 2: Thresholding ---
    # Use a range to identify the PE tissue
    mask = cv2.inRange(img_gray, 75, 205)
    
    # --- STEP 3: Combine Intensity Mask with Edge Logic ---
    # This ensures the region must BOTH be the right brightness AND 
    # be near a detected edge to stay included.
    sticky_mask = cv2.bitwise_and(mask, sticky_zone)
    
    # Close small gaps in the sticky mask
    m_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    refined_mask = cv2.morphologyEx(sticky_mask, cv2.MORPH_CLOSE, m_kernel)

    # --- STEP 4: Hole Filling ---
>>>>>>> e16fe415e8ef5f1b2d99720338800b9705ef0329
    cnts, hier = cv2.findContours(refined_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is not None:
        for i in range(len(cnts)):
            if hier[0][i][3] != -1:
                cv2.drawContours(refined_mask, [cnts[i]], -1, 255, -1)

<<<<<<< HEAD
    #contour selection
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # ignore 60+ percent
=======
    # --- STEP 5: Final Contour Selection ---
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Ignore anything taking up more than 60% of image (usually background leaks)
>>>>>>> e16fe415e8ef5f1b2d99720338800b9705ef0329
        valid_cnts = [c for c in contours if cv2.contourArea(c) < (h * w * 0.6)]
        
        if valid_cnts:
            edema_contour = max(valid_cnts, key=cv2.contourArea)
            
<<<<<<< HEAD
            # approximation
            epsilon = 0.002 * cv2.arcLength(edema_contour, True)
            edema_contour = cv2.approxPolyDP(edema_contour, epsilon, True)

            # diameter calc
=======
            # Use high-fidelity approximation to keep the "sticky" edge detail
            epsilon = 0.002 * cv2.arcLength(edema_contour, True)
            edema_contour = cv2.approxPolyDP(edema_contour, epsilon, True)

            # --- STEP 6: Diameter Calculation ---
>>>>>>> e16fe415e8ef5f1b2d99720338800b9705ef0329
            pts = edema_contour.reshape(-1, 2)
            dist_matrix = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
            idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
            p1, p2 = tuple(pts[idx[0]]), tuple(pts[idx[1]])

<<<<<<< HEAD
            # annotations
=======
            # --- STEP 7: Visualize ---
>>>>>>> e16fe415e8ef5f1b2d99720338800b9705ef0329
            result = img_bgr.copy()
            cv2.drawContours(result, [edema_contour], -1, (255, 255, 0), 2)
            cv2.line(result, p1, p2, (0, 255, 255), 2)

            cv2.imshow("Edge-Sticky Identification", result)
            if cv2.waitKey(0) & 0xFF == ord('q'): break

cv2.destroyAllWindows()