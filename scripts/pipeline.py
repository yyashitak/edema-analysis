import subprocess
import os
import shutil

# ==============================
# 1. RUN EYE DETECTION
# ==============================
print("Running Eye Detection...")
subprocess.run(["python3", "scripts/eye_identification.py"])
print("Eye detection complete.")
print("Running Edema Detection...")
subprocess.run(["python3", "scripts/edema_identification.py"])
print("Edema detection complete.")

# ==============================
# 2. (Other image processing steps)
# Place your edema segmentation, diameter analysis, etc. here
# ==============================

# Example placeholder
# run_segmentation_pipeline()

# ==============================
# 3. CLEAR edema_identification FOLDER
# ==============================
roi_folder = "/Users/yashitak/Desktop/stuff/edema/data/edema_identification"

if os.path.exists(roi_folder):
    for filename in os.listdir(roi_folder):
        file_path = os.path.join(roi_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    print(f"All files in '{roi_folder}' have been cleared.")
else:
    print(f"Folder '{roi_folder}' does not exist.")

print("Pipeline completed.")
