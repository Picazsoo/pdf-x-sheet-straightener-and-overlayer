import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# --- File picker for images ---
root = tk.Tk()
root.withdraw()  # Hide main window

clean_path = filedialog.askopenfilename(
    title="Select CLEAN template image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
)
scan_path = filedialog.askopenfilename(
    title="Select SCAN image",
    filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
)

if not clean_path or not scan_path:
    print("❌ No file selected. Exiting.")
    exit(1)

clean = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
scan = cv2.imread(scan_path, cv2.IMREAD_GRAYSCALE)

# --- GUI for parameter tuning ---
def run_matching(num_features, ransac_thresh, num_matches):
    orb = cv2.ORB_create(num_features)
    kp1, des1 = orb.detectAndCompute(clean, None)
    kp2, des2 = orb.detectAndCompute(scan, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    if des1 is None or des2 is None:
        return None, None, None, None
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    match_vis = cv2.drawMatches(clean, kp1, scan, kp2, matches[:num_matches], None, flags=2)
    if len(matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_thresh)
        if H is not None:
            h, w = clean.shape
            aligned = cv2.warpPerspective(scan, H, (w, h))
            overlay = cv2.addWeighted(clean, 0.5, aligned, 0.5, 0)
            return match_vis, aligned, overlay, H
    return match_vis, None, None, None

def update(*args):
    num_features = feature_slider.get()
    ransac_thresh = ransac_slider.get()
    num_matches = matches_slider.get()
    match_vis, aligned, overlay, H = run_matching(num_features, ransac_thresh, num_matches)
    if match_vis is not None:
        cv2.imshow("Matches", match_vis)
    if overlay is not None:
        cv2.imshow("Overlay (Clean + Aligned Scan)", overlay)
    else:
        cv2.destroyWindow("Overlay (Clean + Aligned Scan)")
    if aligned is not None:
        cv2.imshow("Aligned Scan", aligned)
    else:
        cv2.destroyWindow("Aligned Scan")

# --- Tkinter window for sliders ---
gui = tk.Tk()
gui.title("Fahrplan Matcher Parameters")

tk.Label(gui, text="ORB Features").pack()
feature_slider = tk.Scale(gui, from_=500, to=10000, orient=tk.HORIZONTAL, resolution=100, command=update)
feature_slider.set(5000)
feature_slider.pack(fill=tk.X)

tk.Label(gui, text="RANSAC Threshold").pack()
ransac_slider = tk.Scale(gui, from_=1, to=20, orient=tk.HORIZONTAL, resolution=1, command=update)
ransac_slider.set(5)
ransac_slider.pack(fill=tk.X)

tk.Label(gui, text="Number of Matches to Draw").pack()
matches_slider = tk.Scale(gui, from_=10, to=200, orient=tk.HORIZONTAL, resolution=1, command=update)
matches_slider.set(50)
matches_slider.pack(fill=tk.X)

def save_results():
    num_features = feature_slider.get()
    ransac_thresh = ransac_slider.get()
    num_matches = matches_slider.get()
    match_vis, aligned, overlay, H = run_matching(num_features, ransac_thresh, num_matches)
    if aligned is not None:
        aligned_save_path = filedialog.asksaveasfilename(
            title="Save aligned scan as...",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")]
        )
        if aligned_save_path:
            cv2.imwrite(aligned_save_path, aligned)
            print(f"✅ Alignment done. Saved as '{aligned_save_path}'.")
    if match_vis is not None:
        matches_save_path = filedialog.asksaveasfilename(
            title="Save debug matches image as...",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")]
        )
        if matches_save_path:
            cv2.imwrite(matches_save_path, match_vis)
            print(f"Debug matches saved as '{matches_save_path}'.")

save_btn = tk.Button(gui, text="Save Results", command=save_results)
save_btn.pack(pady=10)

# Initial update and mainloop
update()
gui.mainloop()
cv2.destroyAllWindows()
