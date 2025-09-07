import cv2
import numpy as np
import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

def detect_and_highlight_circle(image_path, region, output_path):
    """
    Detects a black circle (~65px ±4px) with a white inner circle (~33px ±4px)
    in a given region of an image. If found, highlights both in red.

    :param image_path: Path to input image
    :param region: (x, y, w, h) specifying the region of interest
    :param output_path: Path to save output image
    :return: True if circle pair found, False otherwise
    """

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or could not be loaded.")

    x, y, w, h = region
    roi = image[y:y+h, x:x+w]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    found = False

    # Detect small (inner) circles (white)
    inner_circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=100,
        param2=30,
        minRadius=14,
        maxRadius=18
    )

    # Detect large (outer) circles (black)
    outer_circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=100,
        param2=30,
        minRadius=30,
        maxRadius=35
    )

    if inner_circles is not None and outer_circles is not None:
        inner_circles = np.uint16(np.around(inner_circles[0]))
        outer_circles = np.uint16(np.around(outer_circles[0]))
        # Find pairs of circles with close centers
        for icx, icy, ir in inner_circles:
            for ocx, ocy, orad in outer_circles:
                if abs(icx - ocx) < 5 and abs(icy - ocy) < 5:
                    # Draw both circles in red on the original image (adjusted for region offset)
                    cv2.circle(image, (x+ocx, y+ocy), orad, (0, 0, 255), 3)
                    cv2.circle(image, (x+icx, y+icy), ir, (0, 0, 255), 3)
                    found = True
    # Always save the output image, even if not found
    cv2.imwrite(output_path, image)
    return found

class ImageCanvas(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.image = None
        self.photo = None
        self.img_id = None
        self.start_x = self.start_y = None
        self.rect_id = None
        self.region = None
        self.display_scale = 1.0
        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)

    def set_image(self, cv_img):
        self.cv_img = cv_img
        img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        max_w, max_h = 800, 600
        scale = min(max_w / w, max_h / h, 1.0)
        self.display_scale = scale
        disp_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        self.image = Image.fromarray(disp_img)
        self.photo = ImageTk.PhotoImage(self.image)
        self.config(width=self.photo.width(), height=self.photo.height())
        self.delete("all")
        self.img_id = self.create_image(0, 0, anchor="nw", image=self.photo)
        self.region = None
        self.rect_id = None

    def on_press(self, event):
        if self.image is None:
            return
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.delete(self.rect_id)
            self.rect_id = None

    def on_drag(self, event):
        if self.image is None or self.start_x is None or self.start_y is None:
            return
        if self.rect_id:
            self.delete(self.rect_id)
        self.rect_id = self.create_rectangle(
            self.start_x, self.start_y, event.x, event.y,
            outline="red", width=2, dash=(4, 2)
        )

    def on_release(self, event):
        if self.image is None or self.start_x is None or self.start_y is None:
            return
        x1, y1 = self.start_x, self.start_y
        x2, y2 = event.x, event.y
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        self.region = (x1, y1, x2, y2)
        self.start_x = self.start_y = None

        # Log the selected region in pixels
        region_px = self.get_selected_region()
        if region_px:
            print(f"Selected region (pixels): x={region_px[0]}, y={region_px[1]}, w={region_px[2]}, h={region_px[3]}")

    def get_selected_region(self):
        if self.region is None or self.cv_img is None:
            return None
        x1, y1, x2, y2 = self.region
        scale = self.display_scale
        ix1 = int(x1 / scale)
        iy1 = int(y1 / scale)
        ix2 = int(x2 / scale)
        iy2 = int(y2 / scale)
        x = min(ix1, ix2)
        y = min(iy1, iy2)
        w = abs(ix2 - ix1)
        h = abs(iy2 - iy1)
        if w == 0 or h == 0:
            return None
        # Clamp to image size
        img_h, img_w = self.cv_img.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        return (x, y, w, h)

class MainWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Circle Detector")
        self.image_path = None
        self.cv_img = None
        self.result_img = None

        btn_frame = tk.Frame(root)
        btn_frame.pack(fill="x", padx=5, pady=5)

        load_btn = tk.Button(btn_frame, text="Load Image", command=self.load_image)
        load_btn.pack(side="left", padx=2)

        detect_btn = tk.Button(btn_frame, text="Detect Circle", command=self.detect_circle)
        detect_btn.pack(side="left", padx=2)

        save_btn = tk.Button(btn_frame, text="Save Result", command=self.save_result)
        save_btn.pack(side="left", padx=2)

        self.canvas = ImageCanvas(root, bg="gray")
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)

    def load_image(self):
        fname = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
        )
        if fname:
            img = cv2.imread(fname)
            if img is not None:
                self.image_path = fname
                self.cv_img = img
                self.result_img = img.copy()
                self.canvas.set_image(self.result_img)

    def save_result(self):
        if self.result_img is None:
            return
        fname = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".png",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
        )
        if fname:
            cv2.imwrite(fname, self.result_img)

    def detect_circle(self):
        if self.cv_img is None:
            return
        region = self.canvas.get_selected_region()
        if region is None:
            messagebox.showinfo("Info", "Please select a region first.")
            return
        temp_input = "_temp_input.jpg"
        temp_output = "_temp_output.jpg"
        cv2.imwrite(temp_input, self.cv_img)
        found = detect_and_highlight_circle(temp_input, region, temp_output)
        self.result_img = cv2.imread(temp_output)
        self.canvas.set_image(self.result_img)
        if found:
            messagebox.showinfo("Result", "Concentric circles found and highlighted.")
        else:
            messagebox.showinfo("Result", "No matching concentric circles found.")

if __name__ == "__main__":
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()
