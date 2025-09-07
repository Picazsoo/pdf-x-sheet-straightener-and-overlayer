import cv2
import numpy as np
import sys
import os
from pdf2image import convert_from_path
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, portrait
import tkinter as tk
from tkinter import filedialog

def detect_and_highlight_circle_pil(pil_img):
    """
    Detects up to three black dots (blue highlight). If only two are found,
    also tries to find a white dot with a black circle around it (red highlight).
    Returns a new PIL image.
    """
    # Convert PIL image to OpenCV format
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # Threshold to find black dots
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    black_circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 400 or area > 1100:
            continue  # Filter by area (approximate for radius 14-18)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius < 12 or radius > 18:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > 0.7:
            black_circles.append((int(x), int(y), int(radius), circularity, area))

    # Sort by circularity, keep up to 3
    black_circles = sorted(black_circles, key=lambda x: -x[3])[:3]
    # Draw black dots in blue
    for x, y, r, _, _ in black_circles:
        cv2.circle(cv_img, (x, y), r, (255, 0, 0), 3)  # Blue

    # If only two black dots found, try to find a white dot with black circle
    if len(black_circles) == 2:
        # Use HoughCircles to find concentric circles (white inside black)
        gray_blur = cv2.medianBlur(gray, 5)
        # Inner (white) circle
        inner_circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=100,
            param2=30,
            minRadius=14,
            maxRadius=18
        )
        # Outer (black) circle
        outer_circles = cv2.HoughCircles(
            gray_blur,
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
            for icx, icy, ir in inner_circles:
                for ocx, ocy, orad in outer_circles:
                    if abs(icx - ocx) < 5 and abs(icy - ocy) < 5:
                        # Draw both circles in red
                        cv2.circle(cv_img, (ocx, ocy), orad, (0, 0, 255), 3)
                        cv2.circle(cv_img, (icx, icy), ir, (0, 0, 255), 3)
                        break

    # Convert back to PIL
    result_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    return result_pil

def ensure_landscape(pil_img):
    """
    Rotates the image counter-clockwise if it's not in landscape orientation.
    """
    if pil_img.width < pil_img.height:
        return pil_img.rotate(90, expand=True)
    return pil_img

def process_pdf(input_pdf, output_pdf):
    try:
        # Convert PDF to images
        pages = convert_from_path(input_pdf)
    except Exception as e:
        print("Error: Unable to convert PDF to images. Make sure Poppler is installed and in your PATH.")
        print("See: https://github.com/oschwartz10612/poppler-windows or install poppler-utils via your package manager.")
        print(f"Original error: {e}")
        sys.exit(1)
    processed_images = []
    target_size = (3918, 2479)  # width, height
    for idx, page in enumerate(pages):
        img = ensure_landscape(page)
        img = img.resize(target_size, Image.LANCZOS)
        img = detect_and_highlight_circle_pil(img)
        processed_images.append(img)

    # Save images as PDF
    processed_images[0].save(
        output_pdf,
        save_all=True,
        append_images=processed_images[1:],
        resolution=200
    )

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    input_pdf = filedialog.askopenfilename(
        title="Select input PDF",
        filetypes=[("PDF files", "*.pdf")]
    )
    if not input_pdf:
        print("No input PDF selected.")
        sys.exit(1)

    output_pdf = filedialog.asksaveasfilename(
        title="Save output PDF as",
        defaultextension=".pdf",
        filetypes=[("PDF files", "*.pdf")]
    )
    if not output_pdf:
        print("No output PDF selected.")
        sys.exit(1)

    process_pdf(input_pdf, output_pdf)
    print(f"Processed PDF saved to {output_pdf}")
