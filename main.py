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
    Detects concentric circles in a PIL image, highlights them, and returns a new PIL image.
    """
    # Convert PIL image to OpenCV format
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]
    region = (0, 0, w, h)
    # Convert to grayscale
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

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

    found = False
    if inner_circles is not None and outer_circles is not None:
        inner_circles = np.uint16(np.around(inner_circles[0]))
        outer_circles = np.uint16(np.around(outer_circles[0]))
        for icx, icy, ir in inner_circles:
            for ocx, ocy, orad in outer_circles:
                if abs(icx - ocx) < 5 and abs(icy - ocy) < 5:
                    cv2.circle(cv_img, (ocx, ocy), orad, (0, 0, 255), 3)
                    cv2.circle(cv_img, (icx, icy), ir, (0, 0, 255), 3)
                    found = True
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
