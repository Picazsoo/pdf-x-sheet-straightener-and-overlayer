import sys
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path

PAGE_TYPE_50_COORDS = [
    [384, 132, 9],
    [1364, 132, 10],
    [2453, 132, 6],
    [384, 2200, 9],
    [1364, 2200, 10],
    [2453, 2200, 6],
]

PAGE_TYPE_100_COORDS = [
    [384, 155, 9],
    [1364, 155, 10],
    [2453, 155, 6],
    [384, 645, 9],
    [1364, 645, 10],
    [2453, 645, 6],
    [384, 1163, 9],
    [1364, 1163, 10],
    [2453, 1163, 6],
    [384, 1680, 9],
    [1364, 1680, 10],
    [2453, 1680, 6],
]

def find_reference_points(pil_img):
    """
    Finds up to three black dots. If only two, tries to find a white dot with black circle.
    Returns a list of (x, y, type) tuples for the detected points, where type is "BLACK" or "WHITE".
    """
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 400 or area > 1100:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius < 12 or radius > 18:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > 0.7:
            points.append((int(x), int(y), "BLACK"))

    # If only two black dots, try to find a white dot with black circle
    if len(points) == 2:
        gray_blur = cv2.medianBlur(gray, 5)
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
                    if abs(int(icx) - int(ocx)) < 5 and abs(int(icy) - int(ocy)) < 5:
                        points.append((int(icx), int(icy), "WHITE"))
                        break
    return points

def transform_image_by_points(pil_img, points):
    """
    Transforms the image so that the leftmost point is at (284, 2194) and the rightmost at (3284, 2194).
    Only rotates and scales horizontally (no vertical scaling).
    """
    # Use only coordinates for transformation
    if len(points) < 2:
        return pil_img  # Not enough points to transform

    pts = sorted(points, key=lambda p: p[0])
    left = np.array((pts[0][0], pts[0][1]), dtype=np.float32)
    right = np.array((pts[-1][0], pts[-1][1]), dtype=np.float32)

    # Target positions
    target_left = np.array([284, 2194], dtype=np.float32)
    target_right = np.array([3284, 2194], dtype=np.float32)

    # Compute angle and scale
    vec_src = right - left
    vec_dst = target_right - target_left
    angle_src = np.arctan2(vec_src[1], vec_src[0])
    angle_dst = np.arctan2(vec_dst[1], vec_dst[0])
    angle = np.degrees(angle_dst - angle_src)
    scale = np.linalg.norm(vec_dst) / np.linalg.norm(vec_src)

    # Build transformation matrix
    # 1. Translate left to origin
    # 2. Rotate by angle
    # 3. Scale horizontally
    # 4. Translate to target_left
    M_translate1 = np.array([[1, 0, -left[0]], [0, 1, -left[1]], [0, 0, 1]], dtype=np.float32)
    theta = angle_dst - angle_src
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    M_rotate = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32)
    M_scale = np.array([[scale, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    M_translate2 = np.array([[1, 0, target_left[0]], [0, 1, target_left[1]], [0, 0, 1]], dtype=np.float32)

    # Combine transformations: T2 * S * R * T1
    M = M_translate2 @ M_scale @ M_rotate @ M_translate1
    M_affine = M[:2, :]

    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]
    transformed = cv2.warpAffine(cv_img, M_affine, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255,255,255))
    return Image.fromarray(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))

def detect_and_highlight_circle_pil(pil_img):
    """
    Detects up to three black dots (blue highlight). If only two are found,
    also tries to find a white dot with black circle around it (red highlight).
    Returns a new PIL image.
    """
    # ...existing code for detection, but highlight after transformation...
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    black_circles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 400 or area > 1100:
            continue
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius < 12 or radius > 18:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity > 0.7:
            black_circles.append((int(x), int(y), int(radius), circularity, area))

    black_circles = sorted(black_circles, key=lambda x: -x[3])[:3]
    for x, y, r, _, _ in black_circles:
        cv2.circle(cv_img, (x, y), r, (255, 0, 0), 3)  # Blue

    if len(black_circles) == 2:
        gray_blur = cv2.medianBlur(gray, 5)
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
                    # Cast to int to avoid overflow in subtraction
                    if abs(int(icx) - int(ocx)) < 5 and abs(int(icy) - int(ocy)) < 5:
                        cv2.circle(cv_img, (int(ocx), int(ocy)), int(orad), (0, 0, 255), 3)
                        cv2.circle(cv_img, (int(icx), int(icy)), int(ir), (0, 0, 255), 3)
                        break

    result_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    return result_pil

def ensure_landscape(pil_img):
    """
    Rotates the image counter-clockwise if it's not in landscape orientation.
    """
    if pil_img.width < pil_img.height:
        return pil_img.rotate(90, expand=True)
    return pil_img

def get_page_type(points):
    """
    Returns 'PAGE_TYPE_50' or 'PAGE_TYPE_100' based on detected points.
    3 black dots = PAGE_TYPE_50
    2 black + 1 white = PAGE_TYPE_100
    """
    black_count = sum(1 for p in points if p[2] == "BLACK")
    white_count = sum(1 for p in points if p[2] == "WHITE")
    if black_count == 3:
        return 'PAGE_TYPE_50'
    if black_count == 2 and white_count == 1:
        return 'PAGE_TYPE_100'
    # Fallback: treat as PAGE_TYPE_50
    return 'PAGE_TYPE_50'

def overlay_text_fields(pil_img, coords, counter_start):
    """
    Draws text at given coords on pil_img, incrementing counter by step for each.
    Returns the new image and the next counter value.
    """
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    # Try to use a monospaced font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()
    counter = counter_start
    for x, y, step in coords:
        draw.text((x, y), str(counter), fill=(0, 0, 0), font=font)
        counter += step
    return img, counter

def process_pdf(input_pdf, output_pdf):
    try:
        # Convert PDF to images
        pages = convert_from_path(input_pdf)
    except Exception as e:
        print("Error: Unable to convert PDF to images. Make sure Poppler is installed and in your PATH.")
        print("See: https://github.com/oschwartz10612/poppler-windows or install poppler-utils via your package manager. E.g. winget install --id=oschwartz10612.Poppler  -e. You might need to create system variable POPPLER_PATH pointing to the bin folder inside the poppler folder.")
        print(f"Original error: {e}")
        sys.exit(1)
    processed_images = []
    target_size = (3918, 2479)  # width, height
    counter = 1  # Start counter for the document
    for idx, page in enumerate(pages):
        img = ensure_landscape(page)
        img = img.resize(target_size, Image.LANCZOS)
        # --- Find and transform based on reference points ---
        points = find_reference_points(img)
        img = transform_image_by_points(img, points)
        # --- Highlight after transformation ---
        img = detect_and_highlight_circle_pil(img)
        # --- Determine page type and overlay text ---
        page_type = get_page_type(points)
        if page_type == 'PAGE_TYPE_50':
            coords = PAGE_TYPE_50_COORDS
        else:
            coords = PAGE_TYPE_100_COORDS
        img, counter = overlay_text_fields(img, coords, counter)
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
