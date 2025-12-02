import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from skimage.filters import threshold_sauvola

def create_lung_template(reference_image_path, save_path="template_mask.png"):

    file_name = os.path.basename(image_path)
    os.makedirs(outdir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Gagal membaca gambar, periksa path input.")

    canvas_height = 512
    canvas_width = 512
    img = cv2.resize(img, (canvas_width, canvas_height))

    grayxx = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu_thresh_val, _ = cv2.threshold(grayxx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    lower = int(otsu_thresh_val)

    upper = 200

    edges = cv2.Canny(img, lower, upper)

    edge_color = np.zeros_like(img)
    edge_color[edges != 0] = (0, 255, 255)  

    overlay_with_edges = cv2.addWeighted(img, 0.8, edge_color, 0.7, 0)

    gray0 = cv2.cvtColor(overlay_with_edges, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    gray_clahe = clahe.apply(gray0)

    clahe_bgr = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)

    mean_shift = cv2.pyrMeanShiftFiltering(clahe_bgr, sp=15, sr=30)

    gray_blur = cv2.cvtColor(mean_shift, cv2.COLOR_BGR2GRAY)

    otsu_thresh_val, _ = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    adjusted_thresh_val = otsu_thresh_val-5   

    _, thresh = cv2.threshold(gray_blur, adjusted_thresh_val, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    eroded = cv2.erode(thresh,kernel , iterations=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    morphed = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morphed, connectivity=8)
    mask_top2 = np.zeros_like(morphed)

    h, w = morphed.shape
    image_center_x = w // 2

    lung_candidates = []

    for i in range(1, num_labels):  
        x, y, w_i, h_i, area = stats[i]
        cx, cy = centroids[i]

        h_img, w_img = morphed.shape

        touches_top_left = (x <= 5 and y <= 5)
        touches_top_right = (x + w_i >= w_img - 5 and y <= 5)
        touches_bottom_left = (x <= 5 and y + h_i >= h_img - 5)
        touches_bottom_right = (x + w_i >= w_img - 5 and y + h_i >= h_img - 5)

        if any([touches_top_left, touches_top_right, touches_bottom_left, touches_bottom_right]):
            continue  

        touches_top = y <= 5
        touches_bottom = y + h_i >= h_img - 5
        touches_left = x <= 5
        touches_right = x + w_i >= w_img - 5

        if any([touches_top, touches_bottom, touches_left, touches_right]):
            continue  

        aspect_ratio = h_i / (w_i + 1e-5)
        if (area > 3000 and             
            0.5 < aspect_ratio < 6.0 and  
            0.05 * w_img < cx < 0.95 * w_img):  
            lung_candidates.append((i, area))

    top2 = sorted(lung_candidates, key=lambda x: x[1], reverse=True)[:2]
    for i, _ in top2:
        mask_top2[labels == i] = 255

    contours, _ = cv2.findContours(mask_top2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hull_contours = []
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        hull_contours.append(hull)
    img_contour = img.copy()

    cv2.drawContours(img_contour, hull_contours, -1, (0,0,255), 2)

    hull_mask = np.zeros_like(mask_top2)
    cv2.drawContours(hull_mask, hull_contours, -1, 255, -1)

    print("Template mask saved:", save_path)

    return hull_mask

def detect_lungs_grid2(image_path, outdir=r"D:\project\python\2025\tbcdetect\_streamlit\data\TB_Train\Normal"):
    file_name = os.path.basename(image_path)
    os.makedirs(outdir, exist_ok=True)

    # -----------------------------
    # 1. Load image + Resize
    # -----------------------------
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Gagal membaca gambar, periksa path input.")

    canvas_height = 512
    canvas_width = 512
    img = cv2.resize(img, (canvas_width, canvas_height))

    # -----------------------------
    # 2. Tambah border 5px
    # -----------------------------
    pad = 5
    img = cv2.copyMakeBorder(
        img, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=(0,0,0)
    )

    steps = []
    steps.append(("01. Original + Border", img, None))
    

    # -----------------------------
    # 3. Canny di gambar awal (sebelum CLAHE)
    # -----------------------------
    grayxx = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu_thresh_val, _ = cv2.threshold(grayxx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lower = int(otsu_thresh_val)
    upper = 200
    
    edges_full = cv2.Canny(img, lower, upper)
    
    # --- Ambil hanya 50% bagian bawah ---
    h = edges_full.shape[0]
    cut_start = int(h * 0.50)

    edges_bottom = np.zeros_like(edges_full)
    edges_bottom[cut_start:, :] = edges_full[cut_start:, :]

    # Warnai edge bawah
    edge_color_bottom = np.zeros_like(img)
    edge_color_bottom[edges_bottom != 0] = (0, 255, 255)   # soft color

    # Overlay ringan
    overlay_with_edges = cv2.addWeighted(img, 0.97, edge_color_bottom, 0.15, 0)
    
    gray0 = cv2.cvtColor(overlay_with_edges, cv2.COLOR_BGR2GRAY)

    steps.append(("02. Canny bottom", edges_bottom, "gray"))
    steps.append(("03. Overlay Edges", overlay_with_edges, "gray"))
    steps.append(("04. Grayscale", gray0, "gray"))

    # -----------------------------
    # 4. CLAHE
    # -----------------------------
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    gray_clahe = clahe.apply(gray0)
    steps.append(("05. CLAHE", gray_clahe, "gray"))

    # -----------------------------
    # 5. Canny SETELAH CLAHE
    # -----------------------------
    otsu2, _ = cv2.threshold(gray_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lower2 = int(otsu2 * 0.5)
    upper2 = int(otsu2 * 1.5)

    edges_after_clahe = cv2.Canny(gray_clahe, lower2, upper2)
    steps.append(("05b. Canny After CLAHE (Full)", edges_after_clahe, "gray"))

    # -----------------------------
    # 5c. Ambil Top 20% dan Bottom 25%
    # -----------------------------
    h = edges_after_clahe.shape[0]
    top_cut = int(0.20 * h)
    bottom_cut = int(0.75 * h)  # 75% ke bawah = 25% bawah
    
    edges_tb = np.zeros_like(edges_after_clahe)
    
    # Ambil 20% atas
    edges_tb[:top_cut, :] = edges_after_clahe[:top_cut, :]
    
    # Ambil 25% bawah
    edges_tb[bottom_cut:, :] = edges_after_clahe[bottom_cut:, :]
    

    steps.append(("05c. Canny CLAHE Top 20% + Bottom 25%", edges_tb, "gray"))

    # Warnai edge soft
    edge_color2 = np.zeros_like(img)
    edge_color2[edges_tb != 0] = (0, 255, 255)

    # Overlay ringan
    img_next = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)
    
    overlay_edges_tb = cv2.addWeighted(img_next, 0.97, edge_color2, 0.30, 0)
    steps.append(("05E. Overlay Top+Bottom Edges", overlay_edges_tb, None))

    # Grayscale untuk tahapan berikut
    gray_for_next = cv2.cvtColor(overlay_edges_tb, cv2.COLOR_BGR2GRAY)

    # Stabilize histogram setelah overlay
    clahe2 = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(3,3))
    gray_for_next = clahe2.apply(gray_for_next)

    # -----------------------------
    # 6. Mean Shift
    # -----------------------------
    clahe_bgr = cv2.cvtColor(gray_for_next, cv2.COLOR_GRAY2BGR)
    mean_shift = cv2.pyrMeanShiftFiltering(clahe_bgr, sp=15, sr=30)
    gray_blur = cv2.cvtColor(mean_shift, cv2.COLOR_BGR2GRAY)
    steps.append(("06. Mean Shift Filter", gray_blur, "gray"))

    # -----------------------------
    # 7. Threshold Otsu (stable)
    # -----------------------------
    otsu_thresh_val, _ = cv2.threshold(gray_blur, 0, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    _, thresh = cv2.threshold(gray_blur, otsu_thresh_val, 255, cv2.THRESH_BINARY_INV)
    steps.append(("07. Threshold", thresh, "gray"))

    # -----------------------------
    # 8. Erode + Morphology
    # -----------------------------
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    eroded = cv2.erode(thresh, kernel3, iterations=2)
    steps.append(("08. Eroded", eroded, "gray"))

    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    morphed = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel7, iterations=2)
    steps.append(("09. Morphology", morphed, "gray"))

    # -----------------------------
    # 9. Komponen Terbesar (Top-2)
    # -----------------------------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morphed, connectivity=8)
    mask_top2 = np.zeros_like(morphed)

    h_img, w_img = morphed.shape
    lung_candidates = []

    for i in range(1, num_labels):
        x, y, w_i, h_i, area = stats[i]

        batas = 2
        touches_corner = (
            (x <= batas and y <= batas) or
            (x + w_i >= w_img - batas and y <= batas) or
            (x <= batas and y + h_i >= h_img - batas) or
            (x + w_i >= w_img - batas and y + h_i >= h_img - batas)
        )
        if touches_corner:
            continue

        BORDER = 5
        if (y <= BORDER or y + h_i >= h_img - BORDER or 
            x <= BORDER or x + w_i >= w_img - BORDER):
            continue

        aspect_ratio = h_i / (w_i + 1e-5)

        if (area > 3000 and 0.5 < aspect_ratio < 6.0):
            lung_candidates.append((i, area))

    top2 = sorted(lung_candidates, key=lambda x: x[1], reverse=True)[:2]
    for i, _ in top2:
        mask_top2[labels == i] = 255

    # -----------------------------
    # 10. Contours
    # -----------------------------
    contours, _ = cv2.findContours(mask_top2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_contours = [cv2.convexHull(cnt) for cnt in contours]

    img_contour = img.copy()
    cv2.drawContours(img_contour, hull_contours, -1, (0,0,255), 2)
    steps.append(("10. Contours", img_contour, None))

    # -----------------------------
    # 11. Overlay
    # -----------------------------
    hull_mask = np.zeros_like(mask_top2)
    cv2.drawContours(hull_mask, hull_contours, -1, 255, -1)

    color_mask = np.zeros_like(img)
    color_mask[hull_mask == 255] = (0,255,0)

    overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
    steps.append(("11. Overlay", overlay, None))

    # -----------------------------
    # 12. Final Masked Lungs
    # -----------------------------
    masked = cv2.bitwise_and(img, img, mask=hull_mask)
    steps.append(("12. Masked Lungs", masked, None))

    return masked,steps

def detect_lungs_meanshift(image_path,outdir="test"):
    
    resize_to=512

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Gambar tidak ditemukan!")

    img = cv2.resize(img, (resize_to, resize_to))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(3,3))
    clahe_img = clahe.apply(gray)

    clahe_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
    ms = cv2.pyrMeanShiftFiltering(clahe_bgr, sp=15, sr=30)

    gray_ms = cv2.cvtColor(ms, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray_ms, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    h, w = morphed.shape
    middle = w // 2

    left_half = morphed[:, :middle]
    right_half = morphed[:, middle:]

    def extract_largest_component(binary_mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        if num_labels <= 1:
            return np.zeros_like(binary_mask)

        areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
        largest_label = max(areas, key=lambda x: x[1])[0]

        mask = np.zeros_like(binary_mask)
        mask[labels == largest_label] = 255
        return mask

    left_lung = extract_largest_component(left_half)
    right_lung = extract_largest_component(right_half)

    mask_full = np.zeros_like(morphed)
    mask_full[:, :middle] = left_lung
    mask_full[:, middle:] = right_lung

    color_mask = np.zeros_like(img)
    color_mask[mask_full == 255] = (0, 255, 0)  
    overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)

    return {
        "img": img,
        "clahe": clahe_img,
        "mean_shift": gray_ms,
        "thresh": thresh,
        "morphed": morphed,
        "left_lung": left_lung,
        "right_lung": right_lung,
        "mask_full": mask_full,
        "overlay": overlay
    }