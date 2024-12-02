import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from skimage.metrics import structural_similarity as ssim
import pandas as pd
from fpdf import FPDF
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Created By Scienceveli

# قراءة الإعدادات من ملف config.json
def load_config(config_path="config.json"):
    with open(config_path, "r") as file:
        return json.load(file)

config = load_config()

# دالة حساب SSIM
def calculate_ssim(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(img1_gray, img2_gray, full=True)
    return score

# دالة حساب ORB
def calculate_orb_similarity(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches)

# دالة حساب SIFT
def calculate_sift_similarity(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # تطبيق فلتر نسبة منخفضة
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return len(good_matches)

# دالة فلترة الصور بناءً على الجودة
def filter_images_by_resolution(images, min_resolution=(720, 720)):
    filtered_images = []
    for img_path in images:
        img = cv2.imread(img_path)
        if img.shape[0] >= min_resolution[0] and img.shape[1] >= min_resolution[1]:
            filtered_images.append(img_path)
    return filtered_images

# دالة لتحديد أفضل N صورة
def identify_top_n_images(images, n=100, weights=None):
    weights = weights or {"ssim": 0.4, "orb": 0.3, "sift": 0.3}
    total_scores = {img_path: 0 for img_path in images}

    def compare_images(img_path):
        current_image = cv2.imread(img_path)
        for compare_img_path in images:
            if img_path == compare_img_path:
                continue
            compare_image = cv2.imread(compare_img_path)
            ssim_score = calculate_ssim(current_image, compare_image)
            orb_score = calculate_orb_similarity(current_image, compare_image)
            sift_score = calculate_sift_similarity(current_image, compare_image)
            combined_score = (weights["ssim"] * ssim_score) + \
                             (weights["orb"] * orb_score / 1000) + \
                             (weights["sift"] * sift_score / 500)
            total_scores[img_path] += combined_score

    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(compare_images, images)

    sorted_images = sorted(total_scores, key=total_scores.get, reverse=True)
    return sorted_images[:n]

# إنشاء تقرير PDF
def create_pdf_report(image_paths, output_path="report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Image Similarity Report", ln=True, align="C")

    for img_path in image_paths:
        pdf.add_page()
        pdf.image(img_path, x=10, y=30, w=180)
        pdf.ln(85)
        pdf.cell(0, 10, txt=f"Image: {os.path.basename(img_path)}", ln=True)

    pdf.output(output_path)

# كتابة البيانات إلى ملف CSV
def write_csv(image_paths, output_path="image_scores.csv"):
    df = pd.DataFrame({'Image Path': image_paths})
    df.to_csv(output_path, index=False)

# إنشاء رسم بياني لتحليل النتائج
def create_similarity_plot(scores, output_path="similarity_plot.png"):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(scores)), list(scores.values()), color='blue')
    plt.xlabel("Image Index")
    plt.ylabel("Similarity Score")
    plt.title("Image Similarity Scores")
    plt.savefig(output_path)

# المسار الرئيسي
if __name__ == "__main__":
    image_directory = config.get("image_directory", "path/to/your/images")
    image_paths = [os.path.join(image_directory, fname) for fname in os.listdir(image_directory)
                   if fname.endswith(('jpg', 'png', 'jpeg'))]

    if not image_paths:
        print("No images found in the specified directory.")
    else:
        
        filtered_images = filter_images_by_resolution(image_paths, tuple(config["min_image_resolution"]))

        if not filtered_images:
            print("No images meet the minimum resolution requirement.")
        else:
            
            top_images = identify_top_n_images(filtered_images, n=config["num_images"], weights=config["weights"])

            
            write_csv(top_images)
            create_pdf_report(top_images)

           
            scores = {img: i+1 for i, img in enumerate(top_images)}
            create_similarity_plot(scores)

            print("Processing complete. CSV, PDF report, and plot generated.")
