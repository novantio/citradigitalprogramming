import pickle
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import streamlit as st
from detect_lungs import detect_lungs_grid2
import os

# Fungsi ekstraksi fitur
def extract_glcm_features(image):
    gray = cv2.resize(image, (256, 256))
    glcm = graycomatrix(gray, [5], [0], 256, symmetric=True, normed=True)
    return [
        graycoprops(glcm, 'contrast')[0,0],
        graycoprops(glcm, 'homogeneity')[0,0],
        graycoprops(glcm, 'energy')[0,0],
        graycoprops(glcm, 'correlation')[0,0],
        graycoprops(glcm, 'dissimilarity')[0,0],
        graycoprops(glcm, 'ASM')[0,0]
    ]

# Load model
with open(r"D:\project\python\2025\tbcdetect\_streamlit\tb_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

# Streamlit file uploader
uploaded_file = st.file_uploader("Upload CXR image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    outdir = "output_lung"
    os.makedirs(outdir, exist_ok=True)

    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
             caption="Input X-ray")
    ##apply feature
    temp_path = os.path.join(outdir, "temp_input.png")
    cv2.imwrite(temp_path, img)
    
    masked, steps = detect_lungs_grid2(temp_path, outdir=outdir)
    st.image(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB),
             caption="Output masked X-ray")
             
    if masked.ndim == 3 and masked.shape[2] == 3:
        masked_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    else:
        masked_gray = masked
    
    #glcm
    features = np.array(extract_glcm_features(masked_gray)).reshape(1, -1)
    
    pred = clf.predict(features)[0]
    st.write("Prediction:", "TB" if pred==1 else "Normal")
