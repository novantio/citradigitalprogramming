import streamlit as st
import cv2
import numpy as np
import os

from detect_lungs import detect_lungs_grid2  # fungsi segmentation kamu

st.set_page_config(page_title="Detect Lung Image", layout="wide")

st.title("🫁 Detect Lung Image")
st.write("Upload gambar X-ray untuk melakukan segmentasi paru-paru.")

outdir = "output_lung"
os.makedirs(outdir, exist_ok=True)

uploaded_file = st.file_uploader("Upload gambar", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    # Read file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("📥 Input Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
             caption="Input X-ray")

    # Save temporary
    temp_path = os.path.join(outdir, "temp_input.png")
    cv2.imwrite(temp_path, img)

    if st.button("🔍 Process Image"):
        st.info("Processing image, please wait...")

        try:
            # Run detection function
            masked, steps = detect_lungs_grid2(temp_path, outdir=outdir)

            # Show result
            st.subheader("Final Masked Lungs")
            st.image(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB),
                     caption="Masked Lung Area")

            # Download output
            st.download_button(
                "📥 Download Output",
                data=cv2.imencode(".png", masked)[1].tobytes(),
                file_name="lung_masked.png",
                mime="image/png"
            )

            # Show steps grid
            st.subheader("🧩 Processing Steps (Grid Preview)")

            cols = st.columns(3)
            col_i = 0

            for title, img_step, cmap in steps:
                with cols[col_i]:
                    st.markdown(f"**{title}**")

                    if img_step.ndim == 2:  # grayscale
                        st.image(img_step, channels="GRAY", clamp=True, use_container_width =True)
                    else:  # BGR image
                        st.image(cv2.cvtColor(img_step, cv2.COLOR_BGR2RGB), use_container_width =True)

                col_i = (col_i + 1) % 3
                if col_i == 0:
                    cols = st.columns(3)

        except Exception as e:
            st.error(f"Error: {e}")

else:
    st.info("Silakan upload gambar X-ray untuk memulai.")
