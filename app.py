import streamlit as st
from PIL import Image
from model import AIModel

def main():
    st.title('AI Try-On App')

    st.markdown('### Upload Images')
    uploaded_profile_img = st.file_uploader("Upload Your Profile Image", type=["jpg", "png", "jpeg"])
    uploaded_cloth_img = st.file_uploader("Upload Cloth Image", type=["jpg", "png", "jpeg"])

    if uploaded_profile_img and uploaded_cloth_img:
        profile_img = Image.open(uploaded_profile_img)
        cloth_img = Image.open(uploaded_cloth_img)

        st.image(profile_img, caption='Uploaded Profile Image', use_column_width=True)
        st.image(cloth_img, caption='Uploaded Cloth Image', use_column_width=True)

        # Process images using your AIModel
        ai_model = AIModel(source_img_path=uploaded_profile_img, dest_img_path=uploaded_cloth_img)

        # Apply CMate model or other functionality
        final_img = ai_model.load_and_apply_cmate()

        if final_img is not None:
            st.image(final_img, caption='Final Combined Image', use_column_width=True)
        else:
            st.error("Error applying CMate model. Please check logs for details.")

if __name__ == '__main__':
    main()
