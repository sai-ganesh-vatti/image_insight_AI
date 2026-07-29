from PIL import Image
import streamlit as st
import time
import os
import base64
import io
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path

# -------------------- API SETUP --------------------
# Force load_dotenv to look in the exact directory where classification.py is saved

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

st.write("Looking for .env at:", env_path)
st.write("File exists?:", env_path.exists())

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found in .env file")
    st.stop()

st.write("API Key loaded successfully!")
client = Groq(api_key=api_key)    

# Print active Groq models to terminal on app start
active_models = [m.id for m in client.models.list().data]
print("Active Groq Models:", active_models)

# -------------------- STREAMLIT UI --------------------
# ... (rest of your UI code remains exactly the same)
# -------------------- STREAMLIT UI --------------------
st.title("Welcome to Vasagan's Streamlit Journey")
st.subheader("This is Image Insight Agent App")

left_col, right_col = st.columns(2)

with left_col:
    st.write("This column is uploading section. Please upload your image here.")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["jpg", "jpeg", "png"]
    )

    select_area = st.selectbox(
        "Select the field you want to classify in",
        [
            "textual",
            "object detection",
            "general image",
            "famous personality"
        ]
    )

    select_model = st.radio(
        "Select model variant",
        [
            "fast vision",
            "analytical / detailed vision"
        ]
    )

with right_col:

    st.write("This column is preview section.")

    if uploaded_file is not None:

        img = Image.open(uploaded_file)
        img.thumbnail((300, 300))

        st.image(img, caption="Uploaded Image Preview")

        st.write(f"File Name : {uploaded_file.name}")
        st.write(f"File Size : {uploaded_file.size} bytes")

        click = st.button("Run the Insight Agent")

        if click:
            with st.spinner("Analyzing image..."):
                time.sleep(1)

                # 1. Convert image to Base64
                buffer = io.BytesIO()
                # Convert RGBA images to RGB before saving as JPEG
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(buffer, format="JPEG")
                compressed_bytes = buffer.getvalue()
                base64_image = base64.b64encode(compressed_bytes).decode("utf-8")

                # 2. Select model & set prompt instruction based on user choice
                model_name = "qwen/qwen3.6-27b"
                
                if select_model == "fast vision":
                    detail_instruction = (
                        "Provide a 2-bullet point summary under 50 words, "
                        "followed by a bolded final insight in a few words (e.g., **Final Insight: Shri Radha**)."
                    )
                else:
                    detail_instruction = (
                        "Provide a detailed analysis explaining cultural/historical context, "
                        "iconography, and notable visual elements in 3-4 bullet points, "
                        "followed by a bolded final insight in a few words."
                    )

                prompt = f"""
                Analyze this image with a focus on: {select_area}.
                {detail_instruction}
                """

                # 3. Send request to Groq API
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    }
                                ]
                            }
                        ]
                    )

                    st.success("Analysis Complete!")
                    st.subheader("Result")
                    st.write(response.choices[0].message.content)

                except Exception as e:
                    st.error(f"Error during API execution: {e}")

    else:
        st.write("No file uploaded yet. I'm waiting for it.")