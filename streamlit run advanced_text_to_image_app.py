import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
from transformers import pipeline
import numpy as np

# Function to create gradient background
def create_gradient(width, height, start_color, end_color):
    base = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        blend = i / height
        r = int((1 - blend) * start_color[0] + blend * end_color[0])
        g = int((1 - blend) * start_color[1] + blend * end_color[1])
        b = int((1 - blend) * start_color[2] + blend * end_color[2])
        base[i, :, :] = [r, g, b]
    return Image.fromarray(base)

# Function to convert text to image with advanced options
def text_to_image(text, font_path, font_size, text_color, bg_color, image_width, image_height, shape, shape_color, overlay=None):
    if bg_color == "gradient":
        start_color = st.color_picker("Gradient start color", "#FF5733")
        end_color = st.color_picker("Gradient end color", "#33FF57")
        image = create_gradient(image_width, image_height, tuple(int(start_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)),
                                tuple(int(end_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
    else:
        image = Image.new("RGB", (image_width, image_height), bg_color)
    draw = ImageDraw.Draw(image)

    # Overlay image
    if overlay:
        overlay_image = Image.open(overlay).resize((image_width, image_height))
        image.paste(overlay_image, (0, 0), overlay_image)

    # Load font
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    # Calculate text size and position
    text_width, text_height = draw.textsize(text, font=font)
    x = (image_width - text_width) // 2
    y = (image_height - text_height) // 2
    draw.text((x, y), text, fill=text_color, font=font)

    # Draw shapes
    if shape == "circle":
        draw.ellipse([(image_width//4, image_height//4), (3*image_width//4, 3*image_height//4)], outline=shape_color, width=5)
    elif shape == "rectangle":
        draw.rectangle([(image_width//4, image_height//4), (3*image_width//4, 3*image_height//4)], outline=shape_color, width=5)

    return image

# Initialize AI text summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Streamlit App
st.title("Advanced Text to Image Generator")
st.write("Generate professional images with text, gradients, shapes, and AI features.")

# User Inputs
text = st.text_area("Enter your text", "Streamlit takes apps to the next level!")
use_ai = st.checkbox("Summarize text using AI")
if use_ai:
    summarized_text = summarizer(text, max_length=30, min_length=10, do_sample=False)
    text = summarized_text[0]["summary_text"]
    st.success(f"Summarized Text: {text}")

font_size = st.slider("Font size", 10, 100, 30)
image_width = st.slider("Image width", 300, 1200, 800)
image_height = st.slider("Image height", 300, 1200, 600)
text_color = st.color_picker("Pick text color", "#000000")
bg_color = st.selectbox("Background type", ["solid", "gradient"], index=0)
solid_color = st.color_picker("Solid background color", "#FFFFFF") if bg_color == "solid" else None
shape = st.selectbox("Add a shape", ["none", "circle", "rectangle"])
shape_color = st.color_picker("Pick shape color", "#FF0000") if shape != "none" else None

# Font Upload
uploaded_font = st.file_uploader("Upload a font file (e.g., .ttf)", type=["ttf", "otf"])
font_path = uploaded_font if uploaded_font else "arial.ttf"

# Image Overlay
overlay_image = st.file_uploader("Upload an overlay image (optional)", type=["png", "jpg", "jpeg"])

# Generate and Display Image
if st.button("Generate Image"):
    image = text_to_image(text, font_path, font_size, text_color, solid_color or bg_color, image_width, image_height, shape, shape_color, overlay_image)
    st.image(image, caption="Generated Image", use_column_width=True)

    # Download Option
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    st.download_button("Download Image", data=buf, file_name="advanced_text_image.png", mime="image/png")
