import streamlit as st
import numpy as np
import colorsys

from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# -------------------
# Helper functions
# -------------------

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % tuple(map(int, rgb))


def hex_to_rgb(hex_color):

    hex_color = hex_color.lstrip("#")

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return (r, g, b)


def complementary_color(hex_color):

    rgb = hex_to_rgb(hex_color)

    r = rgb[0] / 255
    g = rgb[1] / 255
    b = rgb[2] / 255

    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    new_h = (h + 0.5) % 1

    r2, g2, b2 = colorsys.hsv_to_rgb(
        new_h,
        s,
        v
    )

    return rgb_to_hex(
        (
            r2 * 255,
            g2 * 255,
            b2 * 255
        )
    )


def analogous_colors(hex_color):

    rgb = hex_to_rgb(hex_color)

    r = rgb[0] / 255
    g = rgb[1] / 255
    b = rgb[2] / 255

    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    suggestions = []

    for shift in [0.08, -0.08]:

        new_h = (h + shift) % 1

        r2, g2, b2 = colorsys.hsv_to_rgb(
            new_h,
            s,
            v
        )

        suggestions.append(
            rgb_to_hex(
                (
                    r2 * 255,
                    g2 * 255,
                    b2 * 255
                )
            )
        )

    return suggestions


def get_main_color(image):

    img_array = np.array(image)

    pixels = img_array.reshape(-1, 3)

    kmeans = KMeans(
        n_clusters=5,
        random_state=42
    )

    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_

    labels = kmeans.labels_

    unique, counts = np.unique(
        labels,
        return_counts=True
    )

    main_color_index = unique[np.argmax(counts)]

    main_color = colors[main_color_index]

    return rgb_to_hex(main_color)


# -------------------
# App
# -------------------

st.title("🎨 Fashion Palette AI")

st.write(
    "Upload a clothing image and get layering color suggestions "
    "based on color theory."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    main_hex = get_main_color(image)

    match_color = complementary_color(main_hex)

    analogs = analogous_colors(main_hex)

    recommended_colors = [match_color] + analogs

    st.subheader("Main Color")

    st.code(main_hex)

    st.subheader("Recommended Colors")

    for color in recommended_colors:
        st.write(color)

    all_colors = [main_hex] + recommended_colors

    fig, ax = plt.subplots(
        figsize=(8, 2)
    )

    for i, color in enumerate(all_colors):

        ax.fill_between(
            [i, i + 1],
            0,
            1,
            color=color
        )

    ax.set_xlim(
        0,
        len(all_colors)
    )

    ax.axis("off")

    st.pyplot(fig)

    st.info(
        "These colors work best for layering pieces such as "
        "shirts, overshirts, jackets, hoodies, or knitwear."
    )
