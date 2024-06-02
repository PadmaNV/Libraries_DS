import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import numpy as np

# Data for visualization
libraries = [
    "NumPy", "Pandas", "Matplotlib", "Seaborn", "Scikit-learn",
    "TensorFlow", "Keras", "PyTorch", "SciPy", "Statsmodels",
    "NLTK", "OpenCV", "Beautiful Soup", "XGBoost"
]

# Popularity scores for each library (example values, can be adjusted)
popularity = [95, 90, 85, 75, 92, 88, 80, 87, 78, 70, 65, 82, 60, 84]

# Normalize popularity scores to be out of 100
max_popularity = max(popularity)
normalized_popularity = [score / max_popularity * 100 for score in popularity]

# Paths to the logos
logo_paths = {
    "NumPy": "./logo/NumPy.png",
    "Pandas": "./logo/Pandas.png",
    "Matplotlib": "./logo/Matplotlib-logo.png",
    "Seaborn": "./logo/Seaborn_logo.png",
    "Scikit-learn": "./logo/Scikit.png",
    "TensorFlow": "./logo/Tensorflow.png",
    "Keras": "./logo/Keras_logo.png",
    "PyTorch": "./logo/Pytorch_logo.png",
    "SciPy": "./logo/SciPy_logo.png",
    "Statsmodels": "./logo/Statsmodels.png",
    "NLTK": "./logo/NLTK.png",
    "OpenCV": "./logo/OpenCV.png",
    "Beautiful Soup": "./logo/Beautiful Soup.png",
    "XGBoost": "./logo/XGBoost_logo.png"
}

# Descriptions for each library
descriptions = {
    "NumPy": "Numerical computing with multi-dimensional arrays and matrices.",
    "Pandas": "Data manipulation and analysis with DataFrame and Series.",
    "Matplotlib": "Creating static, animated, and interactive visualizations.",
    "Seaborn": "Statistical data visualization based on Matplotlib.",
    "Scikit-learn": "Tools for machine learning, including classification and regression.",
    "TensorFlow": "Numerical computation and deep learning.",
    "Keras": "High-level neural networks API, running on top of TensorFlow.",
    "PyTorch": "Deep learning with dynamic computational graphs.",
    "SciPy": "Scientific and technical computing.",
    "Statsmodels": "Statistical modeling and hypothesis testing.",
    "NLTK": "Natural language processing toolkit.",
    "OpenCV": "Computer vision and image processing.",
    "Beautiful Soup": "Parsing HTML and XML documents.",
    "XGBoost": "Optimized distributed gradient boosting library."
}

# Function to open, resize and add white background to image
def open_and_resize_image(path):
    img = Image.open(path).convert("RGBA")
    img = img.resize((80, 60), Image.LANCZOS)
    # Add white background
    img_with_bg = Image.new("RGBA", img.size, "WHITE")
    img_with_bg.paste(img, (0, 0), img)
    return img_with_bg

# Open logos
logos = {name: open_and_resize_image(path) for name, path in logo_paths.items()}

# Sort libraries by popularity
libraries_sorted = [x for _, x in sorted(zip(normalized_popularity, libraries))]
normalized_popularity_sorted = sorted(normalized_popularity)

# Generate a color map
colors = plt.cm.viridis(np.linspace(0, 1, len(libraries_sorted)))

fig, ax = plt.subplots(figsize=(12, 9))

# Create the horizontal bar chart
bars = ax.barh(libraries_sorted, normalized_popularity_sorted, color=colors, height=0.7)

ax.set_xlabel('Normalized Popularity Score')
ax.set_title('Normalized Popularity of Key Data Science Libraries')
ax.set_yticks(range(len(libraries_sorted)))
ax.set_yticklabels(libraries_sorted, fontsize=14, fontweight='bold')

# Add the text descriptions and logos
for i, (bar, library) in enumerate(zip(bars, libraries_sorted)):
    # Calculate inverse color for the description text
    bar_color = bar.get_facecolor()
    inv_color = (1.0 - bar_color[0], 1.0 - bar_color[1], 1.0 - bar_color[2], 1.0)
    
    # Add the popularity score
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2, f"{normalized_popularity_sorted[i]:.1f}", va='center', color='blue')
    
    # Add the description text
    ax.text(5, bar.get_y() + bar.get_height() / 2, descriptions[library], 
            va='center', ha='left', color=inv_color, fontsize=10, wrap=True)
    
    # Add the logo image
    if library in logos:
        imagebox = OffsetImage(logos[library], zoom=0.8, resample=True)
        ab = AnnotationBbox(imagebox, (normalized_popularity_sorted[i] - 5, i), frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)

plt.tight_layout()
plt.show()
