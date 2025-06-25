import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LightSource, LinearSegmentedColormap
import matplotlib.cm as cm
import warnings
import streamlit as st
import tempfile
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="pyvista.jupyter")
warnings.filterwarnings("ignore", message=".*tight_layout.*")

# Configuration parameters
SMOOTH_SIGMA = 3.0
DETAIL_OCTAVES = 4
DETAIL_PERSISTENCE = 0.5
DETAIL_SCALE = 10.0
ELEV = 30
AZIM = -60
INVERT = True
SCALE_TO = (200, 0)

# Create temporary directories for input and output
TEMP_DIR = tempfile.mkdtemp()
INPUT_DIR = os.path.join(TEMP_DIR, 'gen_images')
OUTPUT_DIR = os.path.join(TEMP_DIR, 'Inference_3d Plots')
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Custom terrain colormap
def create_terrain_cmap():
    colors = [
        (0.0, (0.0, 0.0, 0.5)),  # Deep blue (ocean)
        (0.1, (0.0, 0.4, 0.8)),  # Medium blue
        (0.2, (0.2, 0.6, 0.9)),  # Light blue
        (0.3, (0.8, 0.8, 0.2)),  # Sand/beach
        (0.4, (0.1, 0.6, 0.1)),  # Lowland green
        (0.6, (0.4, 0.8, 0.4)),  # Highland green
        (0.7, (0.6, 0.5, 0.3)),  # Low mountains
        (0.8, (0.5, 0.4, 0.3)),  # High mountains
        (0.9, (0.9, 0.9, 0.9)),  # Lower snow
        (1.0, (1.0, 1.0, 1.0))   # Upper snow
    ]
    cdict = {'red': [], 'green': [], 'blue': []}
    for pos, color in colors:
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))
    return LinearSegmentedColormap('terrain_natural', cdict)

terrain_cmap = create_terrain_cmap()

# Load height map from image
def load_height_map(img, scale_to=None, invert=True):
    arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
    if invert:
        arr = 1.0 - arr
    if scale_to is not None:
        hi, lo = scale_to
        arr = arr * (hi - lo) + lo
    return arr

# Smooth terrain
def smooth_terrain(height_map, sigma=1.0):
    return gaussian_filter(height_map, sigma=sigma)

# Add fractal detail with fallback
def add_opensimplex_detail(height_map, octaves=3, persistence=0.5, scale=10.0):
    try:
        from opensimplex import OpenSimplex
        h, w = height_map.shape
        detail = np.zeros((h, w))
        noise_gen = OpenSimplex(seed=np.random.randint(0, 1000000))
        for y in range(h):
            for x in range(w):
                value = 0
                amplitude = 1.0
                frequency = 1.0
                for _ in range(octaves):
                    nx = x / w * frequency * 5
                    ny = y / h * frequency * 5
                    value += noise_gen.noise2(nx, ny) * amplitude
                    amplitude *= persistence
                    frequency *= 2
                detail[y, x] = value
        detail = (detail - detail.min()) / (detail.max() - detail.min()) * scale
        return height_map + detail
    except ImportError:
        st.warning("OpenSimplex not installed. Using NumPy fallback.")
        return add_numpy_fractal_detail(height_map, octaves, persistence, scale)

def add_numpy_fractal_detail(height_map, octaves=3, persistence=0.5, scale=10.0):
    h, w = height_map.shape
    detail = np.zeros((h, w))
    for octave in range(octaves):
        freq = 2**octave
        amp = persistence**octave
        octave_size = (max(h // freq, 1), max(w // freq, 1))
        noise = np.random.rand(*octave_size)
        from scipy.ndimage import zoom
        zoomed = zoom(noise, (h/noise.shape[0], w/noise.shape[1]), order=1)
        detail += zoomed * amp
    detail = (detail - detail.min()) / (detail.max() - detail.min()) * scale
    return height_map + detail

# Enhance terrain
def enhance_terrain(height_map, smooth_sigma=1.0, add_detail=True, 
                   detail_octaves=3, detail_persistence=0.5, detail_scale=10.0):
    terrain = smooth_terrain(height_map, sigma=smooth_sigma)
    if add_detail:
        terrain = add_opensimplex_detail(terrain, octaves=detail_octaves,
                                         persistence=detail_persistence, scale=detail_scale)
    terrain = np.clip(terrain, 0, 200)
    if terrain.max() > terrain.min():
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min()) * 200
    else:
        terrain = np.zeros_like(terrain)
    return terrain

# Plot 3D terrain with Matplotlib
def plot_matplotlib_3d(z, title="3D Terrain", cmap=terrain_cmap, elev=30, azim=-60, 
                       with_hillshade=True, output_prefix=""):
    filename = os.path.join(OUTPUT_DIR, f"{output_prefix}_3D_Terrain.png")
    cmap_obj = cmap if isinstance(cmap, LinearSegmentedColormap) else cm.get_cmap(cmap)
    h, w = z.shape
    x = np.arange(w)
    y = np.arange(h)
    x, y = np.meshgrid(x, y)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    norm_z = (z - z.min()) / (z.max() - z.min() if z.max() > z.min() else 1)
    if with_hillshade:
        ls = LightSource(azdeg=315, altdeg=45)
        rgb = ls.shade(z, cmap=cmap_obj, vert_exag=0.3, blend_mode='soft')
        surf = ax.plot_surface(x, y, z, facecolors=rgb, linewidth=0, antialiased=True, 
                              rcount=200, ccount=200)
    else:
        surf = ax.plot_surface(x, y, z, cmap=cmap_obj, linewidth=0, antialiased=True, 
                              rcount=200, ccount=200, vmin=0, vmax=200)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Elevation")
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Elevation", fontsize=12)
    ax.set_zlim(0, 200)
    ax.view_init(elev=elev, azim=azim)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

# Process images
def process_images(uploaded_files, max_images):
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    
    # Save uploaded files to INPUT_DIR
    image_files = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.lower().endswith(valid_extensions):
            file_path = os.path.join(INPUT_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_files.append(uploaded_file.name)
    
    if not image_files:
        st.error(f"No valid images uploaded.")
        return
    
    # Limit to max_images
    image_files = sorted(image_files)[:max_images]
    st.write(f"Processing {len(image_files)} images...")
    
    # Process each image
    for image_name in image_files:
        image_path = os.path.join(INPUT_DIR, image_name)
        base_name = os.path.splitext(image_name)[0]
        
        try:
            # Load and enhance the image
            img = Image.open(image_path)
            z_original = load_height_map(img, scale_to=SCALE_TO, invert=INVERT)
            z_enhanced = enhance_terrain(z_original, smooth_sigma=SMOOTH_SIGMA, add_detail=True,
                                         detail_octaves=DETAIL_OCTAVES, detail_persistence=DETAIL_PERSISTENCE,
                                         detail_scale=DETAIL_SCALE)
            
            # Generate and save 3D terrain plot
            output_file = plot_matplotlib_3d(z_enhanced, title=f"Enhanced 3D Terrain for {base_name}", 
                                             cmap=terrain_cmap, elev=ELEV, azim=AZIM, with_hillshade=True, 
                                             output_prefix=base_name)
            
            # Display the result
            st.write(f"Processed: {image_name}")
            st.image(output_file, caption=f"3D Terrain for {image_name}", use_container_width=True)
            
        except Exception as e:
            st.error(f"Error processing {image_name}: {e}")
            continue

# Streamlit app layout
st.title("3D Terrain Generator")
st.markdown("Upload images to generate 3D terrain visualizations.")

# User input for number of images to process
max_images = st.number_input("Number of images to process", min_value=1, max_value=10, value=2, step=1)

# File uploader
uploaded_files = st.file_uploader("Upload images (PNG, JPG, JPEG, TIF, TIFF)", 
                                 type=['png', 'jpg', 'jpeg', 'tif', 'tiff'], 
                                 accept_multiple_files=True)

# Process button
if st.button("Generate 3D Terrains"):
    if uploaded_files:
        process_images(uploaded_files, max_images)
    else:
        st.error("Please upload at least one image.")

# Clean up temporary directory on app exit
def cleanup():
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

import atexit
atexit.register(cleanup)