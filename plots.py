import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from destruction_utilities import read_raster
    
def plot_raster_with_window(profile_window, image_path=None):
    """Plots a raster image with a rectangle showing the given window."""
    img = read_raster(image_path, dtype='uint8')  # H x W x C (or H x W)

    fig, ax = plt.subplots()
    ax.imshow(img, origin='upper')  # pixel coords: (col=x, row=y)

    rect = Rectangle(
        (profile_window.col_off, profile_window.row_off),  # (x, y) in pixels
        profile_window.width, profile_window.height,
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)

    # show the entire image
    H, W = img.shape[:2]
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)              # invert y to match image coordinates
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()