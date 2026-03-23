import numpy as np
from PIL import Image
import io
import base64
from itertools import combinations
from scipy.special import comb
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Combinatorial inkjet printing algorithm from publication
# https://pubs.rsc.org/en/content/articlehtml/2018/a0/d1ta08841f
def combs(a, r):
    """
    Return successive r-length combinations of elements in the array a.
    Faster than array(list(combinations(a, r)))
    """
    a = np.asarray(a)
    dt = np.dtype([('', a.dtype)] * r)
    b = np.fromiter(combinations(a, r), dt)
    return b.view(a.dtype).reshape(-1, r)

def droplet_position_optimization(n, k):
    """
    Function for generating a base matrix with side n and filling k.
    Optimizes droplet positions to maximize minimum distance between droplets.
    """
    x, y = np.indices((n, n))
    pos = combs(np.arange(n ** 2), k)
    x_base_pos = x.flatten()[pos]
    y_base_pos = y.flatten()[pos]
    points = comb(n ** 2, k, exact=True)
    x_pos = np.zeros((points, k * 9))
    y_pos = np.zeros((points, k * 9))
    count = 0
    for y_rep in [-1, 0, 1]:
        for x_rep in [-1, 0, 1]:
            roi = slice(k * count, k * (count + 1))
            x_pos[:, roi] = x_base_pos + (x_rep * n)
            y_pos[:, roi] = y_base_pos + (y_rep * n)
            count += 1
    x_pos = np.tile(x_pos, (k, 1, 1))
    y_pos = np.tile(y_pos, (k, 1, 1))
    distances = np.sqrt(
        np.power(x_pos - np.broadcast_to(x_base_pos, (k * 9, points, k)).T, 2)
        + np.power(y_pos - np.broadcast_to(y_base_pos, (k * 9, points, k)).T, 2)
    )
    distances[np.where(distances < 1)] = (2 * n) ** 2
    min_dist = np.min(distances, axis=2)
    worst_point = np.argmin(min_dist, axis=0)
    if np.max(min_dist[worst_point, np.arange(points)]) > 1:
        best_option_index = np.argmax(min_dist[worst_point, np.arange(points)])
    else:
        worst_min_dist = np.min(min_dist, axis=0)
        min_dist_copy = min_dist
        min_dist_copy[np.where(min_dist != worst_min_dist)] = 0
        best_option_index = np.argmin(np.sum(min_dist_copy, axis=0))
    return x_base_pos[best_option_index], y_base_pos[best_option_index]

def generate_bases(size):
    """
    Function for generating all base matrices for a certain size.
    Creates optimized patterns for combinatorial printing.
    """
    high = int(np.ceil((size ** 2 + 1) / 2)) - 1
    bases = [np.zeros((size, size), dtype=bool)]
    for fill in range(high):
        best_x, best_y = droplet_position_optimization(size, fill + 1)
        base = np.zeros((size, size), dtype=bool)
        for idx in range(len(best_x)):
            base[best_y[idx], best_x[idx]] = True
        bases.append(base)
    remaining = size ** 2 + 1 - len(bases)
    for idx in range(remaining):
        bases.append(np.invert(bases[remaining - idx - 1]))
    return bases

# Cache for publication bases to avoid regeneration
_publication_bases_cache = {}

def publication_pattern(shape, black_percentage, base_size=4):
    """
    Combinatorial inkjet printing pattern from publication.
    Uses optimized base matrices that maximize minimum distance between droplets.
    
    base_size: 3, 4, or 5 (larger sizes are very slow)
    """
    # Generate or retrieve cached bases
    if base_size not in _publication_bases_cache:
        _publication_bases_cache[base_size] = generate_bases(base_size)
    
    bases = _publication_bases_cache[base_size]
    
    # Select appropriate base based on black percentage
    n_bases = len(bases)
    base_index = int(round(black_percentage / 100.0 * (n_bases - 1)))
    base_index = np.clip(base_index, 0, n_bases - 1)
    
    selected_base = bases[base_index]
    
    # Tile the base across the image
    rows, cols = shape
    n_tiles_y = rows // base_size + 1
    n_tiles_x = cols // base_size + 1
    
    tiled = np.tile(selected_base, (n_tiles_y, n_tiles_x))
    return tiled[:rows, :cols]

def mm_to_pixels(mm, dpi):
    """Convert millimeters to pixels at given DPI"""
    inches = mm / 25.4
    return int(inches * dpi)

def create_bayer_matrix(n):
    """Create Bayer matrix of size n x n (n must be power of 2)"""
    if n == 1:
        return np.array([[0]])
    elif n == 2:
        return np.array([[0, 2], [3, 1]])
    else:
        smaller = create_bayer_matrix(n // 2)
        # Build the larger matrix without normalizing yet
        return np.block([
            [4 * smaller + 0, 4 * smaller + 2],
            [4 * smaller + 3, 4 * smaller + 1]
        ])

def random_dithering(shape, black_percentage, seed=None):
    """Random distribution of black pixels"""
    if seed is not None:
        np.random.seed(seed)
    threshold = black_percentage / 100.0
    return np.random.random(shape) < threshold

def ordered_dithering(shape, black_percentage):
    """Bayer ordered dithering"""
    bayer_size = 8
    bayer = create_bayer_matrix(bayer_size)
    # Normalize to 0-1 range
    bayer = bayer / float(bayer_size * bayer_size)
    
    # Tile the Bayer matrix to cover the image
    rows, cols = shape
    tiled = np.tile(bayer, (rows // bayer_size + 1, cols // bayer_size + 1))
    tiled = tiled[:rows, :cols]
    
    threshold = black_percentage / 100.0
    return tiled < threshold

def halftone_pattern(shape, black_percentage, angle=45):
    """Halftone screen pattern at specified angle"""
    rows, cols = shape
    frequency = 10  # dots per unit
    
    # Create coordinate grid
    y, x = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    
    # Rotate coordinates
    angle_rad = np.radians(angle)
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    
    # Create halftone pattern
    pattern = np.sin(x_rot * 2 * np.pi / frequency) * np.sin(y_rot * 2 * np.pi / frequency)
    
    # Threshold based on black percentage
    threshold = np.percentile(pattern, (1 - black_percentage / 100.0) * 100)
    return pattern <= threshold

def grid_pattern(shape, black_percentage):
    """Grid pattern for mirroring compatibility"""
    rows, cols = shape
    grid_size = 4
    
    # Create base grid
    base = np.zeros((grid_size, grid_size), dtype=bool)
    n_black = int((grid_size * grid_size) * black_percentage / 100.0)
    
    # Fill grid in a pattern that's mirror-compatible
    positions = []
    for i in range(grid_size):
        for j in range(grid_size):
            positions.append((i, j))
    
    # Select positions symmetrically
    positions_sorted = sorted(positions, key=lambda p: p[0] + p[1])
    for i in range(min(n_black, len(positions_sorted))):
        pos = positions_sorted[i]
        base[pos] = True
    
    # Tile to cover image
    tiled = np.tile(base, (rows // grid_size + 1, cols // grid_size + 1))
    return tiled[:rows, :cols]

def checkerboard_pattern(shape, black_percentage):
    """Checkerboard pattern with density control"""
    rows, cols = shape
    
    # Base checkerboard
    row_idx = np.arange(rows)[:, np.newaxis]
    col_idx = np.arange(cols)[np.newaxis, :]
    base_checker = ((row_idx + col_idx) % 2).astype(bool)
    
    # Adjust density
    if black_percentage < 50:
        # Remove some black pixels
        mask = np.random.random(shape) < (black_percentage / 50.0)
        return base_checker & mask
    else:
        # Add black pixels to white areas
        extra_mask = np.random.random(shape) < ((black_percentage - 50) / 50.0)
        return base_checker | (~base_checker & extra_mask)

def poisson_disk_sampling(shape, black_percentage, min_distance=None):
    """
    Poisson disk sampling - guarantees minimum distance between points
    Creates very uniform spatial distribution
    """
    rows, cols = shape
    
    # Calculate number of black pixels needed
    total_pixels = rows * cols
    n_black = int(total_pixels * black_percentage / 100.0)
    
    # Calculate minimum distance if not provided
    if min_distance is None:
        # Estimate based on density
        area_per_point = total_pixels / n_black if n_black > 0 else total_pixels
        min_distance = max(2, int(np.sqrt(area_per_point) * 0.7))
    
    # Initialize grid for fast lookup
    cell_size = min_distance / np.sqrt(2)
    grid_w = int(np.ceil(cols / cell_size))
    grid_h = int(np.ceil(rows / cell_size))
    grid = [[None for _ in range(grid_w)] for _ in range(grid_h)]
    
    def grid_coords(point):
        return int(point[0] / cell_size), int(point[1] / cell_size)
    
    def is_valid(point):
        gx, gy = grid_coords(point)
        # Check nearby cells
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < grid_w and 0 <= ny < grid_h and grid[ny][nx] is not None:
                    other = grid[ny][nx]
                    dist = np.sqrt((point[0] - other[0])**2 + (point[1] - other[1])**2)
                    if dist < min_distance:
                        return False
        return True
    
    # Start with random point
    points = []
    active = []
    
    first_point = (np.random.uniform(0, cols), np.random.uniform(0, rows))
    points.append(first_point)
    active.append(first_point)
    gx, gy = grid_coords(first_point)
    grid[gy][gx] = first_point
    
    # Generate points
    k = 30  # attempts before rejection
    while active and len(points) < n_black:
        idx = np.random.randint(len(active))
        point = active[idx]
        found = False
        
        for _ in range(k):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(min_distance, 2 * min_distance)
            new_point = (
                point[0] + radius * np.cos(angle),
                point[1] + radius * np.sin(angle)
            )
            
            if 0 <= new_point[0] < cols and 0 <= new_point[1] < rows and is_valid(new_point):
                points.append(new_point)
                active.append(new_point)
                gx, gy = grid_coords(new_point)
                grid[gy][gx] = new_point
                found = True
                break
        
        if not found:
            active.pop(idx)
    
    # Convert points to boolean array
    pattern = np.zeros(shape, dtype=bool)
    for x, y in points:
        ix, iy = int(x), int(y)
        if 0 <= iy < rows and 0 <= ix < cols:
            pattern[iy, ix] = True
    
    return pattern

def floyd_steinberg_dither(shape, black_percentage):
    """
    Floyd-Steinberg error diffusion dithering
    Classic algorithm for uniform distribution
    """
    rows, cols = shape
    
    # Create initial grayscale image with target percentage
    threshold_value = (100 - black_percentage) / 100.0
    image = np.full(shape, threshold_value, dtype=float)
    
    # Apply Floyd-Steinberg dithering
    for y in range(rows):
        for x in range(cols):
            old_pixel = image[y, x]
            new_pixel = 1.0 if old_pixel > 0.5 else 0.0
            image[y, x] = new_pixel
            error = old_pixel - new_pixel
            
            # Distribute error to neighbors
            if x + 1 < cols:
                image[y, x + 1] += error * 7/16
            if y + 1 < rows:
                if x > 0:
                    image[y + 1, x - 1] += error * 3/16
                image[y + 1, x] += error * 5/16
                if x + 1 < cols:
                    image[y + 1, x + 1] += error * 1/16
    
    return image < 0.5

def void_and_cluster(shape, black_percentage):
    """
    Void-and-cluster algorithm for homogeneous distribution
    Iteratively fills voids and removes clusters
    """
    rows, cols = shape
    total_pixels = rows * cols
    n_black = int(total_pixels * black_percentage / 100.0)
    
    # Start with random distribution
    pattern = np.random.random(shape) < (black_percentage / 100.0)
    
    # Create Gaussian filter for detecting voids/clusters
    def gaussian_weights(size=5, sigma=1.5):
        ax = np.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        return kernel / kernel.sum()
    
    kernel = gaussian_weights()
    
    # Refine pattern
    iterations = min(20, max(5, int(np.sqrt(total_pixels) / 10)))
    for _ in range(iterations):
        # Convolve to find local density
        from scipy import ndimage
        if not SCIPY_AVAILABLE:
            break
            
        density = ndimage.convolve(pattern.astype(float), kernel, mode='constant')
        
        # Find largest void (lowest density in empty area)
        void_candidates = ~pattern
        if void_candidates.any():
            void_densities = np.where(void_candidates, density, np.inf)
            void_pos = np.unravel_index(np.argmin(void_densities), shape)
            
            # Find largest cluster (highest density in filled area)
            cluster_candidates = pattern
            if cluster_candidates.any():
                cluster_densities = np.where(cluster_candidates, density, -np.inf)
                cluster_pos = np.unravel_index(np.argmax(cluster_densities), shape)
                
                # Swap
                pattern[void_pos] = True
                pattern[cluster_pos] = False
    
    return pattern

def jittered_grid(shape, black_percentage, jitter_amount=0.4):
    """
    Jittered grid - regular grid with random offsets
    Balances structure with randomness
    """
    rows, cols = shape
    total_pixels = rows * cols
    n_black = int(total_pixels * black_percentage / 100.0)
    
    # Calculate grid spacing
    spacing = max(1, int(np.sqrt(total_pixels / n_black)))
    
    pattern = np.zeros(shape, dtype=bool)
    
    # Place points on jittered grid
    for i in range(0, rows, spacing):
        for j in range(0, cols, spacing):
            # Add random jitter
            jitter_x = int(np.random.uniform(-spacing * jitter_amount, spacing * jitter_amount))
            jitter_y = int(np.random.uniform(-spacing * jitter_amount, spacing * jitter_amount))
            
            x = np.clip(j + jitter_x, 0, cols - 1)
            y = np.clip(i + jitter_y, 0, rows - 1)
            
            pattern[y, x] = True
    
    return pattern

def multi_level_halftone(shape, black_percentage):
    """
    Multi-level halftone using different Bayer matrix sizes
    Combines fine and coarse patterns for better uniformity
    """
    rows, cols = shape
    
    # Use multiple Bayer matrices at different scales
    bayer_4 = create_bayer_matrix(4) / 16.0  # 4*4 = 16
    bayer_8 = create_bayer_matrix(8) / 64.0  # 8*8 = 64
    
    # Tile both matrices
    tiled_4 = np.tile(bayer_4, (rows // 4 + 1, cols // 4 + 1))[:rows, :cols]
    tiled_8 = np.tile(bayer_8, (rows // 8 + 1, cols // 8 + 1))[:rows, :cols]
    
    # Combine with weights
    combined = 0.6 * tiled_8 + 0.4 * tiled_4
    
    threshold = black_percentage / 100.0
    return combined < threshold

def complementary_pattern(shape, black_percentage, pattern_id=0):
    """
    Create complementary patterns that combine when rotated 180° or mirrored
    pattern_id: 0 or 1 for complementary pair
    """
    rows, cols = shape
    
    # Create a structured pattern using Bayer-like approach
    bayer = create_bayer_matrix(8)
    # Normalize to 0-1 range
    bayer = bayer / 64.0  # 8*8 = 64
    tiled = np.tile(bayer, (rows // 8 + 1, cols // 8 + 1))[:rows, :cols]
    
    # For complementary patterns, split the threshold space
    threshold = black_percentage / 100.0
    
    if pattern_id == 0:
        # First pattern: use lower threshold values
        return tiled < threshold
    else:
        # Second pattern: use complementary threshold values
        # This ensures that when combined, they cover the full range
        return tiled >= (1 - threshold)

def blue_noise_pattern(shape, black_percentage):
    """Blue noise pattern (high frequency noise)"""
    rows, cols = shape
    # Generate random noise
    noise = np.random.random(shape)
    
    if SCIPY_AVAILABLE:
        # Apply high-pass filter to create blue noise characteristics
        smoothed = ndimage.gaussian_filter(noise, sigma=1.0)
        blue_noise = noise - smoothed + 0.5
        blue_noise = np.clip(blue_noise, 0, 1)
    else:
        # Fallback to simple high-frequency noise
        blue_noise = noise
    
    threshold = 1 - (black_percentage / 100.0)
    return blue_noise < threshold

def create_bitmap(width_mm, height_mm, dpi, margins, black_percentage, pattern_type='random', **kwargs):
    """
    Create bitmap image with specified parameters
    
    margins: dict with 'top', 'bottom', 'left', 'right' in mm
    pattern_type: 'random', 'ordered', 'blue_noise', 'complementary',
                  'poisson', 'floyd_steinberg', 'void_cluster', 'jittered', 'multi_level', 'publication'
    """
    # Convert dimensions to pixels
    width_px = mm_to_pixels(width_mm, dpi)
    height_px = mm_to_pixels(height_mm, dpi)
    
    margin_top_px = mm_to_pixels(margins['top'], dpi)
    margin_bottom_px = mm_to_pixels(margins['bottom'], dpi)
    margin_left_px = mm_to_pixels(margins['left'], dpi)
    margin_right_px = mm_to_pixels(margins['right'], dpi)
    
    # Create white image
    img_array = np.ones((height_px, width_px), dtype=bool)
    
    # Calculate active area (outside margins)
    active_top = margin_top_px
    active_bottom = height_px - margin_bottom_px
    active_left = margin_left_px
    active_right = width_px - margin_right_px
    
    if active_bottom > active_top and active_right > active_left:
        active_shape = (active_bottom - active_top, active_right - active_left)
        
        # Generate pattern based on type
        if pattern_type == 'random':
            pattern = random_dithering(active_shape, black_percentage)
        elif pattern_type == 'ordered':
            pattern = ordered_dithering(active_shape, black_percentage)
        elif pattern_type == 'blue_noise':
            pattern = blue_noise_pattern(active_shape, black_percentage)
        elif pattern_type == 'complementary':
            pattern_id = kwargs.get('pattern_id', 0)
            pattern = complementary_pattern(active_shape, black_percentage, pattern_id)
        elif pattern_type == 'poisson':
            pattern = poisson_disk_sampling(active_shape, black_percentage)
        elif pattern_type == 'floyd_steinberg':
            pattern = floyd_steinberg_dither(active_shape, black_percentage)
        elif pattern_type == 'void_cluster':
            pattern = void_and_cluster(active_shape, black_percentage)
        elif pattern_type == 'jittered':
            pattern = jittered_grid(active_shape, black_percentage)
        elif pattern_type == 'multi_level':
            pattern = multi_level_halftone(active_shape, black_percentage)
        elif pattern_type == 'publication':
            base_size = kwargs.get('base_size', 4)
            pattern = publication_pattern(active_shape, black_percentage, base_size)
        else:
            pattern = random_dithering(active_shape, black_percentage)
        
        # Invert: True = white, False = black
        img_array[active_top:active_bottom, active_left:active_right] = ~pattern
    
    # Convert to PIL Image (True=255=white, False=0=black)
    img = Image.fromarray(img_array.astype(np.uint8) * 255, mode='L')
    
    return img

def image_to_base64(img, format='PNG'):
    """Convert PIL Image to base64 string for display"""
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"

def invert_image(img, width_mm, height_mm, dpi, margins):
    """
    Invert a bitmap image (black <-> white) but keep margins white
    
    margins: dict with 'top', 'bottom', 'left', 'right' in mm
    """
    img_array = np.array(img)
    inverted_array = img_array.copy()
    
    # Calculate margin pixels
    margin_top_px = mm_to_pixels(margins['top'], dpi)
    margin_bottom_px = mm_to_pixels(margins['bottom'], dpi)
    margin_left_px = mm_to_pixels(margins['left'], dpi)
    margin_right_px = mm_to_pixels(margins['right'], dpi)
    
    height_px, width_px = img_array.shape
    
    # Calculate active area
    active_top = margin_top_px
    active_bottom = height_px - margin_bottom_px
    active_left = margin_left_px
    active_right = width_px - margin_right_px
    
    # Invert only the active area
    if active_bottom > active_top and active_right > active_left:
        inverted_array[active_top:active_bottom, active_left:active_right] = \
            255 - img_array[active_top:active_bottom, active_left:active_right]
    
    return Image.fromarray(inverted_array.astype(np.uint8), mode='L')