import ipywidgets as widgets
from IPython.display import display, HTML
from io import BytesIO

def create_size_inputs(default_width=25, default_height=25):
    """Create width and height input widgets"""
    width_input = widgets.FloatText(
        value=default_width,
        description='Width (mm):',
        disabled=False,
        style={'description_width': '100px'}
    )
    
    height_input = widgets.FloatText(
        value=default_height,
        description='Height (mm):',
        disabled=False,
        style={'description_width': '100px'}
    )
    
    size_box = widgets.VBox([
        widgets.HTML("<h3>Image Size</h3>"),
        width_input,
        height_input
    ])
    
    return size_box, width_input, height_input

def create_margin_inputs(default_margin=3):
    """Create margin input widgets with auto-sync option"""
    margin_top = widgets.FloatText(
        value=default_margin,
        description='Top (mm):',
        disabled=False,
        style={'description_width': '100px'}
    )
    
    margin_bottom = widgets.FloatText(
        value=default_margin,
        description='Bottom (mm):',
        disabled=False,
        style={'description_width': '100px'}
    )
    
    margin_left = widgets.FloatText(
        value=default_margin,
        description='Left (mm):',
        disabled=False,
        style={'description_width': '100px'}
    )
    
    margin_right = widgets.FloatText(
        value=default_margin,
        description='Right (mm):',
        disabled=False,
        style={'description_width': '100px'}
    )
    
    sync_margins = widgets.Checkbox(
        value=True,
        description='Sync all margins',
        disabled=False
    )
    
    def on_margin_change(change):
        if sync_margins.value:
            value = change['new']
            if change['owner'] == margin_top:
                margin_bottom.value = value
                margin_left.value = value
                margin_right.value = value
    
    margin_top.observe(on_margin_change, names='value')
    
    margin_box = widgets.VBox([
        widgets.HTML("<h3>Margins</h3>"),
        sync_margins,
        margin_top,
        margin_bottom,
        margin_left,
        margin_right
    ])
    
    margins = {
        'top': margin_top,
        'bottom': margin_bottom,
        'left': margin_left,
        'right': margin_right
    }
    
    return margin_box, margins

def create_dpi_input(default_dpi=350):
    """Create DPI input widget"""
    dpi_input = widgets.IntText(
        value=default_dpi,
        description='DPI:',
        disabled=False,
        style={'description_width': '100px'}
    )
    
    dpi_box = widgets.VBox([
        widgets.HTML("<h3>Resolution</h3>"),
        dpi_input
    ])
    
    return dpi_box, dpi_input

def create_pattern_controls(default_percentage=50):
    """Create pattern type and percentage controls"""
    pattern_type = widgets.Dropdown(
        options=[
            ('Random', 'random'),
            ('Ordered (Bayer Matrix)', 'ordered'),
            ('Blue Noise', 'blue_noise'),
            ('Complementary A', 'complementary_a'),
            ('Complementary B', 'complementary_b'),
            ('Poisson Disk Sampling', 'poisson'),
            ('Floyd-Steinberg Dither', 'floyd_steinberg'),
            ('Void-and-Cluster', 'void_cluster'),
            ('Jittered Grid', 'jittered'),
            ('Multi-Level Halftone', 'multi_level'),
            ('Publication (Base 3)', 'publication_3'),
            ('Publication (Base 4)', 'publication_4'),
            ('Publication (Base 5)', 'publication_5')
        ],
        value='random',
        description='Pattern:',
        style={'description_width': '120px'}
    )
    
    black_percentage = widgets.FloatSlider(
        value=default_percentage,
        min=0,
        max=100,
        step=0.5,
        description='Black %:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        style={'description_width': '120px'}
    )
    
    pattern_info = widgets.HTML(
        "<p style='font-size: 11px; color: #666; margin-top: 5px;'>"
        "<strong>For chemical mixing:</strong><br>"
        "• <em>Publication</em> - Optimized for combinatorial printing<br>"
        "• <em>Poisson/Void-and-Cluster</em> - Best uniformity<br>"
        "• <em>Floyd-Steinberg</em> - Classic, proven<br>"
        "• <em>Blue Noise</em> - Minimizes clustering<br>"
        "• <em>Complementary A+B</em> - Designed for superposition"
        "</p>"
    )
    
    pattern_box = widgets.VBox([
        widgets.HTML("<h3>Pattern Settings</h3>"),
        pattern_type,
        black_percentage,
        pattern_info
    ])
    
    return pattern_box, pattern_type, black_percentage

def create_file_format_selector():
    """Create file format selection widget"""
    format_selector = widgets.Dropdown(
        options=['BMP', 'PNG', 'TIFF', 'JPEG'],
        value='BMP',
        description='Format:',
        style={'description_width': '100px'}
    )
    
    include_inverted = widgets.Checkbox(
        value=False,
        description='Also download inverted',
        disabled=False,
        indent=False
    )
    
    format_box = widgets.VBox([
        widgets.HTML("<h3>Output Format</h3>"),
        format_selector,
        include_inverted
    ])
    
    return format_box, format_selector, include_inverted

def create_action_buttons():
    """Create generate and download buttons"""
    generate_btn = widgets.Button(
        description='Generate Image',
        button_style='primary',
        icon='refresh'
    )
    
    download_btn = widgets.Button(
        description='Download',
        button_style='success',
        icon='download',
        disabled=True
    )
    
    button_box = widgets.HBox([generate_btn, download_btn])
    
    return button_box, generate_btn, download_btn

def create_output_display():
    """Create output display area"""
    output_area = widgets.Output()
    
    display_box = widgets.VBox([
        widgets.HTML("<h3>Preview</h3>"),
        output_area
    ])
    
    return display_box, output_area

def create_download_link(img, filename, format='BMP'):
    """Create a download link for the image"""
    buffer = BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    
    import base64
    b64 = base64.b64encode(buffer.read()).decode()
    
    return HTML(f'''
        <a download="{filename}" href="data:image/{format.lower()};base64,{b64}" target="_blank">
            <button style="padding: 10px 20px; font-size: 14px; background-color: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer;">
                Download {filename}
            </button>
        </a>
    ''')