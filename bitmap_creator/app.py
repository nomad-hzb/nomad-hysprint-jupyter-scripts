import ipywidgets as widgets
from IPython.display import display, HTML
import utils
import gui_components as gui

# Global variables to store generated image
current_image = None
current_params = None

def setup_app():
    """Setup and return the complete application"""
    
    # Create GUI components
    size_box, width_input, height_input = gui.create_size_inputs(default_width=25, default_height=25)
    margin_box, margins = gui.create_margin_inputs(default_margin=3)
    dpi_box, dpi_input = gui.create_dpi_input(default_dpi=350)
    pattern_box, pattern_type, black_percentage = gui.create_pattern_controls(default_percentage=50)
    format_box, format_selector, include_inverted = gui.create_file_format_selector()
    button_box, generate_btn, download_btn = gui.create_action_buttons()
    display_box, output_area = gui.create_output_display()
    
    def generate_image(b):
        """Generate bitmap image based on current parameters"""
        global current_image, current_params
        
        with output_area:
            output_area.clear_output(wait=True)
            
            try:
                # Get parameters
                width_mm = width_input.value
                height_mm = height_input.value
                dpi = dpi_input.value
                
                margin_dict = {
                    'top': margins['top'].value,
                    'bottom': margins['bottom'].value,
                    'left': margins['left'].value,
                    'right': margins['right'].value
                }
                
                black_pct = black_percentage.value
                pattern = pattern_type.value
                
                # Parse pattern type and kwargs
                kwargs = {}
                if pattern.startswith('complementary_'):
                    pattern_id = 0 if pattern.endswith('_a') else 1
                    pattern_base = 'complementary'
                    kwargs['pattern_id'] = pattern_id
                elif pattern.startswith('publication_'):
                    base_size = int(pattern.split('_')[1])
                    pattern_base = 'publication'
                    kwargs['base_size'] = base_size
                else:
                    pattern_base = pattern
                
                # Generate image
                print(f"Generating {width_mm}x{height_mm}mm image at {dpi} DPI...")
                print(f"Using {pattern} pattern for chemical mixing optimization...")
                img = utils.create_bitmap(
                    width_mm, height_mm, dpi, margin_dict, black_pct, pattern_base, **kwargs
                )
                
                current_image = img
                current_params = {
                    'width': width_mm,
                    'height': height_mm,
                    'dpi': dpi,
                    'pattern': pattern,
                    'black_pct': black_pct,
                    'margins': margin_dict
                }
                
                # Display image(s)
                format_str = format_selector.value
                
                # Display original
                display(HTML('<h4>Original:</h4>'))
                img_base64 = utils.image_to_base64(img, format=format_str)
                display(HTML(f'<img src="{img_base64}" style="max-width: 100%; border: 1px solid #ccc;"/>'))
                
                # Display inverted if checkbox is selected
                if include_inverted.value:
                    inverted_img = utils.invert_image(img, width_mm, height_mm, dpi, margin_dict)
                    inverted_base64 = utils.image_to_base64(inverted_img, format=format_str)
                    display(HTML('<h4 style="margin-top: 20px;">Inverted:</h4>'))
                    display(HTML(f'<img src="{inverted_base64}" style="max-width: 100%; border: 1px solid #ccc;"/>'))
                
                # Show info
                width_px = utils.mm_to_pixels(width_mm, dpi)
                height_px = utils.mm_to_pixels(height_mm, dpi)
                print(f"\nImage generated: {width_px}x{height_px} pixels")
                print(f"Size: {width_mm}x{height_mm} mm")
                print(f"Pattern: {pattern}, Black: {black_pct:.1f}%")
                
                # Enable download
                download_btn.disabled = False
                
            except Exception as e:
                print(f"Error: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def download_image(b):
        """Create download link for current image"""
        global current_image, current_params
        
        if current_image is None:
            with output_area:
                print("No image to download. Please generate an image first.")
            return
        
        with output_area:
            output_area.clear_output(wait=True)
            
            try:
                # Generate filename
                format_str = format_selector.value
                params = current_params
                base_filename = f"bitmap_{params['width']:.0f}x{params['height']:.0f}mm_{params['dpi']}dpi_{params['black_pct']:.0f}pct"
                filename = f"{base_filename}.{format_str.lower()}"
                
                # Create download link for original
                download_link = gui.create_download_link(current_image, filename, format_str)
                display(download_link)
                
                # Create inverted version if requested
                if include_inverted.value:
                    params = current_params
                    inverted_img = utils.invert_image(
                        current_image, 
                        params['width'], 
                        params['height'], 
                        params['dpi'], 
                        params['margins']
                    )
                    inverted_filename = f"{base_filename}_inverted.{format_str.lower()}"
                    inverted_link = gui.create_download_link(inverted_img, inverted_filename, format_str)
                    display(HTML("<p style='margin-top: 10px;'><strong>Inverted version:</strong></p>"))
                    display(inverted_link)
                
                # Show the image(s) again
                img_base64 = utils.image_to_base64(current_image, format=format_str)
                display(HTML(f'<h4 style="margin-top: 20px;">Original:</h4>'))
                display(HTML(f'<img src="{img_base64}" style="max-width: 100%; border: 1px solid #ccc;"/>'))
                
                if include_inverted.value:
                    params = current_params
                    inverted_img = utils.invert_image(
                        current_image, 
                        params['width'], 
                        params['height'], 
                        params['dpi'], 
                        params['margins']
                    )
                    inverted_base64 = utils.image_to_base64(inverted_img, format=format_str)
                    display(HTML(f'<h4 style="margin-top: 20px;">Inverted:</h4>'))
                    display(HTML(f'<img src="{inverted_base64}" style="max-width: 100%; border: 1px solid #ccc;"/>'))
                
                print(f"\nClick the button(s) above to download")
                
            except Exception as e:
                print(f"Error creating download: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # Attach event handlers
    generate_btn.on_click(generate_image)
    download_btn.on_click(download_image)
    
    # Create main layout
    header = widgets.HTML(
        "<h1 style='text-align: center; color: #333;'>Bitmap Image Generator</h1>"
        "<p style='text-align: center; color: #666;'>Create black & white bitmap images with customizable patterns for combinatorial inkjet printing</p>"
    )
    
    controls = widgets.VBox([
        size_box,
        margin_box,
        dpi_box,
        pattern_box,
        format_box,
        button_box
    ], layout=widgets.Layout(padding='10px'))
    
    app_layout = widgets.VBox([
        header,
        widgets.HBox([
            controls,
            display_box
        ], layout=widgets.Layout(width='100%')),
    ], layout=widgets.Layout(padding='20px'))
    
    return app_layout