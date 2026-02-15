# gui_layouts.py
"""
GUI layouts and main application class
Assembles widgets into layouts and contains all application logic
"""
import numpy as np
import ipywidgets as widgets
from IPython.display import display
from gui_components import GUIComponents
from data_manager import DataManager
from fitting_engine import FittingEngine
from plot_manager import PlotManager
from exporters import ResultExporter
import config
from utils import debug_print
from data_manager import DataManager, sanitize_float, sanitize_array

import warnings
warnings.filterwarnings('ignore', message='Using UFloat objects with std_dev==0')

# Initialize ipyvuetify FileInput (prevent initialization error)
try:
    from ipyvuetify.extra import FileInput
    from ipyvuetify.extra.file_input import ClientSideFile
    
    # Create a temporary FileInput to initialize the library
    temp_file_input = FileInput()
    temp_file_input.file_info = [{'name': 'test.csv', 'size': 100, 'lastModified': 0, 'type': ''}]
    try:
        c = ClientSideFile(temp_file_input, 0, 1)
        d = c.read()
    except:
        pass
    temp_file_input.file_info = []
    print("✅ ipyvuetify FileInput initialized")
except Exception as e:
    print(f"⚠️ ipyvuetify initialization warning: {e}")


# =============================================================================
# GUI LAYOUTS
# =============================================================================

class GUILayouts:
    """Assembles GUI layouts from components"""
    
    def __init__(self, gui_components):
        """
        Parameters:
        -----------
        gui_components : GUIComponents
            GUI components instance with all widgets
        """
        self.components = gui_components
        self.widgets = gui_components.get_all_widgets()


class GUILayouts:
    """Assembles GUI layouts from components"""
    
    def __init__(self, gui_components):
        """
        Parameters:
        -----------
        gui_components : GUIComponents
            GUI components instance with all widgets
        """
        self.components = gui_components
        self.widgets = gui_components.get_all_widgets()
        
    def create_file_section(self):
        """Create file loading section"""
        if self.components.h5_available:
            section = widgets.VBox([
                widgets.HTML("<h3>📁 Data Loading</h3>"),
                self.widgets['mode_dropdown'],
                widgets.HBox([
                    self.widgets['convert_energy_btn'],
                    self.widgets['energy_unit_display']
                ])
            ])
        else:
            section = widgets.VBox([
                widgets.HTML("<h3>📁 Data Loading</h3>"),
                self.widgets['file_upload'],
                widgets.HBox([
                    self.widgets['convert_energy_btn'],
                    self.widgets['energy_unit_display']
                ])
            ])
        
        debug_print("Created file section layout", "GUI")
        return section
    
    def create_time_control_section(self):
        """Create time control section with wavelength range"""
        section = widgets.VBox([
            widgets.HTML("<h3>⏱️ Time Control</h3>"),
            widgets.HBox([
                self.widgets['time_input'],
                self.widgets['time_display']
            ]),
            self.widgets['time_slider'],
            widgets.HTML("<b>Wavelength Range:</b>"),
            self.widgets['wavelength_range_slider']
        ])
        
        debug_print("Created time control section layout", "GUI")
        return section

    def create_colorbar_section(self):
        """Create colorbar control section (collapsible)"""
        content_widgets = [
            self.widgets['colorbar_range_slider'],
            self.widgets['colorbar_apply_btn']
        ]
        
        section = self.components.create_collapsible_section(
            "🎨 Heatmap Colors",
            content_widgets
        )
        
        # Set to collapsed by default
        section.selected_index = None
        
        debug_print("Created colorbar section layout", "GUI")
        return section
    
    def create_background_section(self):
        """Create unified background handling section"""
        # Create containers for each method
        manual_container = widgets.VBox([
            self.widgets['bg_manual_description'],
            widgets.HBox([
                self.widgets['bg_manual_start'],
                self.widgets['bg_manual_num']
            ])
        ])
        
        linear_container = widgets.VBox([
            widgets.HBox([
                self.widgets['bg_linear_slope'],
                self.widgets['bg_linear_intercept']
            ])
        ])
        
        poly_container = widgets.VBox([
            self.widgets['bg_poly_degree'],
            self.widgets['bg_poly_coeffs']
        ])
        
        exp_container = widgets.VBox([
            widgets.HBox([
                self.widgets['bg_exp_amplitude'],
                self.widgets['bg_exp_decay']
            ]),
            self.widgets['bg_exp_offset']
        ])
        
        custom_container = widgets.VBox([
            self.widgets['bg_custom_description'],
            self.widgets['bg_custom_upload'],
            self.widgets['bg_custom_status']
        ])
        
        # Store containers for later access
        self._bg_containers = {
            'Manual': manual_container,
            'Linear': linear_container,
            'Polynomial': poly_container,
            'Exponential': exp_container,
            'Custom': custom_container
        }
        
        # HIDE ALL CONTAINERS INITIALLY
        for container in self._bg_containers.values():
            container.layout.visibility = 'hidden'
            container.layout.display = 'none'
        
        # Method-specific area that shows only ONE container at a time
        self._method_area = widgets.VBox([
            manual_container,
            linear_container,
            poly_container,
            exp_container,
            custom_container
        ], layout=widgets.Layout(
            min_height='80px',
            max_height='150px'
        ))
        
        content_widgets = [
            self.widgets['background_method'],
            self._method_area,
            widgets.HBox([
                self.widgets['bg_autofit_btn'],
                self.widgets['bg_apply_btn'],
                self.widgets['bg_remove_btn']
            ])
        ]
        
        section = self.components.create_collapsible_section(
            "🎨 Background Handling",
            content_widgets
        )
        
        # Set to collapsed by default
        section.selected_index = None
        
        # Store reference for later access
        self._background_section = section
        
        debug_print("Created background section layout", "GUI")
        return section

    def create_fitting_actions_section(self):
        """Create fitting actions section"""
        section = widgets.VBox([
            widgets.HTML("<h3>🔧 Fitting Actions</h3>"),
            widgets.HBox([
                self.widgets['fit_current_btn'],
                self.widgets['update_params_btn']
            ]),
            self.widgets['r_squared_display']
        ])
        
        debug_print("Created fitting actions section layout", "GUI")
        return section
    
    def create_peak_detection_section(self):
        """Create peak detection and fitting section (collapsible)"""
        # Detection parameters: labels on top, widgets below
        param_labels = widgets.HBox([
            widgets.Label('Height:', layout=widgets.Layout(width='100px')),
            widgets.Label('Prominence:', layout=widgets.Layout(width='100px')),
            widgets.Label('Distance:', layout=widgets.Layout(width='100px'))
        ])
        
        param_widgets = widgets.HBox([
            self.widgets['peak_height_threshold'],
            self.widgets['peak_prominence'],
            self.widgets['peak_distance']
        ])
        
        content_widgets = [
            widgets.HTML("<b>Auto-Detection:</b>"),
            self.widgets['auto_detect_btn'],
            widgets.HTML("<b>Detection Parameters:</b>"),
            param_labels,
            param_widgets,
            widgets.HTML("<hr>"),
            widgets.HTML("<b>Peak Models:</b>"),
            self.widgets['add_peak_btn'],
            self.widgets['peak_list_container']
        ]
        
        section = self.components.create_collapsible_section(
            "🔍 Peak Detection & Models",
            content_widgets
        )
        
        debug_print("Created peak detection section layout", "GUI")
        return section
    
    def create_batch_fitting_section(self):
        """Create batch fitting section with unit conversion at bottom"""
        section = widgets.VBox([
            widgets.HTML("<h3>📊 Batch Fitting</h3>"),
            widgets.HTML("<b>Range Selection:</b>"),
            widgets.HBox([
                self.widgets['fit_start_idx'],
                self.widgets['fit_end_idx']
            ]),
            widgets.HTML("<b>Batch Actions:</b>"),
            widgets.HBox([
                self.widgets['fit_all_btn'],
                self.widgets['fit_all_range_btn']
            ]),
            self.widgets['fit_progress'],  # Add progress bar
            self.widgets['export_btn'],
            self.widgets['export_output'],
            widgets.HTML("<hr>"),
            widgets.HTML("<h4>📏 Unit Conversion (Test)</h4>"),
            widgets.HBox([
                self.widgets['toggle_angstrom_btn'],
                self.widgets['unit_display']
            ])
        ])
        
        debug_print("Created batch fitting section layout", "GUI")
        return section
    
    def create_fitting_control_section(self):
        """Create complete fitting control section"""
        section = widgets.VBox([
            self.create_background_section(),
            self.create_peak_detection_section(),
            self.create_fitting_actions_section(),
            self.create_batch_fitting_section()
        ], layout=widgets.Layout(width=config.CONTROL_PANEL_WIDTH))
        
        debug_print("Created fitting control section layout", "GUI")
        return section
    
    def create_visualization_section(self):
        """Create visualization section"""
        section = widgets.VBox([
            widgets.HTML("<h3>📊 Visualizations</h3>"),
            widgets.HTML("<h4>Heatmap</h4>"),
            self.widgets['heatmap_output'],
            widgets.HTML("<h4>Current Spectrum</h4>"),
            self.widgets['spectrum_output'],
            widgets.HTML("<h4>Time Series Analysis</h4>"),
            self.widgets['time_series_output']
        ], layout=widgets.Layout(flex='2', width='auto'))
        
        debug_print("Created visualization section layout", "GUI")
        return section
    
    def create_control_panel(self):
        """Create complete control panel (left side)"""
        control_panel = widgets.VBox([
            widgets.HTML("<h3>📋 Status</h3>"),
            self.widgets['status_output'],
            widgets.HTML("<hr>"),
            self.create_file_section(),
            self.create_time_control_section(),
            self.create_colorbar_section(),  
            self.create_fitting_control_section()
        ], layout=widgets.Layout(
            width=config.CONTROL_PANEL_WIDTH,
            padding='10px',
            border='1px solid #ddd',
            margin='0px 10px 0px 0px'
        ))
        
        debug_print("Created control panel layout", "GUI")
        return control_panel
    
    def create_main_layout(self):
        """Create complete main layout"""
        # Header
        header = widgets.HTML("<h1>🔬 Peak Analysis App</h1>")
        
        # Main content (control panel + visualization)
        main_content = widgets.HBox([
            self.create_control_panel(),
            self.create_visualization_section()
        ], layout=widgets.Layout(
            width='100%',
            height='auto'
        ))
        
        debug_print("Created main layout", "GUI")
        return header, main_content

# =============================================================================
# MAIN APPLICATION CLASS
# =============================================================================

class PLAnalysisApp:
    """Main Photoluminescence Analysis Application"""
    
    def __init__(self):
        """Initialize the application"""
        debug_print("Initializing PLAnalysisApp", "APP")
        
        # Initialize managers
        self.data_manager = DataManager()
        self.fitting_engine = FittingEngine()
        self.plot_manager = PlotManager()
        self.plot_manager.app_ref = self
        debug_print("PlotManager app_ref set", "APP")
        self.export_utils = ResultExporter()
        
        # Initialize GUI components
        self.gui_components = GUIComponents(h5_available=self.data_manager.is_h5_available())
        self.gui_components.create_all_widgets()

        self.gui_layouts = GUILayouts(self.gui_components)
        
        # Get widget references for easy access
        self.widgets = self.gui_components.get_all_widgets()

        self.wavelength_unit = 'nm'  # Track current wavelength unit ('nm' or 'angstrom')
        
        # State management
        self.peak_models = []  # List of peak model widgets
        self.last_fit_result = None
        self.original_data_matrix = None  # For background subtraction
        self.background_applied = False
        self.background_accordion = None  # Reference to background accordion
        
        # Setup UI and connect callbacks
        self.setup_callbacks()
        
        # If H5 data is available, trigger initial load
        if self.data_manager.is_h5_available():
            debug_print("H5 data available, triggering initial load", "APP")
            self.on_file_upload({'new': [None]})
    
    def setup_callbacks(self):
        """Connect all UI callbacks"""
        debug_print("Setting up UI callbacks", "APP")
        
        # File upload callbacks
        if self.data_manager.is_h5_available():
            self.widgets['mode_dropdown'].observe(self.on_file_upload, names='value')
        else:
            #self.widgets['file_upload'].observe(self.on_file_upload, names='value')
            self.widgets['file_upload'].observe(self.on_file_upload, names='file_info')
        
        # Time control callbacks
        self.widgets['time_slider'].observe(self.on_time_change_slider, names='value')
        self.widgets['time_input'].observe(self.on_time_change_input, names='value')
        
        # Peak detection and model callbacks
        self.widgets['auto_detect_btn'].on_click(self.on_auto_detect_peaks)
        self.widgets['add_peak_btn'].on_click(self.on_add_peak)
        
        # Fitting action callbacks
        self.widgets['fit_current_btn'].on_click(self.on_fit_current)
        self.widgets['update_params_btn'].on_click(self.on_update_parameters)
        
        # Batch fitting callbacks
        self.widgets['fit_all_btn'].on_click(self.on_fit_all)
        self.widgets['fit_all_range_btn'].on_click(self.on_fit_range)
        
        # Export callback
        self.widgets['export_btn'].on_click(self.on_export_results)

        # Colorbar control callbacks
        self.widgets['colorbar_apply_btn'].on_click(self.on_colorbar_apply)
        
        # Auto-apply when slider value changes
        def on_colorbar_slider_change(change):
            """Auto-apply colorbar when slider changes"""
            if self.data_manager.is_data_loaded():
                z_min, z_max = change['new']
                self.update_heatmap_colorbar(z_min, z_max)
                
                with self.widgets['status_output']:
                    self.widgets['status_output'].clear_output()
                    print(f"✅ Colorbar range: {z_min:.1f} - {z_max:.1f}")
        
        self.widgets['colorbar_range_slider'].observe(on_colorbar_slider_change, names='value')
        
        # Background handling callbacks
        self.widgets['background_method'].observe(self.on_background_method_change, names='value')
        self.widgets['bg_autofit_btn'].on_click(self.on_background_autofit)
        self.widgets['bg_apply_btn'].on_click(self.on_background_apply)
        self.widgets['bg_remove_btn'].on_click(self.on_background_remove)
        self.widgets['bg_custom_upload'].observe(self.on_custom_background_upload, names='value')
        
        # Trigger initial background method change to show/hide correct widgets
        self.on_background_method_change({'new': config.DEFAULT_BACKGROUND_METHOD})
        
        # Add initial peak model
        self.add_peak_model()

        # Energy conversion callback
        self.widgets['convert_energy_btn'].on_click(self.on_convert_energy)

        def on_wavelength_range_change(change):
            """Update wavelength range visualization when slider changes"""
            try:
                debug_print(f"===== WAVELENGTH RANGE CALLBACK TRIGGERED: {change['new']} =====", "DEBUG")
                
                if not self.data_manager.is_data_loaded():
                    debug_print("Data not loaded, returning", "DEBUG")
                    return
                
                debug_print("Data is loaded, proceeding...", "DEBUG")
                
                wavelength_range = change['new']
                
                # Check if range is at borders (don't show lines)
                wl_min, wl_max = self.data_manager.wavelengths.min(), self.data_manager.wavelengths.max()
                debug_print(f"wl_min={wl_min}, wl_max={wl_max}, range={wavelength_range}", "DEBUG")
                
                if wavelength_range[0] == wl_min and wavelength_range[1] == wl_max:
                    wavelength_range = None
                    debug_print("Range is at borders, setting to None", "DEBUG")
                
                # Update spectrum plot
                debug_print("About to call _update_wavelength_range_on_spectrum", "DEBUG")
                self._update_wavelength_range_on_spectrum(wavelength_range)
                debug_print("Finished _update_wavelength_range_on_spectrum", "DEBUG")
                
                # Update heatmap
                debug_print("About to call _update_wavelength_range_on_heatmap", "DEBUG")
                self._update_wavelength_range_on_heatmap(wavelength_range)
                debug_print("Finished _update_wavelength_range_on_heatmap", "DEBUG")
                
            except Exception as e:
                debug_print(f"ERROR in wavelength range callback: {e}", "ERROR")
                import traceback
                traceback.print_exc()
        
        self.widgets['wavelength_range_slider'].observe(on_wavelength_range_change, names='value')
        debug_print(f"Wavelength range callback connected to {self.widgets['wavelength_range_slider']}", "APP")
        
        # Setup unit conversion callbacks
        self.setup_unit_conversion_callbacks()
        
        debug_print("Callbacks setup complete", "APP")

    def setup_unit_conversion_callbacks(self):
        """Setup callbacks for unit conversion"""
        self.widgets['toggle_angstrom_btn'].on_click(self.on_toggle_angstrom)
        debug_print("Connected unit conversion callbacks", "APP")

    def _update_wavelength_range_on_spectrum(self, wavelength_range):
        """Update only the wavelength range lines on spectrum plot"""
        debug_print(f"Updating spectrum range to {wavelength_range}", "SPECTRUM")
        debug_print(f"self.plot_manager = {self.plot_manager}", "SPECTRUM")
        debug_print(f"self.plot_manager.spectrum_fig = {self.plot_manager.spectrum_fig}", "SPECTRUM")
        debug_print(f"spectrum_fig is None? {self.plot_manager.spectrum_fig is None}", "SPECTRUM")
        
        if self.plot_manager.spectrum_fig is None:
            debug_print("No spectrum fig, returning early", "SPECTRUM")
            return
        
        fig = self.plot_manager.spectrum_fig
        debug_print(f"Number of shapes before: {len(fig.layout.shapes)}", "SPECTRUM")
        
        # Debug: print all existing shapes
        for i, shape in enumerate(fig.layout.shapes):
            debug_print(f"Shape {i}: type={shape.type}, color={getattr(shape.line, 'color', 'N/A')}, dash={getattr(shape.line, 'dash', 'N/A')}, x0={shape.x0}, x1={shape.x1}", "SPECTRUM")
        
        # Remove existing wavelength range shapes (vertical green dashed lines)
        new_shapes = []
        for shape in fig.layout.shapes:
            # Remove if it's a vertical green dashed line
            is_green = getattr(shape.line, 'color', None) == "green"
            is_dashed = getattr(shape.line, 'dash', None) == "dash"
            is_vertical = (shape.type == "line" and shape.x0 == shape.x1)
            
            should_remove = is_green and is_dashed and is_vertical
            debug_print(f"Shape: green={is_green}, dashed={is_dashed}, vertical={is_vertical}, remove={should_remove}", "SPECTRUM")
            
            if should_remove:
                continue  # Skip this shape (remove it)
            new_shapes.append(shape)
        
        fig.layout.shapes = new_shapes
        debug_print(f"Number of shapes after removal: {len(fig.layout.shapes)}", "SPECTRUM")
        
        # Add new shapes if range is specified
        if wavelength_range is not None:
            # Get y-axis range from figure
            if fig.data and len(fig.data) > 0:
                y_data = fig.data[0].y
                y_min, y_max = min(y_data), max(y_data)
            else:
                y_min, y_max = 0, 1
            
            debug_print(f"Adding lines at x={wavelength_range[0]} and x={wavelength_range[1]}", "SPECTRUM")
            
            # Add left line
            fig.add_shape(
                type="line",
                x0=wavelength_range[0],
                x1=wavelength_range[0],
                y0=y_min,
                y1=y_max,
                line=dict(color="green", width=2, dash="dash")
            )
            
            # Add right line
            fig.add_shape(
                type="line",
                x0=wavelength_range[1],
                x1=wavelength_range[1],
                y0=y_min,
                y1=y_max,
                line=dict(color="green", width=2, dash="dash")
            )
            
            debug_print(f"Number of shapes after adding: {len(fig.layout.shapes)}", "SPECTRUM")

    def _update_wavelength_range_on_heatmap(self, wavelength_range):
        """Update only the wavelength range lines on heatmap"""
        if self.plot_manager.heatmap_fig is None:
            return
        
        fig = self.plot_manager.heatmap_fig
        
        # Remove existing wavelength range shapes (keep time position line)
        new_shapes = []
        for shape in fig.layout.shapes:
            # Keep vertical lines (time position), remove horizontal lines (wavelength range)
            if shape.type == "line" and shape.x0 == shape.x1:  # Vertical line
                new_shapes.append(shape)
        fig.layout.shapes = new_shapes
        
        # Add new shapes if range is specified
        if wavelength_range is not None:
            # Get x-axis range from figure
            if fig.data and len(fig.data) > 0:
                x_data = fig.data[0].x
                x_min, x_max = min(x_data), max(x_data)
            else:
                x_min, x_max = 0, 1
            
            # Add bottom line
            fig.add_shape(
                type="line",
                x0=x_min,
                x1=x_max,
                y0=wavelength_range[0],
                y1=wavelength_range[0],
                line=dict(color="green", width=2, dash="dash")
            )
            
            # Add top line
            fig.add_shape(
                type="line",
                x0=x_min,
                x1=x_max,
                y0=wavelength_range[1],
                y1=wavelength_range[1],
                line=dict(color="green", width=2, dash="dash")
            )

    def _store_accordion_reference(self):
        """Store reference to background accordion for programmatic control"""
        try:
            if hasattr(self.gui_layouts, '_background_section'):
                self.background_accordion = self.gui_layouts._background_section
                debug_print("Stored background accordion reference", "APP")
        except Exception as e:
            debug_print(f"Could not store accordion reference: {e}", "APP")
    
    # =========================================================================
    # FILE LOADING
    # =========================================================================
    
    def on_file_upload(self, change):
        """Handle file upload - works with both ipyvuetify and ipywidgets"""
        debug_print("!!! CALLBACK TRIGGERED !!!", "APP")
        import sys
        sys.stdout.flush()
        
        try:
            debug_print("="*50, "APP")
            debug_print("File upload triggered", "APP")
            
            # Handle H5 mode
            if self.data_manager.is_h5_available():
                mode = self.widgets['mode_dropdown'].value
                with self.widgets['status_output']:
                    self.widgets['status_output'].clear_output()
                    print("Loading data from H5...")
                
                self.data_manager.load_from_h5(mode)
                self.update_ui_after_data_load()
                self.update_visualizations(update_heatmap=True)
                
                data_info = self.data_manager.get_data_info()
                with self.widgets['status_output']:
                    self.widgets['status_output'].clear_output()
                    print(f"✅ Successfully loaded data!")
                    print(f"📊 {data_info['time_points']} time points, {data_info['wavelengths']} wavelengths")
                    print(f"⏱️ Time range: {data_info['time_range'][0]:.3f} - {data_info['time_range'][1]:.3f} s")
                    print(f"🌈 Wavelength range: {data_info['wavelength_range'][0]:.1f} - {data_info['wavelength_range'][1]:.1f} nm")
                return
            
            # Handle ipyvuetify FileInput
            debug_print(f"Change type: {type(change)}", "APP")
            debug_print(f"Change keys: {change.keys() if isinstance(change, dict) else 'not a dict'}", "APP")
            
            # Get files from ipyvuetify FileInput
            file_data_list = self.widgets['file_upload'].get_files()
            
            debug_print(f"File data list type: {type(file_data_list)}", "APP")
            debug_print(f"File data list length: {len(file_data_list) if file_data_list else 0}", "APP")
            
            if not file_data_list or len(file_data_list) == 0:
                debug_print("No files in upload", "APP")
                return
            
            # Get first file
            file_data = file_data_list[0]
            
            debug_print(f"File data type: {type(file_data)}", "APP")
            debug_print(f"File data keys: {file_data.keys() if isinstance(file_data, dict) else 'not a dict'}", "APP")
            
            # Extract filename and content
            if isinstance(file_data, dict):
                filename = file_data.get('name', 'unknown')
                debug_print(f"Filename: {filename}", "APP")
                
                # Read file content
                if 'file_obj' in file_data:
                    file_content = file_data['file_obj'].read()
                    debug_print(f"File content type: {type(file_content)}", "APP")
                    debug_print(f"File content length: {len(file_content)}", "APP")
                    
                    # Show first 100 bytes for debugging
                    if isinstance(file_content, (bytes, memoryview)):
                        debug_print(f"First 100 bytes: {bytes(file_content[:100])}", "APP")
                else:
                    debug_print("No 'file_obj' in file_data", "APP")
                    with self.widgets['status_output']:
                        self.widgets['status_output'].clear_output()
                        print("❌ Error: Could not read file content")
                    return
            else:
                debug_print(f"Unexpected file_data structure: {file_data}", "APP")
                return
            
            # Load the data
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"Loading {filename}...")
            
            debug_print("About to call data_manager.load_from_file", "APP")
            self.data_manager.load_from_file(file_content)
            debug_print("Data loaded successfully", "APP")
            
            # Update UI controls with loaded data
            debug_print("Updating UI after data load", "APP")
            self.update_ui_after_data_load()
            debug_print("UI updated after data load", "APP")
            
            # Initial visualization
            debug_print("Updating visualizations", "APP")
            self.update_visualizations(update_heatmap=True)
            debug_print("Visualizations updated", "APP")
    
            # Initialize wavelength range visualization if not at borders
            wavelength_range = self.widgets['wavelength_range_slider'].value
            wl_min, wl_max = self.data_manager.wavelengths.min(), self.data_manager.wavelengths.max()
            if wavelength_range[0] != wl_min or wavelength_range[1] != wl_max:
                self._update_wavelength_range_on_heatmap(wavelength_range)
            
            # Display success message
            data_info = self.data_manager.get_data_info()
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"✅ Successfully loaded {filename}!")
                print(f"📊 {data_info['time_points']} time points, {data_info['wavelengths']} wavelengths")
                print(f"⏱️ Time range: {data_info['time_range'][0]:.3f} - {data_info['time_range'][1]:.3f} s")
                print(f"🌈 Wavelength range: {data_info['wavelength_range'][0]:.1f} - {data_info['wavelength_range'][1]:.1f} nm")
    
            # Enable energy conversion button for PL data
            if self.data_manager.data_source == 'csv' or (self.data_manager.h5_mode in ['pl_raw', 'pl_binned']):
                self.widgets['convert_energy_btn'].disabled = False
            else:
                self.widgets['convert_energy_btn'].disabled = True
                    
        except Exception as e:
            import traceback
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"❌ Error loading file: {str(e)}")
                print("\n🔍 Full traceback:")
                print(traceback.format_exc())
            debug_print(f"ERROR: {e}", "APP")
            debug_print(f"Full traceback:\n{traceback.format_exc()}", "APP")
    
    def update_ui_after_data_load(self):
        """Update UI controls after data is loaded"""
        debug_print("Updating UI after data load", "update_ui_after_data_load")
        
        # Get time range
        min_idx, max_idx = self.data_manager.get_time_range()
        debug_print(f"Time range indices: min={min_idx}, max={max_idx}", "update_ui_after_data_load")
        
        # Update time controls
        self.widgets['time_slider'].max = max_idx
        self.widgets['time_slider'].value = 0
        self.widgets['time_slider'].disabled = False
        
        self.widgets['time_input'].max = max_idx
        self.widgets['time_input'].value = 0
        self.widgets['time_input'].disabled = False
        
        # Update fit range controls
        self.widgets['fit_start_idx'].max = max_idx
        self.widgets['fit_start_idx'].value = 0
        self.widgets['fit_start_idx'].disabled = False
        
        self.widgets['fit_end_idx'].max = max_idx
        self.widgets['fit_end_idx'].value = max_idx
        self.widgets['fit_end_idx'].disabled = False

        # Enable background handling
        self.widgets['bg_manual_start'].max = max_idx
        self.widgets['bg_manual_num'].max = min(50, max_idx + 1)
        self.widgets['bg_apply_btn'].disabled = False

        # Enable background handling
        self.widgets['bg_manual_start'].max = max_idx
        self.widgets['bg_manual_num'].max = min(50, max_idx + 1)
        self.widgets['bg_apply_btn'].disabled = False
        
        # Enable detection parameters
        self.widgets['peak_height_threshold'].disabled = False
        self.widgets['peak_prominence'].disabled = False
        self.widgets['peak_distance'].disabled = False

        # Get wavelength range
        wavelength_min = float(self.data_manager.wavelengths.min())
        wavelength_max = float(self.data_manager.wavelengths.max())
        debug_print(f"Wavelength range: min={wavelength_min}, max={wavelength_max}", "update_ui_after_data_load")

        # Enable wavelength range controls
        self.widgets['wavelength_range_slider'].max = 1e12
        self.widgets['wavelength_range_slider'].min = wavelength_min
        self.widgets['wavelength_range_slider'].max = wavelength_max
        self.widgets['wavelength_range_slider'].value = [wavelength_min, wavelength_max]
        self.widgets['wavelength_range_slider'].disabled = False

        # Enable colorbar controls
        data_min = sanitize_float(self.data_manager.data_matrix.min(), default=0.0)
        data_max = sanitize_float(self.data_manager.data_matrix.max(), default=1000.0)
        
        # Ensure min < max
        if data_min >= data_max:
            data_max = data_min + 1000.0
            
        # Set colorbar range
        # Set max first if increasing range
        if data_max > self.widgets['colorbar_range_slider'].max:
            self.widgets['colorbar_range_slider'].max = data_max
            self.widgets['colorbar_range_slider'].min = data_min
        else:
            self.widgets['colorbar_range_slider'].min = data_min
            self.widgets['colorbar_range_slider'].max = data_max
            
        self.widgets['colorbar_range_slider'].value = (data_min, data_max)
        self.widgets['colorbar_range_slider'].disabled = False
        self.widgets['colorbar_apply_btn'].disabled = False

        # Set min and max carefully to avoid min > max error
        if data_min < data_max:
            # Set max first if increasing range
            if data_max > self.widgets['colorbar_range_slider'].max:
                self.widgets['colorbar_range_slider'].max = data_max
                self.widgets['colorbar_range_slider'].min = data_min
            else:
                self.widgets['colorbar_range_slider'].min = data_min
                self.widgets['colorbar_range_slider'].max = data_max
            self.widgets['colorbar_range_slider'].value = (data_min, data_max)
        else:
            # Handle edge case where min == max
            self.widgets['colorbar_range_slider'].min = data_min - 1
            self.widgets['colorbar_range_slider'].max = data_max + 1
            self.widgets['colorbar_range_slider'].value = (data_min, data_max)
        self.widgets['colorbar_range_slider'].disabled = False
        self.widgets['colorbar_apply_btn'].disabled = False
        
        debug_print("UI updated after data load", "update_ui_after_data_load")

    def _update_wavelength_range_on_spectrum(self, wavelength_range):
        """Update only the wavelength range lines on spectrum plot"""
        try:
            debug_print(f"Updating spectrum range to {wavelength_range}", "SPECTRUM")
            debug_print(f"self.plot_manager = {self.plot_manager}", "SPECTRUM")
            debug_print(f"self.plot_manager.spectrum_fig = {self.plot_manager.spectrum_fig}", "SPECTRUM")
            debug_print(f"spectrum_fig is None? {self.plot_manager.spectrum_fig is None}", "SPECTRUM")
        except Exception as e:
            print(f"!!!! EXCEPTION in debug prints: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Early return if no range specified or no figure exists
        if wavelength_range is None or self.plot_manager.spectrum_fig is None:
            debug_print("No range or no spectrum fig, returning early", "SPECTRUM")
            return
        
        fig = self.plot_manager.spectrum_fig
        debug_print(f"Number of shapes before: {len(fig.layout.shapes)}", "SPECTRUM")
        
        # Remove existing wavelength range shapes (vertical green dashed lines)
        new_shapes = []
        for shape in fig.layout.shapes:
            # Remove if it's a vertical green dashed line
            is_green = getattr(shape.line, 'color', None) == "green"
            is_dashed = getattr(shape.line, 'dash', None) == "dash"
            is_vertical = (shape.type == "line" and shape.x0 == shape.x1)
            
            should_remove = is_green and is_dashed and is_vertical
            debug_print(f"Shape: x0={shape.x0}, green={is_green}, dashed={is_dashed}, vertical={is_vertical}, remove={should_remove}", "SPECTRUM")
            
            if not should_remove:
                new_shapes.append(shape)
        
        fig.layout.shapes = new_shapes
        debug_print(f"Number of shapes after removal: {len(fig.layout.shapes)}", "SPECTRUM")
        
        # Add new shapes if range is specified
        if wavelength_range is not None:
            # Get y-axis range from figure
            if fig.data and len(fig.data) > 0:
                y_data = fig.data[0].y
                y_min, y_max = min(y_data), max(y_data)
            else:
                y_min, y_max = 0, 1
            
            # Add left line
            left_shape = fig.add_shape(
                type="line",
                x0=wavelength_range[0],
                x1=wavelength_range[0],
                y0=y_min,
                y1=y_max,
                line=dict(color="green", width=2, dash="dash")
            )
            left_shape._is_wl_range = True
            
            # Add right line
            right_shape = fig.add_shape(
                type="line",
                x0=wavelength_range[1],
                x1=wavelength_range[1],
                y0=y_min,
                y1=y_max,
                line=dict(color="green", width=2, dash="dash")
            )
            right_shape._is_wl_range = True
    
    # =========================================================================
    # TIME CONTROL
    # =========================================================================
    
    def on_time_change_slider(self, change):
        """Handle time slider change"""
        new_idx = change['new']
        debug_print(f"Time slider changed to {new_idx}", "APP")
        
        # Update input to match (without triggering its callback)
        self.widgets['time_input'].unobserve(self.on_time_change_input, names='value')
        self.widgets['time_input'].value = new_idx
        self.widgets['time_input'].observe(self.on_time_change_input, names='value')
        
        # Update data manager and visualizations
        self.data_manager.set_current_time(new_idx)
        self.update_visualizations()
    
    def on_time_change_input(self, change):
        """Handle time input change"""
        new_idx = change['new']
        debug_print(f"Time input changed to {new_idx}", "APP")
        
        # Update slider to match (without triggering its callback)
        self.widgets['time_slider'].unobserve(self.on_time_change_slider, names='value')
        self.widgets['time_slider'].value = new_idx
        self.widgets['time_slider'].observe(self.on_time_change_slider, names='value')
        
        # Update data manager and visualizations
        self.data_manager.set_current_time(new_idx)
        self.update_visualizations()
    
    # =========================================================================
    # PEAK DETECTION AND MODELS
    # =========================================================================
    
    def on_auto_detect_peaks(self, button):
        """Auto-detect peaks in current spectrum"""
        if not self.data_manager.is_data_loaded():
            return
        
        try:
            debug_print("Auto-detecting peaks", "APP")
            
            current_spectrum = self.data_manager.get_current_spectrum()
            wavelengths = self.data_manager.wavelengths
            
            # Get wavelength range from slider
            wl_range = self.widgets['wavelength_range_slider'].value
            wl_min = float(wavelengths.min())
            wl_max = float(wavelengths.max())
            is_limited = (wl_range[0] > wl_min) or (wl_range[1] < wl_max)
            
            # If range is limited, only detect peaks in that range
            if is_limited:
                mask = (wavelengths >= wl_range[0]) & (wavelengths <= wl_range[1])
                wavelengths_detect = wavelengths[mask]
                spectrum_detect = current_spectrum[mask]
                debug_print(f"Detecting peaks in range: {wl_range[0]:.1f} - {wl_range[1]:.1f} {self.wavelength_unit}", "APP")
            else:
                wavelengths_detect = wavelengths
                spectrum_detect = current_spectrum
                debug_print("Detecting peaks in full range", "APP")
            
            # Get detection parameters from UI
            min_height = self.widgets['peak_height_threshold'].value
            if min_height == 0:
                min_height = np.max(spectrum_detect) * 0.05
            
            min_prominence = self.widgets['peak_prominence'].value
            if min_prominence == 0:
                min_prominence = np.std(spectrum_detect) * 2
            
            min_distance = self.widgets['peak_distance'].value
            
            # Detect peaks
            peaks_info = self.fitting_engine.detect_peaks(
                wavelengths_detect,
                spectrum_detect,
                height=min_height,
                prominence=min_prominence,
                distance=min_distance
            )
            
            # Clear existing peak models
            self.clear_peak_models()
            
            # Add models for detected peaks (round to 3 decimals for display)
            for peak in peaks_info:
                rounded_peak = {
                    'center': round(peak['center'], 3),
                    'height': round(peak['height'], 3),
                    'sigma': round(peak['sigma'], 3),
                    'type': peak.get('type', config.DEFAULT_PEAK_MODEL)
                }
                self.add_peak_model(peak_info=rounded_peak)
            
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"🔍 Detected {len(peaks_info)} peaks")
                if is_limited:
                    print(f"   Range: {wl_range[0]:.1f} - {wl_range[1]:.1f} {self.wavelength_unit}")
                for i, peak in enumerate(peaks_info):
                    print(f"  Peak {i+1}: {peak['center']:.3f} {self.wavelength_unit}, height: {peak['height']:.1f}")
            
            debug_print(f"Detected {len(peaks_info)} peaks", "APP")
                    
        except Exception as e:
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"❌ Error detecting peaks: {str(e)}")
            debug_print(f"Error in peak detection: {e}", "APP")
    
    def on_add_peak(self, button):
        """Add a new peak model"""
        self.add_peak_model()
    
    def add_peak_model(self, peak_info=None):
        """
        Add a new peak model to the interface
        
        Parameters:
        -----------
        peak_info : dict, optional
            Initial peak information from detection
        """
        if peak_info is None:
            peak_info = {
                'center': 700.0,
                'height': 1000.0,
                'sigma': 10.0,
                'type': config.DEFAULT_PEAK_MODEL
            }
        
        peak_idx = len(self.peak_models)
        debug_print(f"Adding peak model {peak_idx}", "APP")
        
        # Create peak row widget
        peak_row = self.gui_components.create_peak_row_widget(peak_idx, peak_info)
        
        # Setup remove button callback
        peak_row._widgets['remove'].on_click(lambda b: self.remove_peak_model(peak_idx))
        
        # Store peak model
        self.peak_models.append(peak_row)
        
        # Update peak list container
        self.update_peak_list_container()

    def on_toggle_angstrom(self, button):
        """Toggle wavelength unit between nm and Angstrom with data conversion"""
        if not self.data_manager.is_data_loaded():
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print("❌ No data loaded. Please load data first.")
            return
        
        try:
            debug_print(f"Converting from {self.wavelength_unit}", "APP")
            
            if self.wavelength_unit == 'nm':
                # Convert to Angstrom
                success = self.data_manager.convert_wavelengths_to_angstrom()
                if success:
                    self.wavelength_unit = 'angstrom'
                    self.widgets['unit_display'].value = "Current unit: Å"
                    
                    # Update wavelength range slider
                    wl_min = float(self.data_manager.wavelengths.min())
                    wl_max = float(self.data_manager.wavelengths.max())
                    current_range = self.widgets['wavelength_range_slider'].value
                    
                    # Convert current range to Angstrom
                    new_range = (current_range[0] * 10, current_range[1] * 10)
                    
                    # Update in correct order to avoid min > max errors
                    self.widgets['wavelength_range_slider'].min = min(wl_min, self.widgets['wavelength_range_slider'].min)
                    self.widgets['wavelength_range_slider'].max = max(wl_max, self.widgets['wavelength_range_slider'].max)
                    self.widgets['wavelength_range_slider'].value = new_range
                    self.widgets['wavelength_range_slider'].min = wl_min
                    self.widgets['wavelength_range_slider'].max = wl_max

                    # change label
                    self.widgets['wavelength_range_slider'].description = 'λ Range (Å):'
                    
                    with self.widgets['status_output']:
                        self.widgets['status_output'].clear_output()
                        print(f"✅ Converted to Angstrom (Å)")
                        print(f"   Range: {wl_min:.1f} - {wl_max:.1f} Å")
                        
            elif self.wavelength_unit == 'angstrom':
                # Convert to nm
                success = self.data_manager.convert_wavelengths_to_nm()
                if success:
                    self.wavelength_unit = 'nm'
                    self.widgets['unit_display'].value = "Current unit: nm"
                    
                    # Update wavelength range slider
                    wl_min = float(self.data_manager.wavelengths.min())
                    wl_max = float(self.data_manager.wavelengths.max())
                    current_range = self.widgets['wavelength_range_slider'].value
                    
                    # Convert current range to nm
                    new_range = (current_range[0] / 10, current_range[1] / 10)
                    
                    # Update in correct order to avoid min > max errors
                    self.widgets['wavelength_range_slider'].min = min(wl_min, self.widgets['wavelength_range_slider'].min)
                    self.widgets['wavelength_range_slider'].max = max(wl_max, self.widgets['wavelength_range_slider'].max)
                    self.widgets['wavelength_range_slider'].value = new_range
                    self.widgets['wavelength_range_slider'].min = wl_min
                    self.widgets['wavelength_range_slider'].max = wl_max

                    # change label
                    self.widgets['wavelength_range_slider'].description = 'λ Range (nm):'
                    
                    with self.widgets['status_output']:
                        self.widgets['status_output'].clear_output()
                        print(f"✅ Converted to nanometers (nm)")
                        print(f"   Range: {wl_min:.1f} - {wl_max:.1f} nm")
            else:
                # If currently in eV, go back to nm
                self.wavelength_unit = 'nm'
                self.widgets['unit_display'].value = "Current unit: nm"
                with self.widgets['status_output']:
                    self.widgets['status_output'].clear_output()
                    print(f"✅ Changed to nanometers (nm)")
            
            # Update all visualizations with new data
            self.update_visualizations(update_heatmap=True)
            
            debug_print(f"Conversion complete. New unit: {self.wavelength_unit}", "APP")
            
        except Exception as e:
            import traceback
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"❌ Error converting units: {str(e)}")
                print("\n🔍 Full traceback:")
                print(traceback.format_exc())
            debug_print(f"Error in unit conversion: {e}", "APP")

    def on_convert_energy(self, button):
        """Convert between wavelength (nm) and energy (eV)"""
        if not self.data_manager.is_data_loaded():
            return
        
        try:
            debug_print(f"Converting from {self.wavelength_unit}", "APP")
            
            if self.wavelength_unit == 'nm':
                # Convert to eV
                success = self.data_manager.convert_wavelength_to_energy()
                if success:
                    self.wavelength_unit = 'eV'
                    self.widgets['energy_unit_display'].value = "E (eV)"
                    
                    # Update wavelength range slider
                    wl_min = float(self.data_manager.wavelengths.min())
                    wl_max = float(self.data_manager.wavelengths.max())
                    
                    # Expand range first, then set new limits
                    self.widgets['wavelength_range_slider'].min = min(wl_min, self.widgets['wavelength_range_slider'].min)
                    self.widgets['wavelength_range_slider'].max = max(wl_max, self.widgets['wavelength_range_slider'].max)
                    self.widgets['wavelength_range_slider'].value = (wl_min, wl_max)
                    self.widgets['wavelength_range_slider'].min = wl_min
                    self.widgets['wavelength_range_slider'].max = wl_max
                    
                    with self.widgets['status_output']:
                        self.widgets['status_output'].clear_output()
                        print(f"✅ Converted to energy (eV)")
                        print(f"   Range: {wl_min:.2f} - {wl_max:.2f} eV")
            else:
                # Convert to nm
                success = self.data_manager.convert_energy_to_wavelength()
                if success:
                    self.wavelength_unit = 'nm'
                    self.widgets['energy_unit_display'].value = "λ (nm)"
                    
                    # Update wavelength range slider
                    wl_min = float(self.data_manager.wavelengths.min())
                    wl_max = float(self.data_manager.wavelengths.max())
                    
                    # Expand range first, then set new limits
                    self.widgets['wavelength_range_slider'].min = min(wl_min, self.widgets['wavelength_range_slider'].min)
                    self.widgets['wavelength_range_slider'].max = max(wl_max, self.widgets['wavelength_range_slider'].max)
                    self.widgets['wavelength_range_slider'].value = (wl_min, wl_max)
                    self.widgets['wavelength_range_slider'].min = wl_min
                    self.widgets['wavelength_range_slider'].max = wl_max
                    
                    with self.widgets['status_output']:
                        self.widgets['status_output'].clear_output()
                        print(f"✅ Converted to wavelength (nm)")
                        print(f"   Range: {wl_min:.1f} - {wl_max:.1f} nm")
            
            # Update all visualizations
            self.update_visualizations(update_heatmap=True)
            
            debug_print(f"Energy conversion complete. New unit: {self.wavelength_unit}", "APP")
            
        except Exception as e:
            import traceback
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"❌ Error converting: {str(e)}")
                print("\n🔍 Full traceback:")
                print(traceback.format_exc())
            debug_print(f"Error in energy conversion: {e}", "APP")
    
    def on_colorbar_apply(self, button):
        """Apply colorbar range from slider"""
        if not self.data_manager.is_data_loaded():
            return
        
        try:
            # Small delay to ensure any typed values are committed
            import time
            time.sleep(0.1)
            
            z_min, z_max = self.widgets['colorbar_range_slider'].value
            
            self.update_heatmap_colorbar(z_min, z_max)
            
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"✅ Colorbar range applied")
                print(f"   Range: {z_min:.1f} - {z_max:.1f}")
            
            debug_print(f"Colorbar range applied: {z_min:.1f} - {z_max:.1f}", "APP")
            
        except Exception as e:
            debug_print(f"Error applying colorbar: {e}", "APP")
    
    def update_heatmap_colorbar(self, z_min, z_max):
        """Update heatmap colorbar range dynamically"""
        if self.plot_manager.heatmap_fig is None:
            debug_print("No heatmap to update", "APP")
            return
        
        try:
            # Update the heatmap's zmin and zmax using FigureWidget
            with self.plot_manager.heatmap_fig.batch_update():
                self.plot_manager.heatmap_fig.data[0].zmin = z_min
                self.plot_manager.heatmap_fig.data[0].zmax = z_max
            
            debug_print(f"Updated heatmap colorbar: {z_min:.1f} - {z_max:.1f}", "APP")
            
        except Exception as e:
            debug_print(f"Error updating heatmap colorbar: {e}", "APP")
    
    def remove_peak_model(self, peak_idx):
        """Remove a peak model"""
        if len(self.peak_models) <= 1:
            with self.widgets['status_output']:
                print("⚠️ Must keep at least one peak model")
            return
        
        debug_print(f"Removing peak model {peak_idx}", "APP")
        
        # Find and remove the peak model
        model_to_remove = None
        for i, model in enumerate(self.peak_models):
            if hasattr(model, '_peak_idx') and model._peak_idx == peak_idx:
                model_to_remove = i
                break
        
        if model_to_remove is not None:
            self.peak_models.pop(model_to_remove)
            self.update_peak_list_container()
    
    def clear_peak_models(self):
        """Clear all peak models"""
        debug_print("Clearing all peak models", "APP")
        self.peak_models.clear()
        self.update_peak_list_container()
    
    def update_peak_list_container(self):
        """Update the peak list container widget with grid layout"""
        if len(self.peak_models) == 0:
            self.widgets['peak_list_container'].children = tuple()
        else:
            # Use GridBox for better layout (2 columns)
            from ipywidgets import GridBox
            
            self.widgets['peak_list_container'].children = tuple(self.peak_models)
    
    # =========================================================================
    # FITTING OPERATIONS
    # =========================================================================
    
    def prepare_fit_parameters(self):
        """Prepare fitting parameters from UI inputs"""
        params = {
            'background_model': 'None',
            'poly_degree': 2,
            'peak_models': []
        }
        
        for peak_model in self.peak_models:
            model_type = peak_model._widgets['type'].value
            
            peak_params = {
                'type': model_type
            }
            
            # Add parameters based on model type
            if model_type == 'Linear':
                peak_params['slope'] = peak_model._widgets['slope'].value
                peak_params['intercept'] = peak_model._widgets['intercept'].value
                
            elif model_type == 'Polynomial':
                peak_params['poly_degree'] = peak_model._widgets['poly_degree'].value
                
                # Use fitted coefficients if available
                if hasattr(peak_model, '_fitted_coeffs'):
                    peak_params['fitted_coeffs'] = peak_model._fitted_coeffs
                    
            else:  # Gaussian, Voigt, Lorentzian
                peak_params['center'] = peak_model._widgets['center'].value
                peak_params['height'] = peak_model._widgets['height'].value
                peak_params['sigma'] = peak_model._widgets['sigma'].value
                
                # Add fix flags
                peak_params['fix_center'] = peak_model._widgets['fix_center'].value
                peak_params['fix_height'] = peak_model._widgets['fix_height'].value
                peak_params['fix_sigma'] = peak_model._widgets['fix_sigma'].value
            
            params['peak_models'].append(peak_params)
        
        debug_print(f"Prepared fit parameters: {len(params['peak_models'])} models", "APP")
        return params

    def load_custom_background(self, file_content):
        """
        Load custom background from uploaded file
        
        Parameters:
        -----------
        file_content : bytes
            File content with 2 columns: wavelength, intensity
            
        Returns:
        --------
        array: Background intensity values
        """
        import io
        
        try:
            # Convert bytes to string
            if isinstance(file_content, bytes):
                content_str = file_content.decode('utf-8')
            elif isinstance(file_content, memoryview):
                content_str = file_content.tobytes().decode('utf-8')
            else:
                content_str = str(file_content)
            
            # Parse the file
            data = np.loadtxt(io.StringIO(content_str), delimiter=',')
            
            if data.shape[1] != 2:
                raise ValueError("File must have exactly 2 columns: wavelength, intensity")
            
            bg_wavelengths = data[:, 0]
            bg_intensities = data[:, 1]
            
            # Interpolate to match current wavelength grid
            from scipy.interpolate import interp1d
            interp_func = interp1d(bg_wavelengths, bg_intensities, 
                                   kind='linear', 
                                   bounds_error=False, 
                                   fill_value='extrapolate')
            
            background = interp_func(self.data_manager.wavelengths)
            
            debug_print(f"Loaded custom background: {len(bg_wavelengths)} points", "APP")
            return background
            
        except Exception as e:
            raise ValueError(f"Error loading custom background: {str(e)}")

    def on_custom_background_upload(self, change):
            """Handle custom background file upload"""
            if not change['new']:
                return
                
            try:
                # Get the uploaded files - change['new'] is a tuple of FileUpload objects
                uploaded_files = change['new']
                if not uploaded_files or len(uploaded_files) == 0:
                    return
                
                # Get first file from tuple
                file_upload = uploaded_files[0]
                file_name = file_upload['name']
                file_content = file_upload['content']
                
                # Try to load and validate the file
                background = self.load_custom_background(file_content)
                
                # Update status
                self.widgets['bg_custom_status'].value = f"✅ Loaded: {file_name} ({len(background)} points)"
                
                # Enable apply button
                self.widgets['bg_apply_btn'].disabled = False
                
                debug_print(f"Custom background uploaded: {file_name}", "APP")
                
            except Exception as e:
                self.widgets['bg_custom_status'].value = f"❌ Error: {str(e)}"
                debug_print(f"Error loading custom background: {e}", "APP")
    
    def on_fit_current(self, button):
        """Fit current spectrum"""
        if not self.data_manager.is_data_loaded():
            return
        
        try:
            debug_print("Fitting current spectrum", "APP")
            
            # Check if we have any peak models
            if len(self.peak_models) == 0:
                with self.widgets['status_output']:
                    self.widgets['status_output'].clear_output()
                    print("❌ No peak models defined. Add at least one peak model first.")
                return
            
            # Prepare fitting parameters
            fit_params = self.prepare_fit_parameters()
            debug_print(f"Fit params: {fit_params}", "APP")
            
            self.fitting_engine.create_fit_parameters(**fit_params)
            
            # Get wavelength range from slider
            wl_range = self.widgets['wavelength_range_slider'].value
            wavelengths = self.data_manager.wavelengths
            spectrum = self.data_manager.get_current_spectrum()
            
            # Check if range is limited (not the full range)
            wl_min = float(wavelengths.min())
            wl_max = float(wavelengths.max())
            is_limited = (wl_range[0] > wl_min) or (wl_range[1] < wl_max)
            
            if is_limited:
                # Fit only in the selected range
                mask = (wavelengths >= wl_range[0]) & (wavelengths <= wl_range[1])
                wavelengths_fit = wavelengths[mask]
                spectrum_fit = spectrum[mask]
                
                debug_print(f"Fitting limited range: {wl_range[0]:.1f} - {wl_range[1]:.1f} {self.wavelength_unit}", "APP")
                debug_print(f"Full data points: {len(wavelengths)}, Fit data points: {len(wavelengths_fit)}", "APP")
                debug_print(f"Wavelength fit range: {wavelengths_fit.min():.2f} - {wavelengths_fit.max():.2f}", "APP")
            else:
                wavelengths_fit = wavelengths
                spectrum_fit = spectrum
                debug_print("Fitting full wavelength range", "APP")
                debug_print(f"Data points: {len(wavelengths)}", "APP")
            
            # Perform fitting
            result = self.fitting_engine.fit_current_spectrum(
                wavelengths_fit,
                spectrum_fit
            )
            
            debug_print(f"Fit result: success={result.success}, rsquared={result.rsquared}", "APP")
            
            # Store for parameter update
            self.last_fit_result = result
            
            # Update R-squared display
            self.widgets['r_squared_display'].value = f"<b>R²:</b> {result.rsquared:.4f}"
            
            # Automatically update parameters from fit
            self.on_update_parameters(None)
            
            # Enable update parameters button (now just for manual re-update if needed)
            self.widgets['update_params_btn'].disabled = False
            
            # Create full-range arrays for plotting (pad with NaN outside fit range)
            if is_limited:
                # Create arrays same size as original wavelengths
                full_best_fit = np.full_like(wavelengths, np.nan)
                full_residual = np.full_like(wavelengths, np.nan)
                
                # Fill in the fitted range
                full_best_fit[mask] = result.best_fit
                full_residual[mask] = result.residual
                
                debug_print(f"Created padded arrays: {np.sum(~np.isnan(full_best_fit))} non-NaN values", "APP")
                
                # Create a modified result object for plotting
                class PlotResult:
                    def __init__(self, original_result, best_fit, residual, wavelengths_full, mask):
                        self.best_fit = best_fit
                        self.residual = residual
                        self.rsquared = original_result.rsquared
                        self.success = original_result.success
                        # Copy eval_components if it exists
                        if hasattr(original_result, 'eval_components'):
                            self._eval_components = {}
                            components = original_result.eval_components()
                            for comp_name, comp_values in components.items():
                                full_comp = np.full_like(wavelengths_full, np.nan)
                                full_comp[mask] = comp_values
                                self._eval_components[comp_name] = full_comp
                                debug_print(f"Component {comp_name}: {np.sum(~np.isnan(full_comp))} non-NaN values", "APP")
                        else:
                            self._eval_components = {}
                    
                    def eval_components(self):
                        """Return component evaluations - THIS METHOD MUST BE AT CLASS LEVEL, NOT INSIDE __init__"""
                        return self._eval_components
                
                plot_result = PlotResult(result, full_best_fit, full_residual, wavelengths, mask)
            else:
                plot_result = result
                debug_print("Using full result (no padding needed)", "APP")
            
            # Update spectrum plot with fit
            with self.widgets['spectrum_output']:
                self.widgets['spectrum_output'].clear_output()
                
                # Pass wavelength range if limited
                wl_range_display = wl_range if is_limited else None
                
                fig = self.plot_manager.create_spectrum_plot(
                    self.data_manager.wavelengths,
                    self.data_manager.get_current_spectrum(),
                    fit_result=plot_result,
                    wavelength_range=wl_range_display,
                    wavelength_unit=self.wavelength_unit
                )
                
                fig.show(renderer=config.PLOT_RENDERER)
            
            # Automatically update parameters from successful fit
            if result.rsquared > 0.5:  # Only auto-update if fit is decent
                self.on_update_parameters(None)
                auto_updated = True
            else:
                auto_updated = False
            
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"✅ Fit completed. R² = {result.rsquared:.4f}")
                if is_limited:
                    print(f"   Range: {wl_range[0]:.1f} - {wl_range[1]:.1f} {self.wavelength_unit}")
                if auto_updated:
                    print("✅ Parameters automatically updated for batch fitting")
                else:
                    print("⚠️ Low R² - click 'Update from Fit' manually if needed")
            
            debug_print(f"Fit complete: R²={result.rsquared:.4f}, auto-updated={auto_updated}", "APP")
                
        except Exception as e:
            import traceback
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"❌ Error fitting spectrum: {str(e)}")
                print("\n🔍 Full traceback:")
                print(traceback.format_exc())
            debug_print(f"Error in fitting: {e}", "APP")
    
    def on_update_parameters(self, button):
        """Update model parameters with fitted values"""
        try:
            debug_print("DEBUG: on_update_parameters called")
            if self.last_fit_result is None:
                with self.widgets['status_output']:
                    self.widgets['status_output'].clear_output()
                    debug_print("❌ No fit result available. Please fit current spectrum first.")
                return
            
            debug_print("Updating parameters from fit result", "APP")
            
            result = self.last_fit_result
            fitted_params = result.params
            
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print("🔄 Updating model parameters with fitted values...")
            
            # Update ALL model parameters in UI
            updated_count = 0
            for i, model in enumerate(self.peak_models):
                model_type = model._widgets['type'].value
                prefix = f'p{i}_'
                
                if model_type == 'Gaussian' or model_type == 'Voigt' or model_type == 'Lorentzian':
                    # Update center
                    center_param = f'{prefix}center'
                    if center_param in fitted_params:
                        model._widgets['center'].value = round(fitted_params[center_param].value, 3)
                        print(f"  Model {i + 1} center: {fitted_params[center_param].value:.3f}")
                        updated_count += 1
                    
                    # Update sigma
                    sigma_param = f'{prefix}sigma'
                    if sigma_param in fitted_params:
                        model._widgets['sigma'].value = round(fitted_params[sigma_param].value, 3)
                        print(f"  Model {i + 1} sigma: {fitted_params[sigma_param].value:.3f}")
                        updated_count += 1
                    
                    # Update height (converted from amplitude)
                    amplitude_param = f'{prefix}amplitude'
                    if amplitude_param in fitted_params and sigma_param in fitted_params:
                        amplitude = fitted_params[amplitude_param].value
                        sigma = fitted_params[sigma_param].value
                        if sigma > 0:
                            height = amplitude / (sigma * np.sqrt(2 * np.pi))
                            model._widgets['height'].value = round(height, 3)
                            print(f"  Model {i + 1} height: {height:.3f}")
                            updated_count += 1
                            
                elif model_type == 'Linear':
                    # Update slope and intercept
                    slope_param = f'{prefix}slope'
                    intercept_param = f'{prefix}intercept'
                    
                    if slope_param in fitted_params:
                        model._widgets['slope'].value = round(fitted_params[slope_param].value, 3)
                        print(f"  Model {i + 1} slope: {fitted_params[slope_param].value:.3f}")
                        updated_count += 1
                        
                    if intercept_param in fitted_params:
                        model._widgets['intercept'].value = round(fitted_params[intercept_param].value, 3)
                        print(f"  Model {i + 1} intercept: {fitted_params[intercept_param].value:.3f}")
                        updated_count += 1
                        
                elif model_type == 'Polynomial':
                    # Update polynomial coefficients
                    # First, figure out the degree
                    degree = model._widgets['poly_degree'].value
                    
                    # Extract all coefficients
                    coeffs = []
                    for j in range(degree + 1):
                        coeff_param = f'{prefix}c{j}'
                        if coeff_param in fitted_params:
                            coeffs.append(fitted_params[coeff_param].value)
                        else:
                            coeffs.append(0.0)  # Fallback
                    
                    # Store them (we'll need to add a way to pass these)
                    # For now, just print them
                    print(f"  Model {i + 1} (Polynomial degree {degree}):")
                    for j, coeff in enumerate(coeffs):
                        print(f"    c{j}: {coeff:.6f}")
                    
                    # Store coefficients in the model for later use
                    if not hasattr(model, '_fitted_coeffs'):
                        model._fitted_coeffs = {}
                    model._fitted_coeffs = coeffs
                    updated_count += 1
            
            with self.widgets['status_output']:
                print(f"✅ Updated {updated_count} parameters from fit")
            
            debug_print(f"Updated {updated_count} parameters from fit", "APP")
                
        except Exception as e:
            import traceback
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"❌ Error updating parameters: {str(e)}")
                print("\n🔍 Traceback:")
                traceback.print_exc()
            debug_print(f"Error updating parameters: {e}", "APP")
    
    def on_fit_all(self, button):
        """Fit all spectra using batch processing"""
        if not self.data_manager.is_data_loaded():
            return
        
        try:
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print("Starting batch fitting... This may take a while.")
            
            debug_print("Starting batch fitting for all spectra", "APP")
            
            # Show progress bar
            self.widgets['fit_progress'].layout.visibility = 'visible'
            self.widgets['fit_progress'].value = 0
            self.widgets['fit_progress'].description = 'Fitting...'
            
            # Prepare fitting parameters
            fit_params = self.prepare_fit_parameters()
            self.fitting_engine.create_fit_parameters(**fit_params)
            
            # Get wavelength range from slider
            wl_range = self.widgets['wavelength_range_slider'].value
            wavelengths = self.data_manager.wavelengths
            wl_min = float(wavelengths.min())
            wl_max = float(wavelengths.max())
            is_limited = (wl_range[0] > wl_min) or (wl_range[1] < wl_max)
            
            # Apply wavelength range mask if limited
            if is_limited:
                mask = (wavelengths >= wl_range[0]) & (wavelengths <= wl_range[1])
                wavelengths_fit = wavelengths[mask]
                data_matrix_fit = self.data_manager.data_matrix[:, mask]
                debug_print(f"Batch fitting with wavelength range: {wl_range[0]:.1f} - {wl_range[1]:.1f} {self.wavelength_unit}", "APP")
                debug_print(f"Full wavelengths: {len(wavelengths)}, Fit wavelengths: {len(wavelengths_fit)}", "APP")
            else:
                wavelengths_fit = wavelengths
                data_matrix_fit = self.data_manager.data_matrix
                debug_print("Batch fitting full wavelength range", "APP")
            
            # Perform batch fitting
            results = self.fitting_engine.fit_all_spectra(
                wavelengths_fit,
                data_matrix_fit,
                self.data_manager.timestamps
            )
            
            # Count successful fits
            successful = sum(1 for r in results.values() if r and r.get('success', False))
            
            # Store results for export
            self.fitting_engine.fitting_results = results
            
            # Enable export button
            self.widgets['export_btn'].disabled = False
            
            # Create time series visualizations with explicit clear
            with self.widgets['time_series_output']:
                self.widgets['time_series_output'].clear_output(wait=True)

            self.create_time_series_plots()
            
            debug_print("Time series plots refreshed", "APP")
            
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"✅ Batch fitting completed!")
                print(f"📊 Fitted {successful}/{len(results)} spectra successfully.")
            
            # Update progress bar to complete
            self.widgets['fit_progress'].value = 100
            self.widgets['fit_progress'].description = 'Complete!'
            self.widgets['fit_progress'].bar_style = 'success'
            
            debug_print(f"Batch fitting complete: {successful}/{len(results)} successful", "APP")
                
        except Exception as e:
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"❌ Error in batch fitting: {str(e)}")
            debug_print(f"Error in batch fitting: {e}", "APP")
    
    def on_fit_range(self, button):
        """Fit spectra in selected range"""
        if not self.data_manager.is_data_loaded():
            return
        
        try:
            start_idx = self.widgets['fit_start_idx'].value
            end_idx = self.widgets['fit_end_idx'].value
            
            if start_idx >= end_idx:
                with self.widgets['status_output']:
                    self.widgets['status_output'].clear_output()
                    print("❌ Start index must be less than end index")
                return
            
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"Starting range fitting ({start_idx} to {end_idx})...")
            
            debug_print(f"Starting range fitting: {start_idx} to {end_idx}", "APP")
            
            # Show progress bar
            self.widgets['fit_progress'].layout.visibility = 'visible'
            self.widgets['fit_progress'].value = 0
            self.widgets['fit_progress'].description = 'Fitting...'
            
            # Prepare fitting parameters
            fit_params = self.prepare_fit_parameters()
            self.fitting_engine.create_fit_parameters(**fit_params)
            
            # Get wavelength range from slider
            wl_range = self.widgets['wavelength_range_slider'].value
            wavelengths = self.data_manager.wavelengths
            wl_min = float(wavelengths.min())
            wl_max = float(wavelengths.max())
            is_limited = (wl_range[0] > wl_min) or (wl_range[1] < wl_max)
            
            # Apply wavelength range mask if limited
            if is_limited:
                mask = (wavelengths >= wl_range[0]) & (wavelengths <= wl_range[1])
                wavelengths_fit = wavelengths[mask]
                data_matrix_fit = self.data_manager.data_matrix[:, mask]
                debug_print(f"Range fitting with wavelength range: {wl_range[0]:.1f} - {wl_range[1]:.1f} {self.wavelength_unit}", "APP")
                debug_print(f"Full wavelengths: {len(wavelengths)}, Fit wavelengths: {len(wavelengths_fit)}", "APP")
            else:
                wavelengths_fit = wavelengths
                data_matrix_fit = self.data_manager.data_matrix
                debug_print("Range fitting full wavelength range", "APP")
            
            # Perform batch fitting on range with progress callback
            results = self.fitting_engine.fit_all_spectra(
                wavelengths_fit,
                data_matrix_fit,
                self.data_manager.timestamps,
                fit_range=(start_idx, end_idx)
            )
            
            # Count successful fits
            successful = sum(1 for r in results.values() if r and r.get('success', False))
            
            # Enable export button
            self.widgets['export_btn'].disabled = False
            
            # Create time series visualizations with explicit clear
            with self.widgets['time_series_output']:
                self.widgets['time_series_output'].clear_output(wait=True)
            
            self.create_time_series_plots()
            
            debug_print("Time series plots refreshed", "APP")
            
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"✅ Range fitting completed!")
                print(f"📊 Fitted {successful}/{len(results)} spectra successfully.")
            
            # Update progress bar to complete
            self.widgets['fit_progress'].value = 100
            self.widgets['fit_progress'].description = 'Complete!'
            self.widgets['fit_progress'].bar_style = 'success'
            
            debug_print(f"Range fitting complete: {successful}/{len(results)} successful", "APP")
                
        except Exception as e:
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"❌ Error in range fitting: {str(e)}")
            debug_print(f"Error in range fitting: {e}", "APP")
    
    # =========================================================================
    # EXPORT
    # =========================================================================
    
    def on_export_results(self, button):
        """Export fitting results"""
        if not self.fitting_engine.has_fitting_results():
            with self.widgets['export_output']:
                self.widgets['export_output'].clear_output()
                print("❌ No fitting results to export. Please fit spectra first.")
            return
        
        try:
            debug_print("Exporting results", "APP")
            
            from datetime import datetime
            import base64
            from IPython.display import HTML, display
            
            # Create filename
            filename = f"pl_fitting_results_{datetime.now().strftime(config.TIMESTAMP_FORMAT)}.xlsx"
            
            # Export to Excel in memory
            import io
            excel_buffer = io.BytesIO()
            
            # Use export_utils to create Excel file
            self.export_utils.export_to_excel(
                self.fitting_engine.fitting_results,
                self.data_manager.timestamps,
                filename=excel_buffer  # Write to buffer instead of file
            )
            
            # Get the buffer content
            excel_buffer.seek(0)
            excel_data = excel_buffer.read()
            
            # Encode to base64
            b64 = base64.b64encode(excel_data).decode()
            
            # Create download link with JavaScript
            js_code = f"""
            var link = document.createElement('a');
            link.href = 'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}';
            link.download = '{filename}';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            """
            
            with self.widgets['export_output']:
                self.widgets['export_output'].clear_output()
                print(f"✅ Results prepared for download!")
                print(f"📊 File: {filename}")
                print()
                
                # Display download button
                download_button_html = f"""
                <button onclick="{js_code}" style="
                    background-color: #28a745;
                    color: white;
                    padding: 10px 20px;
                    border: none;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 14px;
                    font-weight: bold;
                ">
                    📥 Download {filename}
                </button>
                """
                
                display(HTML(download_button_html))
                print()
                print("💡 Click the button above to download. If it doesn't work automatically, the download should start shortly.")
            
            debug_print(f"Results prepared for download: {filename}", "APP")
                
        except Exception as e:
            import traceback
            with self.widgets['export_output']:
                self.widgets['export_output'].clear_output()
                print(f"❌ Error exporting results: {str(e)}")
                print("\n🔍 Full traceback:")
                print(traceback.format_exc())
            debug_print(f"Error exporting: {e}", "APP")
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def update_visualizations(self, update_heatmap: bool = False):
        """Update both heatmap and spectrum visualizations
        Parameters
        ----------
        update_heatmap      : boolean value that decides whether a new heatmap gets drawn"""
        if not self.data_manager.is_data_loaded():
            return
        
        debug_print("Updating visualizations", "APP")
        
        # Update time display
        current_time = self.data_manager.get_current_time_value()
        self.widgets['time_display'].value = f"Time: {current_time:.3f}s"
        
        # Update heatmap
        self.update_heatmap(update_heatmap=update_heatmap)
        
        # Update spectrum plot
        self.update_spectrum_plot()
    
    def update_heatmap(self, update_heatmap: bool = False):
        """Update heatmap visualization - only update line if heatmap exists
        Parameters
        ----------
        update_heatmap      : boolean value that decides whether a new heatmap gets drawn"""

        # Check if heatmap already exists
        if self.plot_manager.heatmap_fig is None or update_heatmap:
            # Create heatmap for the first time
            with self.widgets['heatmap_output']:
                self.widgets['heatmap_output'].clear_output()
                
                fig = self.plot_manager.create_heatmap(
                    self.data_manager.data_matrix,
                    self.data_manager.wavelengths,
                    self.data_manager.timestamps,
                    current_time_idx=self.data_manager.current_time_idx,
                    wavelength_unit=self.wavelength_unit
                )
                
                display(fig)
        else:
            # Just update the line position (no redraw needed)
            self.plot_manager.update_heatmap_line(
                self.data_manager.timestamps,
                self.data_manager.current_time_idx
            )
    
    def update_spectrum_plot(self, fit_result=None):
        """Update spectrum plot"""
        with self.widgets['spectrum_output']:
            self.widgets['spectrum_output'].clear_output()
            
            # Get wavelength range for display if limited
            wl_range = self.widgets['wavelength_range_slider'].value
            wavelengths = self.data_manager.wavelengths
            wl_min = float(wavelengths.min())
            wl_max = float(wavelengths.max())
            is_limited = (wl_range[0] > wl_min) or (wl_range[1] < wl_max)
            
            wl_range_display = wl_range if is_limited else None
            
            fig = self.plot_manager.create_spectrum_plot(
                self.data_manager.wavelengths,
                self.data_manager.get_current_spectrum(),
                fit_result=fit_result,
                wavelength_range=wl_range_display,
                wavelength_unit=self.wavelength_unit
            )
            
            # Apply current wavelength range visualization BEFORE displaying
            wavelength_range = self.widgets['wavelength_range_slider'].value
            if wavelength_range[0] != wl_min or wavelength_range[1] != wl_max:
                self._update_wavelength_range_on_spectrum(wavelength_range)
            
            # Display the FigureWidget directly
            display(fig)
    
    def create_time_series_plots(self):
        """Create time series plots from fitting results"""
        if not self.fitting_engine.has_fitting_results():
            return
        
        try:
            debug_print("Creating time series plots", "APP")
            
            # Explicitly clear output first
            with self.widgets['time_series_output']:
                self.widgets['time_series_output'].clear_output(wait=True)
            
            self.plot_manager.create_time_series_plots(
                self.fitting_engine.fitting_results,
                output_widget=self.widgets['time_series_output'],
                wavelength_unit=self.wavelength_unit
            )
            
            debug_print("Time series plots created", "APP")
            
        except Exception as e:
            with self.widgets['time_series_output']:
                self.widgets['time_series_output'].clear_output()
                print(f"❌ Error creating time series plots: {str(e)}")
            debug_print(f"Error creating time series plots: {e}", "APP")
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def on_background_method_change(self, change):
        """Handle background method selection change"""
        method = change['new']
        debug_print(f"Background method changed to: {method}", "APP")
        
        # Get data loaded state
        data_loaded = self.data_manager.is_data_loaded()
        
        # Hide all containers first
        if hasattr(self.gui_layouts, '_bg_containers'):
            for container in self.gui_layouts._bg_containers.values():
                container.layout.visibility = 'hidden'
                container.layout.display = 'none'
        
        # If None is selected, disable all buttons and return
        if method == 'None':
            self.widgets['bg_apply_btn'].disabled = True
            self.widgets['bg_autofit_btn'].disabled = True
            self.widgets['bg_autofit_btn'].layout.visibility = 'hidden'
            return
        
        # Show and enable the selected method's container
        if hasattr(self.gui_layouts, '_bg_containers') and method in self.gui_layouts._bg_containers:
            self.gui_layouts._bg_containers[method].layout.visibility = 'visible'
            self.gui_layouts._bg_containers[method].layout.display = 'flex'
            
            # Enable widgets for the selected method
            if method == 'Manual' and data_loaded:
                self.widgets['bg_manual_start'].disabled = False
                self.widgets['bg_manual_num'].disabled = False
            elif method == 'Linear' and data_loaded:
                self.widgets['bg_linear_slope'].disabled = False
                self.widgets['bg_linear_intercept'].disabled = False
            elif method == 'Polynomial' and data_loaded:
                self.widgets['bg_poly_degree'].disabled = False
                self.widgets['bg_poly_coeffs'].disabled = False
            elif method == 'Exponential' and data_loaded:
                self.widgets['bg_exp_amplitude'].disabled = False
                self.widgets['bg_exp_decay'].disabled = False
                self.widgets['bg_exp_offset'].disabled = False
            elif method == 'Custom' and data_loaded:
                self.widgets['bg_custom_upload'].disabled = False
        
        # Enable/disable action buttons
        can_apply = data_loaded
        self.widgets['bg_apply_btn'].disabled = not can_apply
        
        # Auto-fit button only for model-based methods
        if method in ['Linear', 'Polynomial', 'Exponential']:
            self.widgets['bg_autofit_btn'].disabled = not data_loaded
            self.widgets['bg_autofit_btn'].layout.visibility = 'visible'
        else:
            self.widgets['bg_autofit_btn'].disabled = True
            self.widgets['bg_autofit_btn'].layout.visibility = 'hidden'
    
    def on_background_autofit(self, button):
        """Auto-fit background model to current spectrum"""
        if not self.data_manager.is_data_loaded():
            return
        
        method = self.widgets['background_method'].value
        current_spectrum = self.data_manager.get_current_spectrum()
        wavelengths = self.data_manager.wavelengths
        
        try:
            debug_print(f"Auto-fitting background: {method}", "APP")
            
            if method == 'Linear':
                # Fit linear: y = slope * x + intercept
                coeffs = np.polyfit(wavelengths, current_spectrum, 1)
                self.widgets['bg_linear_slope'].value = float(coeffs[0])
                self.widgets['bg_linear_intercept'].value = float(coeffs[1])
                self.background_model = coeffs[0] * wavelengths + coeffs[1]
                debug_print(f"Background model set (Linear): shape={self.background_model.shape}, min={self.background_model.min():.2f}, max={self.background_model.max():.2f}", "APP")
                
            elif method == 'Polynomial':
                degree = self.widgets['bg_poly_degree'].value
                coeffs = np.polyfit(wavelengths, current_spectrum, degree)
                self.widgets['bg_poly_coeffs'].value = ', '.join([f'{c:.2e}' for c in coeffs])
                self.background_model = np.polyval(coeffs, wavelengths)
                debug_print(f"Background model set (Polynomial): shape={self.background_model.shape}, min={self.background_model.min():.2f}, max={self.background_model.max():.2f}", "APP")
                
            elif method == 'Exponential':
                # Fit exponential: y = amplitude * exp(-decay * x) + offset
                from scipy.optimize import curve_fit
                
                def exp_func(x, amplitude, decay, offset):
                    return amplitude * np.exp(-decay * x) + offset
                
                # Initial guess
                p0 = [current_spectrum.max(), 0.001, current_spectrum.min()]
                popt, _ = curve_fit(exp_func, wavelengths, current_spectrum, p0=p0, maxfev=5000)
                
                self.widgets['bg_exp_amplitude'].value = float(popt[0])
                self.widgets['bg_exp_decay'].value = float(popt[1])
                self.widgets['bg_exp_offset'].value = float(popt[2])
                self.background_model = exp_func(wavelengths, *popt)
                debug_print(f"Background model set (Exponential): shape={self.background_model.shape}, min={self.background_model.min():.2f}, max={self.background_model.max():.2f}", "APP")
            
            # Update spectrum plot to show background
            self.update_spectrum_with_background()
            
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"✅ {method} background auto-fitted")
                print("   Adjust parameters if needed, then click 'Apply Background'")
            
            debug_print(f"Background auto-fit complete: {method}", "APP")
                
        except Exception as e:
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"❌ Error auto-fitting background: {str(e)}")
            debug_print(f"Error in background autofit: {e}", "APP")
    
    def on_background_apply(self, button):
        """Apply background removal to data"""
        if not self.data_manager.is_data_loaded():
            return
        
        method = self.widgets['background_method'].value
        
        if method == 'None':
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print("⚠️ Select a background method first")
            return
        
        try:
            debug_print(f"Applying background: {method}", "APP")
            
            # Store original if not already stored
            if self.original_data_matrix is None:
                self.original_data_matrix = self.data_manager.data_matrix.copy()
            
            wavelengths = self.data_manager.wavelengths
            
            if method == 'Manual':
                start_idx = self.widgets['bg_manual_start'].value
                num_curves = self.widgets['bg_manual_num'].value
                end_idx = min(start_idx + num_curves, len(self.data_manager.timestamps))
                
                # Calculate average background
                background = np.mean(self.original_data_matrix[start_idx:end_idx, :], axis=0)
                
            elif method == 'Linear':
                slope = self.widgets['bg_linear_slope'].value
                intercept = self.widgets['bg_linear_intercept'].value
                background = slope * wavelengths + intercept
                
            elif method == 'Polynomial':
                coeffs_str = self.widgets['bg_poly_coeffs'].value
                coeffs = [float(c.strip()) for c in coeffs_str.split(',')]
                background = np.polyval(coeffs, wavelengths)
                
            elif method == 'Exponential':
                amplitude = self.widgets['bg_exp_amplitude'].value
                decay = self.widgets['bg_exp_decay'].value
                offset = self.widgets['bg_exp_offset'].value
                background = amplitude * np.exp(-decay * wavelengths) + offset
            
            elif method == 'Custom':
                            # Check if file was uploaded
                            uploaded_files = self.widgets['bg_custom_upload'].value
                            if not uploaded_files or len(uploaded_files) == 0:
                                with self.widgets['status_output']:
                                    self.widgets['status_output'].clear_output()
                                    print("❌ Please upload a background file first")
                                return
                            
                            # Get the uploaded file - value is a tuple
                            file_upload = uploaded_files[0]
                            background = self.load_custom_background(file_upload['content'])
            
            # Apply background subtraction
            self.data_manager.data_matrix = self.original_data_matrix - background
            self.data_manager.current_spectrum = self.data_manager.data_matrix[
                self.data_manager.current_time_idx, :
            ]
            
            self.background_applied = True
            self.background_model = background
            
            # Enable remove button
            self.widgets['bg_remove_btn'].disabled = False
            
            # Update visualizations
            self.plot_manager.heatmap_fig = None
            self.update_visualizations()
            
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"✅ {method} background applied to all data")

                # Close the background handling accordion
                if self.background_accordion is not None:
                    self.background_accordion.selected_index = None
                    debug_print("Closed background handling accordion", "APP")
            
            # Close the background handling accordion
            try:
                # Find the background accordion widget
                for widget in self.gui_layouts.create_fitting_control_section().children:
                    if hasattr(widget, 'set_title') and hasattr(widget, 'selected_index'):
                        # Check if this is the background handling accordion
                        if widget.get_title(0) == "🎨 Background Handling":
                            widget.selected_index = None
                            debug_print("Closed background handling accordion", "APP")
                            break
            except Exception as e:
                debug_print(f"Could not close accordion: {e}", "APP")
            
            debug_print(f"Background applied: {method}", "APP")
                
        except Exception as e:
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"❌ Error applying background: {str(e)}")
            debug_print(f"Error applying background: {e}", "APP")
    
    def on_background_remove(self, button):
        """Remove applied background and restore original data"""
        if self.original_data_matrix is None:
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print("⚠️ No background to remove")
            return
        
        try:
            debug_print("Removing background", "APP")
            
            # Restore original data
            self.data_manager.data_matrix = self.original_data_matrix.copy()
            self.data_manager.current_spectrum = self.data_manager.data_matrix[
                self.data_manager.current_time_idx, :
            ]
            
            self.background_applied = False
            self.background_model = None
            
            # Disable remove button
            self.widgets['bg_remove_btn'].disabled = True
            
            # Update visualizations
            self.plot_manager.heatmap_fig = None
            self.update_visualizations()
            
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print("✅ Background removed - original data restored")
            
            debug_print("Background removed", "APP")
                
        except Exception as e:
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"❌ Error removing background: {str(e)}")
            debug_print(f"Error removing background: {e}", "APP")
    
    def update_spectrum_with_background(self):
        """Update spectrum plot showing the fitted background"""
        with self.widgets['spectrum_output']:
            self.widgets['spectrum_output'].clear_output()
            
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            # Current spectrum
            fig.add_trace(go.Scatter(
                x=self.data_manager.wavelengths,
                y=self.data_manager.get_current_spectrum(),
                mode='lines',
                name='Spectrum',
                line=dict(color='blue', width=2)
            ))
            
            # Background model
            if self.background_model is not None:
                fig.add_trace(go.Scatter(
                    x=self.data_manager.wavelengths,
                    y=self.background_model,
                    mode='lines',
                    name='Background Model',
                    line=dict(color='red', width=2, dash='dash')
                ))
            
            fig.update_layout(
                title="Spectrum with Background Model",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Intensity",
                height=500,
                template='plotly_white'
            )
            
            fig.show(renderer=config.PLOT_RENDERER)

    def on_subtract_background(self, button):
        """Apply or remove manual background subtraction"""
        if not self.data_manager.is_data_loaded():
            return
        
        try:
            start_idx = self.widgets['background_time_start'].value
            num_curves = self.widgets['background_num_curves'].value
            
            # Ensure we don't go beyond the data
            end_idx = min(start_idx + num_curves, len(self.data_manager.timestamps))
            
            if start_idx >= len(self.data_manager.timestamps):
                with self.widgets['status_output']:
                    self.widgets['status_output'].clear_output()
                    print("❌ Start time index is beyond the data range")
                return
            
            # Calculate background (average of selected time range)
            background_spectra = self.data_manager.data_matrix[start_idx:end_idx, :]
            background_average = np.mean(background_spectra, axis=0)
            
            # Store original data if not already stored
            if self.original_data_matrix is None:
                self.original_data_matrix = self.data_manager.data_matrix.copy()
            
            # Apply or remove background subtraction
            if self.widgets['background_subtract_checkbox'].value:
                self.data_manager.data_matrix = self.original_data_matrix - background_average
            else:
                self.data_manager.data_matrix = self.original_data_matrix.copy()
            
            # Update current spectrum
            self.data_manager.current_spectrum = self.data_manager.data_matrix[
                self.data_manager.current_time_idx, :
            ]
            
            # Force heatmap recreation since data changed
            self.plot_manager.heatmap_fig = None
            
            # Update visualizations
            self.update_visualizations()
            
            with self.widgets['status_output']:
                        self.widgets['status_output'].clear_output()
                        if self.widgets['background_subtract_checkbox'].value:
                            time_start = self.data_manager.timestamps[start_idx]
                            time_end = self.data_manager.timestamps[end_idx-1]
                            print(f"✅ Manual background subtracted from data")
                            print(f"   Region: {time_start:.3f}s - {time_end:.3f}s ({num_curves} curves)")
                            print(f"   💡 Tip: You can still use model-based background during fitting")
                        else:
                            print("✅ Manual background removed - original data restored")
            
            debug_print(f"Background subtraction applied: {self.widgets['background_subtract_checkbox'].value}", "APP")
                    
        except Exception as e:
            with self.widgets['status_output']:
                self.widgets['status_output'].clear_output()
                print(f"❌ Error in background subtraction: {str(e)}")
            debug_print(f"Error in background subtraction: {e}", "APP")
    
    # =========================================================================
    # DISPLAY
    # =========================================================================
    
    def display_app(self):
        """Display the complete application interface"""
        debug_print("Displaying application", "APP")
        
        header, main_content = self.gui_layouts.create_main_layout()
        
        # Store accordion reference after layout is created
        self._store_accordion_reference()
        
        display(header)
        display(main_content)
        
        debug_print("Application displayed", "APP")