# gui_components.py
"""
GUI components for Photoluminescence Analysis App
Creates all individual widgets
"""
import ipywidgets as widgets
import config
from utils import debug_print


class GUIComponents:
    """Factory class for creating GUI widgets"""
    
    def __init__(self, h5_available=False):
        self.h5_available = h5_available
        self.widgets = {}
        
    def create_file_upload_widgets(self):
        """Create file upload related widgets"""
        if self.h5_available:
            # H5 mode dropdown
            mode_options = [(v, k) for k, v in config.H5_MODES.items()]
            self.widgets['mode_dropdown'] = widgets.Dropdown(
                options=mode_options,
                value=config.DEFAULT_H5_MODE,
                description='Measurement:',
                style={'description_width': 'initial'}
            )
            debug_print("Created H5 mode dropdown", "GUI")
        else:
            # File upload widget
            self.widgets['file_upload'] = widgets.FileUpload(
                accept=config.ACCEPTED_FILE_TYPES,
                multiple=config.ACCEPT_MULTIPLE_FILES,
                description='Upload Data'
            )
            debug_print("Created file upload widget", "GUI")
        
        # Status output
        self.widgets['status_output'] = widgets.Output()

        # Energy conversion button (only for PL data)
        self.widgets['convert_energy_btn'] = widgets.Button(
            description='🔄 nm ↔ eV',
            button_style='info',
            tooltip='Convert between wavelength (nm) and energy (eV)',
            layout=widgets.Layout(width='120px'),
            disabled=True  # Disabled until PL data is loaded
        )
        
        # Energy unit display
        self.widgets['energy_unit_display'] = widgets.Label(value="λ (nm)")
        
        debug_print("Created energy conversion widgets", "GUI")
        
        return self.widgets
    
    def create_time_control_widgets(self):
        """Create time control widgets"""
        # Time slider
        self.widgets['time_slider'] = widgets.IntSlider(
            value=0,
            min=0,
            max=100,
            step=1,
            description='',
            disabled=True,
            continuous_update=False,
            layout=widgets.Layout(width=config.TIME_SLIDER_WIDTH)
        )
        
        # Time input
        self.widgets['time_input'] = widgets.BoundedIntText(
            value=0,
            min=0,
            max=100,
            step=1,
            description='Time Index:',
            disabled=True,
            layout=widgets.Layout(width=config.TIME_INPUT_WIDTH)
        )
        
        # Time display label
        self.widgets['time_display'] = widgets.Label(value="Time: Not loaded")
        
        debug_print("Created time control widgets", "GUI")
        return self.widgets
    
    def create_peak_detection_widgets(self):
        """Create peak detection widgets"""
        # Auto-detect button
        self.widgets['auto_detect_btn'] = widgets.Button(
            description='🔍 Auto-Detect Peaks',
            button_style='info',
            tooltip='Automatically detect peaks in current spectrum',
            layout=widgets.Layout(width='180px')
        )
        
        # Detection parameters
        self.widgets['peak_height_threshold'] = widgets.FloatText(
            value=0.0,
            description='Min Height:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='160px')
        )
        
        self.widgets['peak_prominence'] = widgets.FloatText(
            value=0.0,
            description='Min Prom:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='160px')
        )
        
        self.widgets['peak_distance'] = widgets.IntText(
            value=5,
            description='Min Dist:',
            style={'description_width': '80px'},
            layout=widgets.Layout(width='160px')
        )
        
        debug_print("Created peak detection widgets", "GUI")
        return self.widgets
    
    def create_peak_model_widgets(self):
        """Create peak model configuration widgets"""
        # Add peak button
        self.widgets['add_peak_btn'] = widgets.Button(
            description='➕ Add Model',  # Changed from '➕ Add Peak'
            button_style='success',
            tooltip='Add a new model to the fit',  # Changed tooltip
            layout=widgets.Layout(width='120px')
        )
        
        # Peak list container
        self.widgets['peak_list_container'] = widgets.VBox([])
        
        debug_print("Created peak model widgets", "GUI")
        return self.widgets
    
    def create_background_handling_widgets(self):
        """Create unified background handling widgets"""
        # Main dropdown
        self.widgets['background_method'] = widgets.Dropdown(
            options=config.BACKGROUND_OPTIONS,
            value=config.DEFAULT_BACKGROUND_METHOD,
            description='Method:',
            style={'description_width': 'initial'}
        )
        
        # Manual method widgets
        self.widgets['bg_manual_start'] = widgets.BoundedIntText(
            value=config.DEFAULT_BACKGROUND_START_IDX,
            min=0,
            max=100,
            description='Start Index:',
            disabled=True,
            layout=widgets.Layout(width='180px', visibility='none')
        )
        
        self.widgets['bg_manual_num'] = widgets.BoundedIntText(
            value=config.DEFAULT_BACKGROUND_NUM_CURVES,
            min=1,
            max=50,
            description='# Curves:',
            disabled=True,
            layout=widgets.Layout(width='180px', visibility='none')
        )

        self.widgets['bg_manual_description'] = widgets.HTML(
            value="<i>Select starting time index and number of curves to average and subtract from all spectra</i>",
            layout=widgets.Layout(width='380px', margin='5px 0px 10px 0px')
        )
        
        # Linear method widgets
        self.widgets['bg_linear_slope'] = widgets.FloatText(
            value=0.0,
            description='Slope:',
            disabled=True,
            layout=widgets.Layout(width='180px', visibility='none')
        )
        
        self.widgets['bg_linear_intercept'] = widgets.FloatText(
            value=0.0,
            description='Intercept:',
            disabled=True,
            layout=widgets.Layout(width='180px', visibility='none')
        )
        
        # Polynomial method widgets
        self.widgets['bg_poly_degree'] = widgets.IntText(
            value=config.DEFAULT_POLY_DEGREE,
            description='Degree:',
            min=1,
            max=10,
            disabled=True,
            layout=widgets.Layout(width='180px', visibility='none')
        )
        
        self.widgets['bg_poly_coeffs'] = widgets.Textarea(
            value='',
            description='Coefficients:',
            placeholder='Will show after auto-fit',
            disabled=True,
            layout=widgets.Layout(width='350px', height='60px', visibility='none')
        )
        
        # Exponential method widgets
        self.widgets['bg_exp_amplitude'] = widgets.FloatText(
            value=1000.0,
            description='Amplitude:',
            disabled=True,
            layout=widgets.Layout(width='180px', visibility='none')
        )
        
        self.widgets['bg_exp_decay'] = widgets.FloatText(
            value=0.001,
            description='Decay:',
            disabled=True,
            layout=widgets.Layout(width='180px', visibility='none')
        )
        
        self.widgets['bg_exp_offset'] = widgets.FloatText(
            value=0.0,
            description='Offset:',
            disabled=True,
            layout=widgets.Layout(width='180px', visibility='none')
        )
        
        # Action buttons
        self.widgets['bg_autofit_btn'] = widgets.Button(
            description='Auto-Fit Background',
            button_style='info',
            disabled=True,
            layout=widgets.Layout(width='180px', visibility='none')
        )
        
        self.widgets['bg_apply_btn'] = widgets.Button(
            description='Apply Background',
            button_style='warning',
            disabled=True,
            layout=widgets.Layout(width='180px')
        )
        
        self.widgets['bg_remove_btn'] = widgets.Button(
            description='Remove Background',
            button_style='danger',
            disabled=True,
            layout=widgets.Layout(width='180px')
        )
        
        debug_print("Created background handling widgets", "GUI")
        return self.widgets

    def create_background_handling_widgets(self):
        """Create unified background handling widgets"""
        # Main dropdown
        self.widgets['background_method'] = widgets.Dropdown(
            options=config.BACKGROUND_OPTIONS,
            value=config.DEFAULT_BACKGROUND_METHOD,
            description='Method:',
            style={'description_width': 'initial'}
        )
        
        # Manual method widgets
        self.widgets['bg_manual_start'] = widgets.BoundedIntText(
            value=config.DEFAULT_BACKGROUND_START_IDX,
            min=0,
            max=100,
            description='Start Index:',
            disabled=True,
            layout=widgets.Layout(width='180px', visibility='hidden')
        )
        
        self.widgets['bg_manual_num'] = widgets.BoundedIntText(
            value=config.DEFAULT_BACKGROUND_NUM_CURVES,
            min=1,
            max=50,
            description='# Curves:',
            disabled=True,
            layout=widgets.Layout(width='180px', visibility='hidden')
        )

        self.widgets['bg_manual_description'] = widgets.HTML(
            value="<i>Select starting time index and number of curves to average and subtract from all spectra</i>",
            layout=widgets.Layout(width='380px', margin='5px 0px 10px 0px', visibility='hidden')
        )
        
        # Linear method widgets
        self.widgets['bg_linear_slope'] = widgets.FloatText(
            value=0.0,
            description='Slope:',
            disabled=True,
            layout=widgets.Layout(width='180px', visibility='hidden')
        )
        
        self.widgets['bg_linear_intercept'] = widgets.FloatText(
            value=0.0,
            description='Intercept:',
            disabled=True,
            layout=widgets.Layout(width='180px', visibility='hidden')
        )
        
        # Polynomial method widgets
        self.widgets['bg_poly_degree'] = widgets.IntText(
            value=config.DEFAULT_POLY_DEGREE,
            min=1,
            max=10,
            description='Degree:',
            disabled=True,
            layout=widgets.Layout(width='180px', visibility='hidden')
        )
        
        self.widgets['bg_poly_coeffs'] = widgets.Textarea(
            value='',
            description='Coefficients:',
            placeholder='Will show after auto-fit',
            disabled=True,
            layout=widgets.Layout(width='350px', height='60px', visibility='hidden')
        )
        
        # Exponential method widgets
        self.widgets['bg_exp_amplitude'] = widgets.FloatText(
            value=1000.0,
            description='Amplitude:',
            disabled=True,
            layout=widgets.Layout(width='180px', visibility='hidden')
        )
        
        self.widgets['bg_exp_decay'] = widgets.FloatText(
            value=0.001,
            description='Decay:',
            disabled=True,
            layout=widgets.Layout(width='180px', visibility='hidden')
        )
        
        self.widgets['bg_exp_offset'] = widgets.FloatText(
            value=0.0,
            description='Offset:',
            disabled=True,
            layout=widgets.Layout(width='180px', visibility='hidden')
        )
        
        # Custom background widgets
        self.widgets['bg_custom_description'] = widgets.HTML(
            value="<i>Upload a 2-column file (wavelength, intensity) to use as background</i>",
            layout=widgets.Layout(width='380px', margin='5px 0px 10px 0px', visibility='hidden')
        )
        
        self.widgets['bg_custom_upload'] = widgets.FileUpload(
            accept='.txt,.csv,.dat',
            multiple=False,
            description='Upload BG:',
            disabled=True,
            layout=widgets.Layout(width='380px', visibility='hidden'),
            style={'description_width': '80px'}
        )
        
        self.widgets['bg_custom_status'] = widgets.HTML(
            value="",
            layout=widgets.Layout(width='380px', visibility='hidden')
        )
        
        # Action buttons
        self.widgets['bg_autofit_btn'] = widgets.Button(
            description='Auto-Fit Background',
            button_style='info',
            disabled=True,
            layout=widgets.Layout(width='180px', visibility='hidden')
        )
        
        self.widgets['bg_apply_btn'] = widgets.Button(
            description='Apply Background',
            button_style='warning',
            disabled=True,
            layout=widgets.Layout(width='180px')
        )
        
        self.widgets['bg_remove_btn'] = widgets.Button(
            description='Remove Background',
            button_style='danger',
            disabled=True,
            layout=widgets.Layout(width='180px')
        )
        
        debug_print("Created background handling widgets", "GUI")
        return self.widgets
    
    def create_fitting_action_widgets(self):
        """Create fitting action buttons"""
        self.widgets['fit_current_btn'] = widgets.Button(
            description='🎯 Fit Current',
            button_style='primary',
            tooltip='Fit the current spectrum',
            layout=widgets.Layout(width='160px')
        )
        
        self.widgets['update_params_btn'] = widgets.Button(
            description='🔄 Update from Fit',
            button_style='',
            tooltip='Update initial parameters from current fit',
            layout=widgets.Layout(width='160px')
        )
        
        self.widgets['r_squared_display'] = widgets.HTML(
            value="<b>R²:</b> Not fitted yet"
        )
        
        debug_print("Created fitting action widgets", "GUI")
        return self.widgets

    def create_wavelength_range_widgets(self):
        """Create wavelength range selection widgets"""
        self.widgets['wavelength_range_slider'] = widgets.FloatRangeSlider(
            value=[400, 800],
            min=300,
            max=1000,
            step=0.001,
            description='λ Range (nm):',
            disabled=True,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.3f',
            layout=widgets.Layout(width='400px'),
            style={'description_width': 'initial'}
        )
        
        debug_print("Created wavelength range widgets", "GUI")
        return self.widgets
    
    def create_batch_fitting_widgets(self):
        """Create batch fitting widgets"""
        self.widgets['fit_all_btn'] = widgets.Button(
            description='▶️ Fit All Spectra',
            button_style='warning',
            tooltip='Fit all spectra in dataset',
            layout=widgets.Layout(width='160px')
        )
        
        self.widgets['fit_all_range_btn'] = widgets.Button(
            description='📊 Fit Range',
            button_style='warning',
            tooltip='Fit spectra in selected range',
            layout=widgets.Layout(width='160px')
        )
        
        # Range selection
        self.widgets['fit_start_idx'] = widgets.BoundedIntText(
            value=0,
            min=0,
            max=100,
            description='Start:',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='140px')
        )
        
        self.widgets['fit_end_idx'] = widgets.BoundedIntText(
            value=100,
            min=0,
            max=100,
            description='End:',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='140px')
        )
        
        debug_print("Created batch fitting widgets", "GUI")
        return self.widgets
    
    def create_export_widgets(self):
        """Create export widgets"""
        self.widgets['export_btn'] = widgets.Button(
            description='💾 Export Results',
            button_style='success',
            tooltip='Export fitting results to Excel',
            layout=widgets.Layout(width='160px')
        )
        
        self.widgets['export_output'] = widgets.Output()
        
        debug_print("Created export widgets", "GUI")
        return self.widgets
    
    def create_output_widgets(self):
        """Create output display widgets"""
        self.widgets['heatmap_output'] = widgets.Output()
        self.widgets['spectrum_output'] = widgets.Output()
        self.widgets['time_series_output'] = widgets.Output()
        
        debug_print("Created output widgets", "GUI")
        return self.widgets
    
    def create_all_widgets(self):
        """Create all GUI widgets"""
        self.create_file_upload_widgets()
        self.create_time_control_widgets()
        self.create_peak_detection_widgets()
        self.create_peak_model_widgets()
        self.create_wavelength_range_widgets()
        self.create_background_handling_widgets()
        self.create_fitting_action_widgets()
        self.create_batch_fitting_widgets()
        self.create_export_widgets()
        self.create_output_widgets()
        self.create_unit_conversion_widgets()  # Add this line
        
        debug_print("Created all GUI widgets", "GUI")
        return self.widgets
    
    def create_peak_row_widget(self, peak_idx, peak_info):
        """
        Create a widget row for a single peak/model
        
        Parameters:
        -----------
        peak_idx : int
            Peak index
        peak_info : dict
            Peak information (center, height, sigma, type)
            
        Returns:
        --------
        widgets.VBox: Peak row widget
        """
        # Peak type dropdown - includes Linear and Polynomial
        peak_type = widgets.Dropdown(
            options=['Gaussian', 'Voigt', 'Lorentzian', 'Linear', 'Polynomial'],
            value=peak_info.get('type', config.DEFAULT_PEAK_MODEL),
            description='Type:',
            style={'description_width': '40px'},
            layout=widgets.Layout(width='140px')
        )
        
        # Center position (hidden for Linear/Polynomial)
        center_input = widgets.FloatText(
            value=peak_info.get('center', 700),
            description='Center:',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='140px')
        )
        
        # Height/Amplitude (changes meaning based on type)
        height_input = widgets.FloatText(
            value=peak_info.get('height', 1000),
            description='Height:',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='140px')
        )
        
        # Sigma (width) - hidden for Linear/Polynomial
        sigma_input = widgets.FloatText(
            value=peak_info.get('sigma', 10),
            description='Sigma:',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='140px')
        )
        
        # Linear-specific: Slope
        slope_input = widgets.FloatText(
            value=peak_info.get('slope', 0.0),
            description='Slope:',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='140px', visibility='hidden')
        )
        
        # Linear-specific: Intercept
        intercept_input = widgets.FloatText(
            value=peak_info.get('intercept', 0.0),
            description='Y-inter:',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='140px', visibility='hidden')
        )
        
        # Polynomial-specific: Degree
        poly_degree_input = widgets.IntText(
            value=peak_info.get('poly_degree', 2),
            description='Degree:',
            min=1,
            max=5,
            style={'description_width': '50px'},
            layout=widgets.Layout(width='140px', visibility='hidden')
        )
        
        # Remove button
        remove_btn = widgets.Button(
            description='❌',
            button_style='danger',
            tooltip=f'Remove model {peak_idx}',
            layout=widgets.Layout(width='40px')
        )
        
        # Callback to show/hide widgets based on peak type
        def on_peak_type_change(change):
            peak_type_val = change['new']
            debug_print(f"Peak {peak_idx} type changed to {peak_type_val}", "GUI")
            
            if peak_type_val == 'Linear':
                # Show: slope, intercept
                # Hide: center, height, sigma, poly_degree
                center_input.layout.visibility = 'hidden'
                height_input.layout.visibility = 'hidden'
                sigma_input.layout.visibility = 'hidden'
                slope_input.layout.visibility = 'visible'
                intercept_input.layout.visibility = 'visible'
                poly_degree_input.layout.visibility = 'hidden'
                
            elif peak_type_val == 'Polynomial':
                # Show: poly_degree
                # Hide: center, height, sigma, slope, intercept
                center_input.layout.visibility = 'hidden'
                height_input.layout.visibility = 'hidden'
                sigma_input.layout.visibility = 'hidden'
                slope_input.layout.visibility = 'hidden'
                intercept_input.layout.visibility = 'hidden'
                poly_degree_input.layout.visibility = 'visible'
                
            else:  # Gaussian, Voigt, Lorentzian
                # Show: center, height, sigma
                # Hide: slope, intercept, poly_degree
                center_input.layout.visibility = 'visible'
                height_input.layout.visibility = 'visible'
                sigma_input.layout.visibility = 'visible'
                slope_input.layout.visibility = 'hidden'
                intercept_input.layout.visibility = 'hidden'
                poly_degree_input.layout.visibility = 'hidden'
        
        peak_type.observe(on_peak_type_change, names='value')
        
        # Trigger initial update
        on_peak_type_change({'new': peak_type.value})
        
        # Store references in the widgets for later access
        peak_type._peak_idx = peak_idx
        center_input._peak_idx = peak_idx
        center_input._param_name = 'center'
        height_input._peak_idx = peak_idx
        height_input._param_name = 'height'
        sigma_input._peak_idx = peak_idx
        sigma_input._param_name = 'sigma'
        slope_input._peak_idx = peak_idx
        slope_input._param_name = 'slope'
        intercept_input._peak_idx = peak_idx
        intercept_input._param_name = 'intercept'
        poly_degree_input._peak_idx = peak_idx
        poly_degree_input._param_name = 'poly_degree'
        remove_btn._peak_idx = peak_idx
        
        # Create layout - all widgets in a row, visibility controlled by callbacks
        row = widgets.HBox([
            widgets.HTML(value=f"<b>Model {peak_idx}:</b>", layout=widgets.Layout(width='70px')),
            peak_type,
            center_input,
            height_input,
            sigma_input,
            slope_input,
            intercept_input,
            poly_degree_input,
            remove_btn
        ])
        
        peak_row = widgets.VBox([row])
        peak_row._peak_idx = peak_idx
        peak_row._widgets = {
            'type': peak_type,
            'center': center_input,
            'height': height_input,
            'sigma': sigma_input,
            'slope': slope_input,
            'intercept': intercept_input,
            'poly_degree': poly_degree_input,
            'remove': remove_btn
        }
        
        return peak_row
    
    def create_collapsible_section(self, title, children):
        """
        Create a collapsible accordion section
        
        Parameters:
        -----------
        title : str
            Section title
        children : list
            List of child widgets
            
        Returns:
        --------
        widgets.Accordion: Collapsible section
        """
        content = widgets.VBox(children)
        accordion = widgets.Accordion(children=[content])
        accordion.set_title(0, title)
        accordion.selected_index = 0  # Start expanded
        
        return accordion
    
    def get_widget(self, name):
        """Get a widget by name"""
        return self.widgets.get(name, None)
    
    def get_all_widgets(self):
        """Get all widgets dictionary"""
        return self.widgets

    def create_unit_conversion_widgets(self):
        """Create unit conversion widgets"""
        # Toggle button for wavelength unit
        self.widgets['toggle_angstrom_btn'] = widgets.Button(
            description='🔄 Toggle Å/nm',
            button_style='warning',
            tooltip='Toggle wavelength unit between Angstrom and nanometers',
            layout=widgets.Layout(width='150px')
        )
        
        # Unit display label
        self.widgets['unit_display'] = widgets.Label(value="Unit: nm")
        
        debug_print("Created unit conversion widgets", "GUI")
        return self.widgets
