"""
GUI Components for HySprint Data Analysis Tool
Creates and manages all user interface widgets
"""

import ipywidgets as widgets
import plotly.graph_objects as go
from batch_selection import create_batch_selection


class GUIManager:
    """Manages all GUI widgets for the application."""
    
    def __init__(self):
        """Initialize all widgets."""
        self._create_widgets()
    
    def _create_widgets(self):
        """Create all GUI widgets."""
        # ====================================================================
        # BATCH SELECTION WIDGETS
        # ====================================================================
        # Store batch selection container for later use
        self._batch_selection_container = None
        # These will be set when create_batch_selection is called
        self.search_box = None
        self.batch_selector = None
        self.load_batches_button = None
        
        # ====================================================================
        # DATA SOURCE SELECTORS
        # ====================================================================
        # Subbatch grouping checkbox
        self.group_by_subbatch = widgets.Checkbox(
            value=True,
            description='Group by subbatch (e.g., HZB_User_1_3 groups _C-1, _C-2, _C-3)',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='95%', margin='10px 0px')
        )
        
        self.x_data_source_selector = widgets.Dropdown(
            description='X Data Source:',
            style={'description_width': '120px'},
            layout={'width': '400px'},
            disabled=True
        )
        
        self.y_data_source_selector = widgets.Dropdown(
            description='Y Data Source:',
            style={'description_width': '120px'},
            layout={'width': '400px'},
            disabled=True
        )
        
        self.color_data_source_selector = widgets.Dropdown(
            description='Color Data Source:',
            style={'description_width': '120px'},
            layout={'width': '400px'},
            disabled=True
        )
        
        # ====================================================================
        # MATERIAL/LAYER SELECTORS
        # ====================================================================
        self.x_material_selector = widgets.Dropdown(
            description='X Material:',
            options=['All'],
            value='All',
            style={'description_width': '120px'},
            layout={'width': '400px'},
            disabled=True
        )
        
        self.y_material_selector = widgets.Dropdown(
            description='Y Material:',
            options=['All'],
            value='All',
            style={'description_width': '120px'},
            layout={'width': '400px'},
            disabled=True
        )
        
        self.color_material_selector = widgets.Dropdown(
            description='Color Material:',
            options=['All'],
            value='All',
            style={'description_width': '120px'},
            layout={'width': '400px'},
            disabled=True
        )
        
        # ====================================================================
        # PARAMETER SELECTORS
        # ====================================================================
        self.x_param_selector = widgets.Dropdown(
            description='X Parameter:',
            style={'description_width': '120px'},
            layout={'width': '400px'},
            disabled=True
        )
        
        self.y_param_selector = widgets.Dropdown(
            description='Y Parameter:',
            style={'description_width': '120px'},
            layout={'width': '400px'},
            disabled=True
        )
        
        self.color_param_selector = widgets.Dropdown(
            description='Color By:',
            style={'description_width': '120px'},
            layout={'width': '400px'},
            disabled=True
        )
        
        # ====================================================================
        # PLOT CONFIGURATION
        # ====================================================================
        self.jv_aggregation_selector = widgets.Dropdown(
            options=['All Points', 'Mean', 'Max', 'Min', 'Median'],
            value='Mean',
            description='JV Data Display:',
            style={'description_width': '120px'},
            layout={'width': '400px'}
        )

        self.plot_type_selector = widgets.Dropdown(
            options=['Scatter', 'Boxplot'],
            value='Scatter',
            description='Plot Type:',
            style={'description_width': '120px'},
            layout={'width': '400px'}
        )
        
        self.colorscale_selector = widgets.Dropdown(
            options=[
                # ── Colorscales ──
                'Viridis', 'Plasma', 'Inferno', 'Magma', 'Cividis',
                'Turbo', 'RdBu', 'Spectral', 'Blues', 'Reds', 'YlOrRd',
                # ── Single colors ──
                'Blue', 'Red', 'Green', 'Purple', 'Orange', 'Gray',
            ],
            value='Viridis',
            description='Color Scale:',
            style={'description_width': '120px'},
            layout={'width': '400px'}
        )
        
        self.show_varying_only = widgets.ToggleButton(
            value=False,
            description='Show Only Varying Parameters',
            tooltip='Filter to show only parameters that vary across samples'
        )
        
        self.create_plot_button = widgets.Button(
            description='Create Plot',
            button_style='success',
            icon='chart-line',
            layout={'width': '150px'}
        )
        
        self.debug_checkbox = widgets.Checkbox(
            value=False,
            description='Show plot data for debugging',
            style={'description_width': 'initial'}
        )

        self.download_button = widgets.Button(
            description='Download CSV',
            button_style='info',
            icon='download',
            layout={'width': '150px'}
        )
        
        self.download_output = widgets.Output()

        self.click_output = widgets.Output(
            layout={'border': '1px solid #ddd', 'padding': '4px',
                    'min_height': '30px', 'margin_top': '4px'}
        )
        
        # ====================================================================
        # OUTPUT WIDGETS
        # ====================================================================
        self.status_output = widgets.Output(
            layout={'border': '1px solid #ddd', 'padding': '10px', 'height': '150px'}
        )
        
        self.stats_output = widgets.Output(
            layout={'border': '1px solid #ddd', 'padding': '10px', 'height': '200px'}
        )
        
        self.param_summary_output = widgets.Output()
        self.param_summary_accordion = widgets.Accordion(
            children=[self.param_summary_output],
            titles=('Parameter Summary',),
            layout={'width': '100%'}
        )
        self.param_summary_accordion.selected_index = None  # Start collapsed
        
        # ====================================================================
        # PLOT WIDGET
        # ====================================================================
        self.plot_widget = go.FigureWidget()
        self.plot_widget.update_layout(
            height=600,
            template='plotly_white',
            title='Select data and click "Create Plot"'
        )
    
    def connect_callbacks(self, callbacks: dict):
        """
        Connect widget callbacks to handler functions.
        
        Args:
            callbacks: Dictionary mapping widget names to callback functions
        """
        # Batch selection
        if 'search_box' in callbacks:
            self.search_box.observe(callbacks['search_box'], names='value')
        if 'load_batches' in callbacks:
            self.load_batches_button.on_click(callbacks['load_batches'])
        
        # Data sources
        if 'x_data_source' in callbacks:
            self.x_data_source_selector.observe(callbacks['x_data_source'], names='value')
        if 'y_data_source' in callbacks:
            self.y_data_source_selector.observe(callbacks['y_data_source'], names='value')
        if 'color_data_source' in callbacks:
            self.color_data_source_selector.observe(callbacks['color_data_source'], names='value')
        
        # Materials
        if 'x_material' in callbacks:
            self.x_material_selector.observe(callbacks['x_material'], names='value')
        if 'y_material' in callbacks:
            self.y_material_selector.observe(callbacks['y_material'], names='value')
        if 'color_material' in callbacks:
            self.color_material_selector.observe(callbacks['color_material'], names='value')
        
        # Plot controls
        if 'create_plot' in callbacks:
            self.create_plot_button.on_click(callbacks['create_plot'])
        if 'toggle_varying' in callbacks:
            self.show_varying_only.observe(callbacks['toggle_varying'], names='value')
            
        if 'download' in callbacks:
            self.download_button.on_click(callbacks['download'])

    def setup_batch_selection(self, url, token, load_data_function):
        """Setup batch selection widget using batch_selection module."""
        self._batch_selection_container = create_batch_selection(url, token, load_data_function)
        
        # Extract individual widgets for compatibility
        self.search_box = self._batch_selection_container.children[0]
        self.batch_selector = self._batch_selection_container.children[1]
        self.load_batches_button = self._batch_selection_container.children[2]
    
    def create_layout(self) -> widgets.Widget:
        """Create the complete interface layout."""
        # Title
        title = widgets.HTML(
            "<h1 style='text-align: center; color: #2E86AB;'>HySprint Data Analysis Tool</h1>"
        )
        
        # Step 1: Batch Selection
        step1 = widgets.VBox([
            widgets.HTML("<h3 style='color: #A23B72;'>Step 1: Select Batches</h3>"),
            self._batch_selection_container
        ])
        
        # Step 2: Data Sources
        step2 = widgets.VBox([
            widgets.HTML("<h3 style='color: #A23B72;'>Step 2: Select Data Sources</h3>"),
            widgets.HTML("<h4 style='color: #666;'>X-Axis Data:</h4>"),
            self.group_by_subbatch,
            self.x_data_source_selector,
            self.x_material_selector,
            self.x_param_selector,
            widgets.HTML("<h4 style='color: #666;'>Y-Axis Data:</h4>"),
            self.y_data_source_selector,
            self.y_material_selector,
            self.y_param_selector,
            widgets.HTML("<h4 style='color: #666;'>Color Data:</h4>"),
            self.color_data_source_selector,
            self.color_material_selector,
            self.color_param_selector
        ])
        
        # Step 3: Plot Configuration
        step3 = widgets.VBox([
            widgets.HTML("<h3 style='color: #A23B72;'>Step 3: Configure Plot</h3>"),
            self.plot_type_selector,
            self.jv_aggregation_selector,
            self.colorscale_selector,
            self.show_varying_only,
            self.debug_checkbox,
            widgets.HBox([self.create_plot_button, self.download_button]),
            self.download_output
        ])
        
        # Left panel - controls
        left_panel = widgets.VBox(
            [step1, step2, step3],
            layout={'width': '500px', 'padding': '20px'}
        )
        
        # Right panel - visualization
        right_panel = widgets.VBox([
            widgets.HTML("<h3 style='color: #A23B72;'>Visualization</h3>"),
            self.plot_widget,
            self.click_output,
            widgets.HTML("<h3 style='color: #A23B72;'>Statistics</h3>"),
            self.stats_output
        ], layout={'width': '900px', 'padding': '20px'})
        
        # Top section - two columns
        top_section = widgets.HBox([left_panel, right_panel])
        
        # Bottom section - full width
        bottom_section = widgets.VBox([
            widgets.HTML("<h3 style='color: #A23B72;'>Parameter Summary</h3>"),
            self.param_summary_accordion,
            widgets.HTML("<h3 style='color: #A23B72;'>Status</h3>"),
            self.status_output
        ], layout={'width': '100%', 'padding': '20px'})
        
        # Main layout
        main_layout = widgets.VBox([top_section, bottom_section])
        
        return widgets.VBox([title, main_layout])
