"""
GUI Components Module
Contains all UI components for the JV Analysis Dashboard.
"""

__author__ = "Edgar Nandayapa"
__institution__ = "Helmholtz-Zentrum Berlin"
__created__ = "August 2025"

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML, Markdown
import base64
import io
import zipfile
import plotly.graph_objects as go
import requests
import json
import plotly.express as px

class WidgetFactory:
    @staticmethod
    def create_button(description, button_style='', tooltip='', icon='', min_width=True):
        layout = widgets.Layout(min_width='150px') if min_width else widgets.Layout(width='auto')
        return widgets.Button(
            description=description, 
            button_style=button_style, 
            tooltip=tooltip, 
            icon=icon, 
            layout=layout
        )
    
    @staticmethod
    def create_dropdown(options, description='', width='standard', value=None):
        dropdown = widgets.Dropdown(options=options, description=description)
        if value is not None:
            dropdown.value = value
        return dropdown
    
    @staticmethod
    def create_text_input(placeholder='', description='', width='standard', password=False):
        widget_class = widgets.Password if password else widgets.Text
        return widget_class(
            placeholder=placeholder, 
            description=description, 
            style={'description_width': 'initial'}
        )
    
    @staticmethod
    def create_output(min_height='standard', scrollable=False, border=True):
        layout_props = {}
        if scrollable:
            layout_props.update({'width': '400px', 'height': '300px', 'overflow': 'scroll'})
        if border:
            layout_props.update({'border': '1px solid #eee', 'padding': '10px', 'margin': '10px 0 0 0'})
        return widgets.Output(layout=widgets.Layout(**layout_props))
    
    @staticmethod
    def create_radio_buttons(options, description='', value=None, width='standard'):
        radio = widgets.RadioButtons(options=options, description=description)
        if value is not None:
            radio.value = value
        return radio
    
    @staticmethod
    def create_filter_row():
        dropdown1 = widgets.Dropdown(
            options=['Voc(V)', 'Jsc(mA/cm2)', 'FF(%)', 'PCE(%)', 'V_MPP(V)', 'J_MPP(mA/cm2)'],
            layout=widgets.Layout(width='66%')
        )
        dropdown2 = widgets.Dropdown(
            options=['>', '>=', '<', '<=', '==', '!='],
            layout=widgets.Layout(width='33%')
        )
        text_input = widgets.Text(
            placeholder='Write a value',
            layout=widgets.Layout(width='33%')
        )
        return widgets.HBox([dropdown1, dropdown2, text_input])


class AuthenticationUI:
    """Handles all authentication-related UI components"""
    
    def __init__(self, auth_manager):
        self.auth_manager = auth_manager
        self.auth_manager.set_status_callback(self._update_status)
        self._create_widgets()
        self._setup_observers()
    
    def _create_widgets(self):
        """Create all authentication widgets"""
        self.auth_method_selector = widgets.RadioButtons(
            options=['Username/Password', 'Token (from ENV)'],
            description='Auth Method:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(margin='10px 0 0 0')
        )
        
        self.username_input = WidgetFactory.create_text_input(
            placeholder='Enter Username (e.g., email)',
            description='Username:'
        )
        
        self.password_input = WidgetFactory.create_text_input(
            placeholder='Enter Password',
            description='Password:',
            password=True
        )
        
        self.token_input = WidgetFactory.create_text_input(
            placeholder='Token will be read from ENV',
            description='Token:',
            width='wide',
            password=True
        )
        self.token_input.disabled = True
        
        self.auth_button = WidgetFactory.create_button(
            description='Authenticate',
            button_style='info',
            tooltip='Authenticate using the selected method'
        )
        
        self.auth_status_label = widgets.Label(
            value='Status: Not Authenticated',
            layout=widgets.Layout(margin='5px 0 0 0')
        )
        
        # Container widgets
        self.local_auth_box = widgets.VBox([self.username_input, self.password_input])
        self.token_auth_box = widgets.VBox([self.token_input])
        self.token_auth_box.layout.display = 'none'
        
        self.auth_action_box = widgets.VBox([self.auth_button, self.auth_status_label])
        
        # Settings toggle
        self.settings_toggle_button = WidgetFactory.create_button(
            description='▼ Connection Settings',
            min_width=False
        )
        self.settings_toggle_button.layout.width = '200px'
        
        self.settings_content = widgets.VBox([
            widgets.HTML("<p><strong>SE Oasis:</strong> https://nomad-hzb-se.de/nomad-oasis/api/v1</p>"),
            self.auth_method_selector,
            self.local_auth_box,
            self.token_auth_box,
            self.auth_action_box
        ], layout=widgets.Layout(padding='10px', margin='0 0 10px 0'))
        
        self.settings_box = widgets.VBox([
            self.settings_toggle_button,
            self.settings_content
        ], layout=widgets.Layout(border='1px solid #ccc', padding='10px', margin='0 0 20px 0'))
    
    def _setup_observers(self):
        """Setup event observers"""
        self.auth_method_selector.observe(self._on_auth_method_change, names='value')
        self.auth_button.on_click(self._on_auth_button_clicked)
        self.settings_toggle_button.on_click(self._toggle_settings)
    
    def _on_auth_method_change(self, change):
        """Handle authentication method change"""
        if change['new'] == 'Username/Password':
            self.local_auth_box.layout.display = 'flex'
            self.token_auth_box.layout.display = 'none'
        else:
            self.local_auth_box.layout.display = 'none'
            self.token_auth_box.layout.display = 'none'
        
        self.auth_status_label.value = 'Status: Not Authenticated (Method changed)'
        self.auth_manager.clear_authentication()
    
    def _on_auth_button_clicked(self, b):
        """Handle authentication button click"""
        self._update_status('Status: Authenticating...', 'orange')
        
        try:
            if self.auth_method_selector.value == 'Username/Password':
                token = self.auth_manager.authenticate_with_credentials(
                    self.username_input.value, 
                    self.password_input.value
                )
                self.password_input.value = ''
            else:
                token = self.auth_manager.authenticate_with_token()
            
            # Verify token
            user_info = self.auth_manager.verify_token()
            user_display = user_info.get('name', user_info.get('username', 'Unknown User'))
            self._update_status(f'Status: Authenticated as {user_display} on SE Oasis.', 'green')
            
            # Trigger success callback if set
            if hasattr(self, 'success_callback') and self.success_callback:
                self.success_callback()
                
        except Exception as e:
            # Handle authentication errors directly
            if isinstance(e, ValueError):
                self._update_status(f'Status: Error - {e}', 'red')
            elif isinstance(e, requests.exceptions.RequestException):
                error_message = f"Network/API Error: {e}"
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json().get('detail', e.response.text)
                        if isinstance(error_detail, list):
                            error_message = f"API Error ({e.response.status_code}): {json.dumps(error_detail)}"
                        else:
                            error_message = f"API Error ({e.response.status_code}): {error_detail or e.response.text}"
                    except:
                        error_message = f"API Error ({e.response.status_code}): {e.response.text}"
                self._update_status(f'Status: {error_message}', 'red')
            else:
                self._update_status(f'Status: Unexpected Error - {e}', 'red')
            
            self.auth_manager.clear_authentication()
    
    def _update_status(self, message, color=None):
        """Update status label with message and optional color"""
        self.auth_status_label.value = message
        if color:
            self.auth_status_label.style.text_color = color
        else:
            self.auth_status_label.style.text_color = None
    
    def _toggle_settings(self, b):
        """Toggle settings visibility"""
        if self.settings_content.layout.display == 'none':
            self.settings_content.layout.display = 'flex'
            self.settings_toggle_button.description = '▼ Connection Settings'
        else:
            self.settings_content.layout.display = 'none'
            self.settings_toggle_button.description = '▶ Connection Settings'
    
    def close_settings(self):
        """Close settings panel"""
        self.settings_content.layout.display = 'none'
        self.settings_toggle_button.description = '▶ Connection Settings'
    
    def set_success_callback(self, callback):
        """Set callback to execute on successful authentication"""
        self.success_callback = callback
    
    def get_widget(self):
        """Get the main settings widget"""
        return self.settings_box


class FilterUI:
    """Handles filter-related UI components with sample-based condition selection"""
    
    def __init__(self):
        self.filter_presets = {
            "Default": [("PCE(%)", "<", "40"), ("FF(%)", "<", "89"), ("FF(%)", ">", "24"), 
                       ("Voc(V)", "<", "2"), ("Jsc(mA/cm2)", ">", "-30")],
            "Preset 2": [("FF(%)", "<", "15"), ("PCE(%)", ">=", "10")]
        }
        self._create_widgets()
        self._setup_observers()
        self._apply_preset()  # Initialize with default preset
    
    def _create_widgets(self):
        """Create filter widgets"""
        self.preset_dropdown = WidgetFactory.create_dropdown(
            options=list(self.filter_presets.keys()),
            description='Filters'
        )
        self.preset_dropdown.layout.width = 'fit-content'
        self.preset_dropdown.layout.align_self = 'flex-end'
        
        self.direction_radio = WidgetFactory.create_radio_buttons(
            options=['Both', 'Reverse', 'Forward'],
            value='Both',
            description='Direction:'
        )
        
        self.add_button = WidgetFactory.create_button("Add Filter", 'primary')
        self.remove_button = WidgetFactory.create_button("Remove Filter", 'danger')
        self.apply_preset_button = WidgetFactory.create_button("Load Preset", 'info')
        self.apply_filter_button = WidgetFactory.create_button("Apply Filter", 'success')
        
        self.confirmation_output = WidgetFactory.create_output()
        self.main_output = WidgetFactory.create_output(scrollable=True)
        
        # Create initial filter row
        self.widget_groups = [WidgetFactory.create_filter_row()]
        self.groups_container = widgets.VBox(self.widget_groups)
        
        # Sample-based condition selection
        self.condition_toggle_button = widgets.Button(
            description='▼ Sample Selection',
            button_style='',
            layout=widgets.Layout(width='250px'),
            style={'font_weight': 'bold'}
        )
        
        self.condition_selection_content = widgets.Output(
            layout=widgets.Layout(
                display='flex',
                width='100%',
                overflow='visible'
            )
        )
        
        # Store data and selections for sample-based approach
        self.sample_data = None
        self.selected_samples = set()  # Set of selected "batch_sample" keys
        self.sample_checkboxes = {}  # Dict of sample checkboxes
        
        # Status widgets
        self.condition_status_output = widgets.Output()
        
        self.condition_selection_box = widgets.VBox([
            self.condition_toggle_button,
            self.condition_selection_content
        ], layout=widgets.Layout(border='1px solid #ddd', padding='10px', margin='5px 0'))
        
        # Layout components
        self.direction_container = widgets.VBox([
            widgets.HTML("<b>Filter by Cell Direction:</b>"), 
            self.direction_radio
        ])
        
        self.filter_conditions_container = widgets.VBox([
            widgets.HTML("<b>Filter Conditions:</b>"), 
            self.groups_container
        ])
        
        self.controls = widgets.VBox([
            self.add_button, self.remove_button, 
            self.preset_dropdown, self.apply_preset_button,
            widgets.HBox([self.apply_filter_button])
        ], layout=widgets.Layout(width='200px'))
        
        self.top_section = widgets.HBox([
            self.controls,
            widgets.VBox([self.direction_container, self.filter_conditions_container]),
            self.main_output
        ])
        
        self.layout = widgets.VBox([
            self.top_section,
            self.condition_selection_box
        ])

    def _setup_observers(self):
        """Setup event observers"""
        self.add_button.on_click(self._add_filter_row)
        self.remove_button.on_click(self._remove_filter_row)
        self.apply_preset_button.on_click(self._apply_preset)
        self.condition_toggle_button.on_click(self._toggle_condition_selection)

    def _add_filter_row(self, b):
        """Add a new filter row"""
        self.widget_groups.append(WidgetFactory.create_filter_row())
        self.groups_container.children = self.widget_groups

    def _remove_filter_row(self, b):
        """Remove the last filter row"""
        if len(self.widget_groups) > 1:
            self.widget_groups.pop()
            self.groups_container.children = self.widget_groups

    def _apply_preset(self, b=None):
        """Apply selected preset"""
        selected_preset = self.preset_dropdown.value
        self.widget_groups.clear()
        
        if selected_preset in self.filter_presets:
            for variable, operator, value in self.filter_presets[selected_preset]:
                group = WidgetFactory.create_filter_row()
                group.children[0].value = variable
                group.children[1].value = operator
                group.children[2].value = value
                self.widget_groups.append(group)
        else:
            self.widget_groups.append(WidgetFactory.create_filter_row())
        
        self.groups_container.children = self.widget_groups

    def _toggle_condition_selection(self, b):
        """Toggle sample selection visibility"""
        if self.condition_selection_content.layout.display == 'none':
            self.condition_selection_content.layout.display = 'flex'
            self.condition_toggle_button.description = '▼ Sample Selection'
        else:
            self.condition_selection_content.layout.display = 'none'
            self.condition_toggle_button.description = '▶ Sample Selection'

    def set_sample_data(self, data):
        """Set the data and create the sample selector"""
        self.sample_data = data
        if data is not None and 'jvc' in data:
            self._create_condition_selector()

    def _create_condition_selector(self):
        """Create sample-based selector interface grouped by batch"""
        with self.condition_selection_content:
            clear_output(wait=True)
            
            if not self.sample_data or 'jvc' not in self.sample_data:
                print("No data available for sample selection")
                return
            
            df = self.sample_data['jvc']
            
            # Title
            display(widgets.HTML("<h4>Select Samples to Include in Analysis:</h4>"))
            
            # Group by batch and sample, then get condition and counts
            batch_sample_info = df.groupby(['batch', 'sample', 'condition']).agg({
                'cell': 'nunique',  # unique cells per sample
                'sample': 'size'    # total measurements per sample
            }).rename(columns={'cell': 'num_cells', 'sample': 'num_measurements'})
            
            print(f"Dataset Overview:")
            total_samples = len(batch_sample_info)
            total_cells = batch_sample_info['num_cells'].sum()
            total_measurements = batch_sample_info['num_measurements'].sum()
            print(f"   • {total_samples} samples")
            print(f"   • {total_cells} cells")
            print(f"   • {total_measurements} measurements")
            
            # Quick selection buttons
            clear_all_button = widgets.Button(
                description="Clear All",
                button_style='warning',
                layout=widgets.Layout(width='100px')
            )
            
            select_all_button = widgets.Button(
                description="Select All",
                button_style='info',
                layout=widgets.Layout(width='100px')
            )
            
            # Button handlers for sample-based selection
            def clear_all_samples(b):
                """Clear all sample selections"""
                self.selected_samples.clear()
                self._update_sample_display()
                self._update_sample_status()
            
            def select_all_samples(b):
                """Select all available samples"""
                self.selected_samples = set()
                selected_count = 0
                for index, info in batch_sample_info.iterrows():
                    batch, sample, condition = index
                    sample_key = f"{batch}_{sample}"
                    self.selected_samples.add(sample_key)
                    selected_count += 1
                
                self._update_sample_display()
                self._update_sample_status()
            
            clear_all_button.on_click(clear_all_samples)
            select_all_button.on_click(select_all_samples)
            
            button_row = widgets.HBox([clear_all_button, select_all_button])
            display(button_row)
            
            # Create sample checkboxes grouped by batch
            self.sample_checkboxes = {}
            
            # Group data by batch for display
            batches = df['batch'].unique()
            batch_widgets = []
            
            for batch in sorted(batches):
                batch_df = df[df['batch'] == batch]
                batch_sample_info_for_display = batch_df.groupby(['sample', 'condition']).agg({
                    'cell': 'nunique',
                    'sample': 'size'
                }).rename(columns={'cell': 'num_cells', 'sample': 'num_measurements'})
                
                # Get display batch name if available
                display_batch = batch
                if 'display_batch' in batch_df.columns:
                    display_batch = batch_df['display_batch'].iloc[0]
                
                # Create batch header
                batch_header = widgets.HTML(f"<h5>📁 Batch: {display_batch}</h5>")
                
                # Create sample checkboxes for this batch
                sample_widgets = []
                for index, info in batch_sample_info_for_display.iterrows():
                    sample, condition = index
                    num_cells = info['num_cells']
                    num_measurements = info['num_measurements']
                    
                    checkbox_label = f"{sample} ({condition}) - {num_cells} cells, {num_measurements} measurements"
                    
                    checkbox = widgets.Checkbox(
                        value=True,
                        description=checkbox_label,
                        style={'description_width': 'initial'},
                        layout=widgets.Layout(
                            margin='2px 0 2px 20px',
                            width='auto'
                        )
                    )
                    
                    sample_key = f"{batch}_{sample}"
                    
                    # Handler for sample selection
                    def create_sample_checkbox_handler(sample_key):
                        def handler(change):
                            if change['new']:
                                self.selected_samples.add(sample_key)
                            else:
                                self.selected_samples.discard(sample_key)
                            self._update_sample_status()
                        return handler
                    
                    checkbox.observe(create_sample_checkbox_handler(sample_key), names='value')
                    self.sample_checkboxes[sample_key] = checkbox
                    sample_widgets.append(checkbox)
                
                # Add batch section
                batch_section = widgets.VBox([batch_header] + sample_widgets)
                batch_widgets.append(batch_section)
            
            # Display all batches
            all_batches_display = widgets.VBox(
                batch_widgets, 
                layout=widgets.Layout(
                    width='100%',
                    overflow='visible'
                )
            )
            display(all_batches_display)
            
            # Status display
            display(widgets.HTML("<h4>Selection Status:</h4>"))
            display(self.condition_status_output)
            
            # Initialize with all samples selected
            self.selected_samples = set()
            for index, info in batch_sample_info.iterrows():
                batch, sample, condition = index
                sample_key = f"{batch}_{sample}"
                self.selected_samples.add(sample_key)
            self._update_sample_status()

    def _update_sample_display(self):
        """Update checkbox display to match selected_samples set"""
        for sample_key, checkbox in self.sample_checkboxes.items():
            checkbox.value = sample_key in self.selected_samples

    def _update_sample_status(self):
        """Update status display for sample-based selection"""
        with self.condition_status_output:
            clear_output(wait=True)
            
            if not self.selected_samples:
                print("No samples selected")
                return
            
            df = self.sample_data['jvc']
            
            # Calculate statistics for selected samples
            selected_conditions = {}
            total_measurements = 0
            total_cells = 0
            
            for sample_key in self.selected_samples:
                batch, sample = sample_key.split('_', 1)
                
                sample_df = df[(df['batch'] == batch) & (df['sample'] == sample)]
                if not sample_df.empty:
                    condition = sample_df['condition'].iloc[0]
                    num_cells = sample_df['cell'].nunique()
                    num_measurements = len(sample_df)
                    
                    if condition not in selected_conditions:
                        selected_conditions[condition] = {
                            'samples': 0,
                            'cells': 0,
                            'measurements': 0
                        }
                    
                    selected_conditions[condition]['samples'] += 1
                    selected_conditions[condition]['cells'] += num_cells
                    selected_conditions[condition]['measurements'] += num_measurements
                    
                    total_cells += num_cells
                    total_measurements += num_measurements
            
            print(f"📋 Selected {len(self.selected_samples)} samples:")
            print(f"   📊 Total: {total_cells} cells, {total_measurements} measurements")
            print()
            
            for condition, stats in sorted(selected_conditions.items()):
                samples = stats['samples']
                cells = stats['cells']
                measurements = stats['measurements']
                print(f"   • {condition}: {samples} samples, {cells} cells, {measurements} measurements")
            
            # Check if only expected conditions
            expected = {'BL Printing', 'Slot_SAM', 'Spin_SAM'}
            selected_condition_names = set(selected_conditions.keys())
            
            if selected_condition_names == expected:
                print(f"\n🎯 Perfect! Only expected conditions selected.")
            elif selected_condition_names.issubset(expected):
                print(f"\n✅ Good! Only expected conditions (subset).")
            else:
                unexpected = selected_condition_names - expected
                if unexpected:
                    print(f"\n⚠️ Note: Additional conditions selected: {sorted(unexpected)}")

    def get_selected_items(self):
        """Get list of selected sample_cell combinations from sample selection"""
        if not hasattr(self, 'selected_samples') or not self.selected_samples:
            return None
        
        if not self.sample_data or 'jvc' not in self.sample_data:
            return None
        
        df = self.sample_data['jvc']
        selected_cell_combinations = []
                
        # Recreate the batch_sample_info mapping locally
        batch_sample_groups = df.groupby(['batch', 'sample', 'condition']).agg({
            'cell': 'nunique',
            'sample': 'size'  
        }).rename(columns={'cell': 'num_cells', 'sample': 'num_measurements'})
        
        # Create a mapping from sample_key to (batch, sample)
        sample_key_mapping = {}
        for index, info in batch_sample_groups.iterrows():
            batch, sample, condition = index
            sample_key = f"{batch}_{sample}"
            sample_key_mapping[sample_key] = (batch, sample)
        
        # Process each selected sample
        for sample_key in self.selected_samples:
            if sample_key in sample_key_mapping:
                batch, sample = sample_key_mapping[sample_key]
                
                # Get all cells for this sample
                sample_df = df[(df['batch'] == batch) & (df['sample'] == sample)]
                
                for _, row in sample_df.iterrows():
                    cell_key = f"{row['sample']}_{row['cell']}"
                    if cell_key not in selected_cell_combinations:
                        selected_cell_combinations.append(cell_key)
            else:
                print(f"Sample key '{sample_key}' not found in mapping")
        
        return selected_cell_combinations

    def get_filter_values(self):
        """Get current filter values"""
        filter_values = []
        for group in self.widget_groups:
            variable = group.children[0].value
            operator = group.children[1].value
            value = group.children[2].value
            filter_values.append((variable, operator, value))
        return filter_values

    def get_direction_value(self):
        """Get selected direction"""
        return self.direction_radio.value

    def set_apply_callback(self, callback):
        """Set callback for apply filter button"""
        self.apply_filter_button.on_click(callback)

    def get_widget(self):
        """Get the main filter widget"""
        return widgets.VBox([
            widgets.HTML("<h3>Select Filters</h3>"),
            widgets.HTML("<p>Using the dropdowns below, select filters for the data you want to keep, not remove.</p>"),
            self.layout
        ])


class PlotUI:
    """Handles plot selection UI components"""
    
    def __init__(self):
        self.plot_presets = {
            "Default": [("Boxplot", "PCE", "by Variable"), ("Boxplot", "Voc", "by Variable"), 
                       ("Boxplot", "Jsc", "by Variable"), ("Boxplot", "FF", "by Variable"), 
                       ("JV Curve", "Best device only", "")],
            "Preset 2": [("Boxplot", "Voc", "by Cell"), ("Histogram", "Voc", ""), 
                        ("JV Curve", "Best device only", "")],
            "Advanced Analysis": [("Boxplot", "PCE", "by Status"), ("Boxplot", "PCE", "by Status and Variable"),
                                 ("Boxplot", "PCE", "by Direction and Variable"), ("Boxplot", "PCE", "by Cell and Variable"),
                                 ("Boxplot", "PCE", "by Direction, Status and Variable")]
        }
        self._create_widgets()
        self._setup_observers()
        self._load_preset()  # Initialize with default preset
    
    def _create_widgets(self):
        """Create plot widgets"""
        self.preset_dropdown = WidgetFactory.create_dropdown(
            options=list(self.plot_presets.keys()),
            description='Presets',
        )
        self.preset_dropdown.layout = widgets.Layout(width='150px')
        
        self.add_button = WidgetFactory.create_button("Add Plot Type", 'primary')
        self.remove_button = WidgetFactory.create_button("Remove Plot Type", 'danger')
        self.load_preset_button = WidgetFactory.create_button("Load Preset", 'info')
        self.plot_button = WidgetFactory.create_button("Plot Selection", 'success')
        
        self.plotted_content = WidgetFactory.create_output()
        
        # Create initial plot type row
        self.plot_type_groups = [self._create_plot_type_row()]
        self.groups_container = widgets.VBox(self.plot_type_groups)
        
        self.controls = widgets.VBox([
            self.add_button, self.remove_button, 
            self.preset_dropdown, self.load_preset_button,
            self.plot_button
        ])
    
    def _create_plot_type_row(self):
        """Create a plot type selection row"""
        plot_type_dropdown = WidgetFactory.create_dropdown(
            options=['Boxplot', 'Boxplot (omitted)', 'Histogram', 'JV Curve'],
            description='Plot Type:',
            #width='wide'
            width='100px'
        )
        
        option1_dropdown = WidgetFactory.create_dropdown(
            options=[],
            description='Option 1:',
            #width='wide'
            width='100px'
        )
        
        option2_dropdown = WidgetFactory.create_dropdown(
            options=[],
            description='Option 2:',
            #width='wide'
            width='100px'
        )
        
        # Update options based on plot type
        self._update_plot_options(plot_type_dropdown, option1_dropdown, option2_dropdown)
        plot_type_dropdown.observe(
            lambda change: self._update_plot_options(plot_type_dropdown, option1_dropdown, option2_dropdown),
            names='value'
        )
        
        return widgets.HBox([plot_type_dropdown, option1_dropdown, option2_dropdown])
    
    def _update_plot_options(self, plot_type_dropdown, option1_dropdown, option2_dropdown):
        """Update option dropdowns based on plot type"""
        plot_type = plot_type_dropdown.value
        
        if plot_type == 'Boxplot':
            option1_dropdown.options = ['Voc', 'Jsc', 'FF', 'PCE', 'R_ser', 'R_shu', 'V_mpp', 'J_mpp', 'P_mpp']
            option2_dropdown.options = ['by Batch', 'by Variable', 'by Sample', 'by Cell', 'by Scan Direction',
                                       'by Status', 'by Status and Variable', 'by Direction and Variable', 'by Cell and Variable',
                                       'by Direction, Status and Variable']
        elif plot_type == 'Boxplot (omitted)':
            option1_dropdown.options = ['Voc', 'Jsc', 'FF', 'PCE', 'R_ser', 'R_shu', 'V_mpp', 'J_mpp', 'P_mpp']
            option2_dropdown.options = ['by Batch', 'by Variable', 'by Sample', 'by Cell', 'by Scan Direction', 'by Status']
        elif plot_type == 'Histogram':
            option1_dropdown.options = ['Voc', 'Jsc', 'FF', 'PCE', 'R_ser', 'R_shu', 'V_mpp', 'J_mpp', 'P_mpp']
            option2_dropdown.options = ['']
        elif plot_type == 'JV Curve':
            option1_dropdown.options = ['All cells', 'Only working cells', 'Rejected cells', 
                                       'Best device only', 'Separated by cell (all)', 'Separated by cell (working only)', 
                                       'Separated by substrate (all)', 'Separated by substrate (working only)']
            option2_dropdown.options = ['']
        else:
            option1_dropdown.options = []
            option2_dropdown.options = []
    
    def _setup_observers(self):
        """Setup event observers"""
        self.add_button.on_click(self._add_plot_type)
        self.remove_button.on_click(self._remove_plot_type)
        self.load_preset_button.on_click(self._load_preset)
    
    def _add_plot_type(self, b):
        """Add new plot type row"""
        self.plot_type_groups.append(self._create_plot_type_row())
        self.groups_container.children = tuple(self.plot_type_groups)
    
    def _remove_plot_type(self, b):
        """Remove last plot type row"""
        if len(self.plot_type_groups) > 1:
            self.plot_type_groups.pop()
            self.groups_container.children = tuple(self.plot_type_groups)
    
    def _load_preset(self, b=None):
        """Load selected preset"""
        selected_preset = self.preset_dropdown.value
        self.plot_type_groups.clear()
        
        if selected_preset in self.plot_presets:
            for plot_type, option1, option2 in self.plot_presets[selected_preset]:
                new_group = self._create_plot_type_row()
                new_group.children[0].value = plot_type
                new_group.children[1].value = option1
                new_group.children[2].value = option2
                self.plot_type_groups.append(new_group)
        else:
            self.plot_type_groups.append(self._create_plot_type_row())
        
        self.groups_container.children = tuple(self.plot_type_groups)
    
    def get_plot_selections(self):
        """Get current plot selections"""
        selections = []
        for group in self.plot_type_groups:
            plot_type = group.children[0].value
            option1 = group.children[1].value
            option2 = group.children[2].value
            selections.append((plot_type, option1, option2))
        return selections
    
    def set_plot_callback(self, callback):
        """Set callback for plot button"""
        self.plot_button.on_click(callback)
    
    def get_widget(self):
        """Get the main plot widget"""
        return widgets.VBox([
            widgets.HTML("<h3>Select Plots</h3>"),
            widgets.HTML("<p>Using the dropdowns below, select the plots you want to create.</p>"),
            widgets.HBox([self.controls, self.groups_container])
            # plotted_content moved to main app layout
        ])


class SaveUI:
    """Handles save functionality UI"""
    
    def __init__(self):
        self._create_widgets()
    
    def _create_widgets(self):
        """Create save widgets"""
        self.save_plots_button = WidgetFactory.create_button('Save All Plots', 'primary')
        self.save_data_button = WidgetFactory.create_button('Save Data', 'info')
        self.save_all_button = WidgetFactory.create_button('Save Data & Plots', 'success')
        self.download_output = WidgetFactory.create_output()
    
    def trigger_download(self, content, filename, content_type='text/json'):
        """Trigger file download"""
        content_b64 = base64.b64encode(content if isinstance(content, bytes) else content.encode()).decode()
        data_url = f'data:{content_type};charset=utf-8;base64,{content_b64}'
        js_code = f"""
            var a = document.createElement('a');
            a.setAttribute('download', '{filename}');
            a.setAttribute('href', '{data_url}');
            a.click()
        """
        with self.download_output:
            clear_output()
            display(HTML(f'<script>{js_code}</script>'))
    
    def create_plots_zip(self, figures, names):
        """Create zip file with plots"""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
            for fig, name in zip(figures, names):
                try:
                    html_str = fig.to_html(include_plotlyjs='cdn')
                    zip_file.writestr(name, html_str)
                    
                    # Try to save as PNG if possible
                    try:
                        import plotly.io as pio
                        img_bytes = pio.to_image(fig, format='png')
                        zip_file.writestr(name.replace('.html', '.png'), img_bytes)
                    except:
                        pass  # Skip PNG if not possible
                except Exception as e:
                    print(f"Error saving {name}: {e}")
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    def set_save_callbacks(self, plots_callback, data_callback, all_callback):
        """Set callbacks for save buttons"""
        self.save_plots_button.on_click(plots_callback)
        self.save_data_button.on_click(data_callback)
        self.save_all_button.on_click(all_callback)
    
    def get_widget(self):
        """Get the main save widget"""
        return widgets.VBox([
            widgets.HTML("<h3>Save Plots and Data</h3>"),
            widgets.HBox([self.save_plots_button, self.save_data_button, self.save_all_button]),
            self.download_output
        ])


class ColorSchemeSelector:
    """Color scheme selector with preview"""
    
    def __init__(self):
        self.color_schemes = {
            # Plotly sequential color schemes (only guaranteed ones)
            'Viridis': px.colors.sequential.Viridis,
            'Plasma': px.colors.sequential.Plasma,
            'Inferno': px.colors.sequential.Inferno,
            'Magma': px.colors.sequential.Magma,
            'Blues': px.colors.sequential.Blues,
            'Reds': px.colors.sequential.Reds,
            'Greens': px.colors.sequential.Greens,
            'Oranges': px.colors.sequential.Oranges,
            'Purples': px.colors.sequential.Purples,
            'BuGn': px.colors.sequential.BuGn,
            'YlOrRd': px.colors.sequential.YlOrRd,
            # Plotly qualitative color schemes (better for categorical data)
            'Plotly': px.colors.qualitative.Plotly,
            'D3': px.colors.qualitative.D3,
            'G10': px.colors.qualitative.G10,
            'T10': px.colors.qualitative.T10,
            'Set1': px.colors.qualitative.Set1,
            'Set2': px.colors.qualitative.Set2,
            'Set3': px.colors.qualitative.Set3,
            'Pastel1': px.colors.qualitative.Pastel1,
            'Pastel2': px.colors.qualitative.Pastel2,
            'Dark2': px.colors.qualitative.Dark2,
            # Custom schemes
            'Default (Current)': [
                'rgba(93, 164, 214, 0.7)', 'rgba(255, 144, 14, 0.7)', 
                'rgba(44, 160, 101, 0.7)', 'rgba(255, 65, 54, 0.7)', 
                'rgba(207, 114, 255, 0.7)', 'rgba(127, 96, 0, 0.7)',
                'rgba(255, 140, 184, 0.7)', 'rgba(79, 90, 117, 0.7)'
            ],
            'Scientific': [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ],
            'Nature': [
                '#228B22', '#32CD32', '#90EE90', '#006400', '#9ACD32',
                '#8FBC8F', '#7CFC00', '#ADFF2F', '#98FB98', '#00FF7F'
            ],
            'Ocean': [
                '#000080', '#0000CD', '#4169E1', '#1E90FF', '#00BFFF',
                '#87CEEB', '#87CEFA', '#ADD8E6', '#B0C4DE', '#F0F8FF'
            ],
            'Warm': [
                '#FF4500', '#FF6347', '#FF7F50', '#FFA500', '#FFB347',
                '#FFCCCB', '#FFE4B5', '#FFEFD5', '#FFF8DC', '#FFFACD'
            ]
        }
        
        # Add color schemes that might exist, but safely
        self._add_optional_color_schemes()
        
        self.selected_scheme = 'Default (Current)'
        self._create_widgets()
    
    def _add_optional_color_schemes(self):
        """Add color schemes that might not exist in all Plotly versions"""
        optional_schemes = {
            'Cividis': 'px.colors.sequential.Cividis',
            'RdBu': 'px.colors.sequential.RdBu',
            'Spectral': 'px.colors.sequential.Spectral',
            'Rainbow': 'px.colors.sequential.Rainbow',
            'Turbo': 'px.colors.sequential.Turbo',
            'Alphabet': 'px.colors.qualitative.Alphabet',
            'Sunsetdark': 'px.colors.sequential.Sunsetdark',
            'Peach': 'px.colors.sequential.Peach',
            'Mint': 'px.colors.sequential.Mint'
        }
        
        for name, attr_path in optional_schemes.items():
            try:
                # Try to access the color scheme
                color_scheme = eval(attr_path)
                self.color_schemes[name] = color_scheme
            except (AttributeError, NameError):
                # Skip if the color scheme doesn't exist
                continue
    
    def _create_widgets(self):
        """Create color scheme selector widgets"""
        self.color_dropdown = widgets.Dropdown(
            options=list(self.color_schemes.keys()),
            value=self.selected_scheme,
            description='Color Scheme:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )

        self.sampling_dropdown = widgets.Dropdown(
            options=['sequential', 'even'],
            value='sequential',
            description='Sampling:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        
        self.preview_output = widgets.Output(
            layout=widgets.Layout(width='400px', height='60px', border='1px solid #ccc')
        )
        
        self.color_dropdown.observe(self._on_color_change, names='value')
        self.sampling_dropdown.observe(self._on_sampling_change, names='value')
        
        # Initial preview
        self._update_preview()
        
        self.widget = widgets.HBox([
            self.color_dropdown,
            self.sampling_dropdown,
            self.preview_output
        ])

    def _on_sampling_change(self, change):
        """Handle sampling method change"""
        self._update_preview()
    
    def _on_color_change(self, change):
        """Handle color scheme change"""
        self.selected_scheme = change['new']
        self._update_preview()
    
    def _update_preview(self):
        """Update color preview"""
        with self.preview_output:
            clear_output(wait=True)
            
            colors = self.color_schemes[self.selected_scheme]
            
            # Show both sequential and even sampling for comparison
            if self.sampling_dropdown.value == 'even':
                preview_colors = self.get_colors(8, 'even')
                sampling_text = "Even Sampling"
            else:
                preview_colors = colors[:8] if len(colors) >= 8 else colors
                sampling_text = "Sequential"
            
            html_preview = '<div style="display: flex; align-items: center; padding: 5px;">'
            html_preview += f'<span style="margin-right: 10px; font-weight: bold;">{self.selected_scheme} ({sampling_text}):</span>'
            
            for color in preview_colors:
                html_preview += f'<span style="background-color: {color}; width: 30px; height: 30px; display: inline-block; margin: 2px; border: 1px solid #333; border-radius: 3px;"></span>'
            
            html_preview += '</div>'
            
            display(HTML(html_preview))
    
    def get_colors(self, num_colors=None, sampling='sequential'):
        """Get colors from selected scheme with improved sampling"""
        colors = self.color_schemes[self.selected_scheme]
        
        if num_colors is None:
            return colors
        
        if sampling == 'even' and len(colors) > num_colors:
            # Improved even sampling - distribute across the full spectrum
            if num_colors == 1:
                return [colors[len(colors)//2]]  # Take middle color for single color
            
            # Generate evenly spaced indices across the full color range
            indices = []
            for i in range(num_colors):
                # Map i from [0, num_colors-1] to [0, len(colors)-1]
                index = int(round(i * (len(colors) - 1) / (num_colors - 1)))
                indices.append(index)
            
            return [colors[i] for i in indices]
            
        elif num_colors <= len(colors):
            # Sequential sampling - take first n colors
            return colors[:num_colors]
        else:
            # Need more colors than available - cycle through the scheme
            repeated_colors = []
            for i in range(num_colors):
                repeated_colors.append(colors[i % len(colors)])
            return repeated_colors

    def get_widget(self):
        """Get the color scheme selector widget"""
        return widgets.VBox([
            widgets.HTML("<h4>Color Scheme Selection</h4>"),
            self.widget
        ])
    

class InfoUI:
    """What's New and Manual UI component using HTML files"""
    
    def __init__(self):
        self._create_widgets()
    
    def _create_widgets(self):
        """Create info widgets"""
        self.whats_new_button = widgets.Button(
            description='🎉 What\'s New',
            button_style='info',
            layout=widgets.Layout(width='140px', margin='0 5px 0 0'),
            tooltip='See the latest features and improvements'
        )
        
        self.manual_button = widgets.Button(
            description='📖 Manual',
            button_style='success',
            layout=widgets.Layout(width='120px', margin='0 5px 0 0'),
            tooltip='User manual and guide'
        )
        
        self.content_output = widgets.Output(
            layout=widgets.Layout(
                display='none',
                border='1px solid #ddd',
                border_radius='6px',
                padding='0px',
                margin='10px 0',
                max_height='500px',
                overflow_y='auto',
                background_color='white',
                width='100%'
            )
        )
        
        self.current_content = None  # Track what's currently displayed
        
        self.whats_new_button.on_click(self._show_whats_new)
        self.manual_button.on_click(self._show_manual)
        
        self.widget = widgets.VBox([
            widgets.HBox([self.whats_new_button, self.manual_button]),
            self.content_output
        ])
    
    def _load_html_file(self, filename):
        """Load HTML content from file and make it Voila-friendly"""
        try:
            import os
            # Get the directory where this Python file is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(current_dir, filename)
            
            # Fallback to current working directory if file not found
            if not os.path.exists(file_path):
                file_path = os.path.join(os.getcwd(), filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Extract only the body content and wrap it safely for Voila
            import re
            
            # Extract content between <body> tags
            body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.DOTALL)
            if body_match:
                body_content = body_match.group(1)
            else:
                # If no body tags, use the whole content but remove problematic elements
                body_content = html_content
            
            # Remove problematic CSS that breaks Voila layout
            body_content = re.sub(r'<style[^>]*>.*?</style>', '', body_content, flags=re.DOTALL)
            
            # Wrap in a safe container with Voila-friendly styling
            voila_safe_html = f"""
            <div style="
                max-width: 100%; 
                overflow-x: auto; 
                padding: 20px; 
                background-color: white; 
                border-radius: 8px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
            ">
                <style scoped>
                    /* Voila-safe CSS */
                    h1 {{ color: #2c3e50; font-size: 1.8em; margin-bottom: 10px; }}
                    h2 {{ color: #34495e; font-size: 1.3em; margin: 25px 0 10px 0; }}
                    h3 {{ color: #495057; font-size: 1.1em; margin: 15px 0 8px 0; }}
                    ul {{ margin: 0 0 15px 0; padding-left: 20px; }}
                    li {{ margin: 6px 0; line-height: 1.4; }}
                    .emoji {{ font-size: 1.1em; margin-right: 6px; }}
                    .highlight {{ background-color: #fff3cd; padding: 2px 4px; border-radius: 3px; }}
                    .sub-section {{ 
                        margin: 15px 0; 
                        padding: 15px; 
                        background-color: #f8f9fa; 
                        border-left: 3px solid #007bff; 
                        border-radius: 0 4px 4px 0; 
                    }}
                    .conclusion {{ 
                        margin-top: 30px; 
                        padding: 20px; 
                        background-color: #f8f9fa; 
                        border-radius: 6px; 
                        text-align: center; 
                        font-style: italic; 
                    }}
                    table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f8f9fa; }}
                    strong {{ color: #2c3e50; }}
                    .version-info {{ text-align: center; margin-bottom: 20px; color: #6c757d; }}
                    .header {{ text-align: center; margin-bottom: 20px; }}
                </style>
                {body_content}
            </div>
            """
            
            return voila_safe_html
            
        except FileNotFoundError:
            return f"""
            <div style="padding: 20px; text-align: center; color: #dc3545; max-width: 100%;">
                <h3>📄 File Not Found</h3>
                <p>Could not find <code>{filename}</code> in the current directory.</p>
            </div>
            """
        except Exception as e:
            return f"""
            <div style="padding: 20px; text-align: center; color: #dc3545; max-width: 100%;">
                <h3>❌ Error Loading Content</h3>
                <p>Error reading <code>{filename}</code>: {str(e)}</p>
            </div>
            """
    
    def _show_whats_new(self, b):
        """Show what's new content"""
        if self.current_content == 'whats_new' and self.content_output.layout.display == 'block':
            # Hide if already showing
            self.content_output.layout.display = 'none'
            self.whats_new_button.description = '🎉 What\'s New'
            self.whats_new_button.button_style = 'info'
            self.current_content = None
        else:
            # Show what's new
            self.content_output.layout.display = 'block'
            self.whats_new_button.description = '🔽 Hide What\'s New'
            self.whats_new_button.button_style = 'warning'
            self.manual_button.description = '📖 Manual'
            self.manual_button.button_style = 'success'
            self.current_content = 'whats_new'
            
            with self.content_output:
                clear_output(wait=True)
                html_content = self._load_html_file('whats_new.html')
                display(HTML(html_content))
    
    def _show_manual(self, b):
        """Show manual content"""
        if self.current_content == 'manual' and self.content_output.layout.display == 'block':
            # Hide if already showing
            self.content_output.layout.display = 'none'
            self.manual_button.description = '📖 Manual'
            self.manual_button.button_style = 'success'
            self.current_content = None
        else:
            # Show manual
            self.content_output.layout.display = 'block'
            self.manual_button.description = '🔽 Hide Manual'
            self.manual_button.button_style = 'warning'
            self.whats_new_button.description = '🎉 What\'s New'
            self.whats_new_button.button_style = 'info'
            self.current_content = 'manual'
            
            with self.content_output:
                clear_output(wait=True)
                html_content = self._load_html_file('manual.html')
                display(HTML(html_content))
    
    def get_widget(self):
        """Get the info widget"""
        return self.widget