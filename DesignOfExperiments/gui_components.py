"""
GUI Components for the Design of Experiments application.
Reusable UI components, widgets, forms, and buttons using ipywidgets.
"""

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Tuple
import json
import base64
import io

from data_manager import Variable, VariableType, DataManager
from utils import ValidationUtils, Constants

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


class GUIComponents:
    """Factory class for creating reusable GUI components."""
    
    def __init__(self):
        """Initialize GUI components."""
        self.validator = ValidationUtils()
        self.variable_widgets = []
        self.callbacks = {}
        self.current_samples = None
        self.current_variables = []
        self.current_metrics = {}
        self.current_algorithm = ""
        self.current_random_seed = 42
        
        # Styling constants
        self.BUTTON_STYLE = {'description_width': 'initial'}
        self.INPUT_STYLE = {'description_width': '120px'}
        self.WIDE_INPUT_STYLE = {'description_width': '150px'}

    def _parse_float(self, value_str):
        """Parse float value handling comma as decimal separator."""
        if isinstance(value_str, (int, float)):
            return float(value_str)
        
        try:
            # Replace comma with dot for decimal parsing
            clean_str = str(value_str).replace(',', '.')
            return float(clean_str)
        except (ValueError, TypeError):
            return 0.0

    def create_algorithm_selector(self) -> widgets.Widget:
        """Create algorithm selection interface."""
        # Algorithm dropdown with lab-focused descriptions
        algorithms = {
            "Latin Hypercube Sampling": "Ensures each experimental parameter is tested across its full range with minimal overlap. Ideal for screening studies and response surface methodology in materials research.",
            
            "Sobol Sequences": "Generates highly uniform experimental designs that efficiently explore parameter space. Particularly useful for computational experiments and sensitivity analysis of process variables.",
            
            "Halton Sequences": "Creates systematic experimental designs that avoid clustering of test conditions. Well-suited for optimization studies where you need consistent coverage of factor combinations.",
            
            "Random Sampling": "Simple random selection of experimental conditions. Best for baseline comparisons or when you need unbiased sampling without assumptions about parameter relationships.",
            
            "Uniform Grid Sampling": "Tests all combinations at regular intervals across parameter ranges. Perfect for fundamental studies mapping complete response surfaces with systematic precision.",
            
            "Maximin Distance Design": "Optimizes spacing between experimental points to maximize information gain. Excellent for expensive experiments where each test must provide maximum insight."
        }
        
        self.algorithm_dropdown = widgets.Dropdown(
            options=list(algorithms.keys()),
            value="Latin Hypercube Sampling",
            description="Algorithm:",
            style=self.INPUT_STYLE,
            layout=widgets.Layout(width='400px')
        )
        
        # Algorithm description
        self.algorithm_description = widgets.HTML(
            value=f"<i>{algorithms['Latin Hypercube Sampling']}</i>",
            layout=widgets.Layout(margin='5px 0')
        )
        
        # Update description when algorithm changes
        def update_description(change):
            self.algorithm_description.value = f"<i>{algorithms[change['new']]}</i>"
        
        self.algorithm_dropdown.observe(update_description, names='value')
        
        return widgets.VBox([
            self.algorithm_dropdown,
            self.algorithm_description
        ])

    def _on_export_protocol(self, button):
        """Handle export protocol button click."""
        if not hasattr(self, 'current_samples') or self.current_samples is None:
            self.protocol_display.value = "<p style='color: red;'>No samples to export. Generate samples first.</p>"
            return
        
        try:
            # Create experimental protocol
            protocol_content = self._generate_experimental_protocol()
            
            # Create download link
            download_link = self.create_download_link(
                protocol_content,
                "experimental_protocol.txt",
                "Experimental protocol document"
            )
            
            self.protocol_display.value = download_link
            
        except Exception as e:
            self.protocol_display.value = f"<p style='color: red;'>Protocol export failed: {str(e)}</p>"

    def _generate_experimental_protocol(self) -> str:
        """Generate experimental protocol document."""
        protocol_lines = []
        protocol_lines.append("EXPERIMENTAL PROTOCOL")
        protocol_lines.append("=" * 50)
        protocol_lines.append("")
        
        # Header information with algorithm
        protocol_lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        protocol_lines.append(f"Algorithm: {self.current_algorithm}")  # Add this line
        protocol_lines.append(f"Total Experiments: {len(self.current_samples)}")
        protocol_lines.append(f"Variables: {len(self.current_variables)}")
        protocol_lines.append("")
        
        # Variable definitions
        protocol_lines.append("VARIABLE DEFINITIONS:")
        protocol_lines.append("-" * 20)
        for var in self.current_variables:
            protocol_lines.append(f"• {var.name}: {var.type.value}")
            if var.type == VariableType.CONTINUOUS:
                protocol_lines.append(f"  Range: {var.min_value} - {var.max_value}")
            elif var.type == VariableType.DISCRETE:
                protocol_lines.append(f"  Range: {var.min_value} - {var.max_value} (step: {var.step_size})")
            elif var.type == VariableType.CATEGORICAL:
                protocol_lines.append(f"  Categories: {', '.join(var.categories)}")
            if var.description:
                protocol_lines.append(f"  Description: {var.description}")
            protocol_lines.append("")
        
        # Experimental conditions
        protocol_lines.append("EXPERIMENTAL CONDITIONS:")
        protocol_lines.append("-" * 25)
        
        # Get experimental order
        samples = self.current_samples.copy()
        if self.randomize_order.value:
            samples = samples.sample(frac=1, random_state=42).reset_index(drop=True)
            protocol_lines.append("Execution order: RANDOMIZED")
        else:
            protocol_lines.append("Execution order: SEQUENTIAL")
        
        protocol_lines.append("")
        
        # Sample details
        for idx, (_, row) in enumerate(samples.iterrows(), 1):
            protocol_lines.append(f"Experiment {idx}:")
            for var in self.current_variables:
                if var.name in row:
                    protocol_lines.append(f"  {var.name}: {row[var.name]}")
            protocol_lines.append("")
        
        # Footer
        protocol_lines.append("=" * 50)
        protocol_lines.append("End of Protocol")
        
        return "\n".join(protocol_lines)

    def _on_export_data(self, button):
        """Handle export data button click."""
        if not hasattr(self, 'current_samples') or self.current_samples is None:
            self.export_download_area.value = "<p style='color: red;'>No data to export. Generate samples first.</p>"
            return
        
        format_type = self.export_format_dropdown.value
        
        try:
            if format_type == 'csv':
                # Export samples as CSV
                csv_content = self.current_samples.to_csv(index=False)
                download_link = self.create_download_link(
                    csv_content, 
                    "doe_samples.csv", 
                    "Sample data exported as CSV"
                )
                
            elif format_type == 'json':
                # Export configuration as JSON
                config = {
                    'variables': [var.to_dict() for var in self.current_variables] if hasattr(self, 'current_variables') else [],
                    'algorithm': getattr(self, 'current_algorithm', 'Latin Hypercube Sampling'),
                    'n_samples': len(self.current_samples),
                    'random_seed': getattr(self, 'current_random_seed', 42),
                    'generated_timestamp': pd.Timestamp.now().isoformat(),
                    'samples': self.current_samples.to_dict('records')
                }
                json_content = json.dumps(config, indent=2)
                download_link = self.create_download_link(
                    json_content,
                    "doe_configuration.json",
                    "Complete configuration exported as JSON"
                )
                
            elif format_type == 'excel':
                # Check if openpyxl is available
                if not HAS_OPENPYXL:
                    self.export_download_area.value = "<p style='color: red;'>Excel export requires openpyxl. Install with: pip install openpyxl</p>"
                    return
                
                # Export as Excel with multiple sheets
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    # Samples sheet
                    self.current_samples.to_excel(writer, sheet_name='Samples', index=False)
                    
                    # Variables sheet
                    if hasattr(self, 'current_variables'):
                        var_data = []
                        for var in self.current_variables:
                            var_dict = var.to_dict()
                            var_data.append(var_dict)
                        var_df = pd.DataFrame(var_data)
                        var_df.to_excel(writer, sheet_name='Variables', index=False)
                    
                    # Metrics sheet
                    if hasattr(self, 'current_metrics'):
                        # Flatten metrics for Excel export
                        metrics_data = []
                        for key, value in self.current_metrics.items():
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    metrics_data.append({'Metric': f"{key}.{subkey}", 'Value': str(subvalue)})
                            else:
                                metrics_data.append({'Metric': key, 'Value': str(value)})
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                
                # Create download link for Excel
                excel_content = base64.b64encode(buffer.getvalue()).decode()
                download_link = f"""
                <p>Complete data package exported as Excel</p>
                <a download="doe_complete.xlsx" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{excel_content}" 
                   style="background-color: #28a745; color: white; padding: 10px 15px; 
                          text-decoration: none; border-radius: 4px; display: inline-block;">
                    📥 Download doe_complete.xlsx
                </a>
                """
            
            self.export_download_area.value = download_link
            
        except Exception as e:
            self.export_download_area.value = f"<p style='color: red;'>Export failed: {str(e)}</p>"

    def set_current_data(self, samples: pd.DataFrame, variables: List[Variable], 
                         metrics: Dict[str, Any], algorithm: str, random_seed: int):
        """Store current data for export functionality."""
        self.current_samples = samples
        self.current_variables = variables
        self.current_metrics = metrics
        self.current_algorithm = algorithm
        self.current_random_seed = random_seed
    
    def create_variable_configurator(self) -> widgets.Widget:
        """Create variable configuration interface."""
        # Variable list container
        self.variable_list = widgets.VBox()
        
        # Add variable button
        add_button = widgets.Button(
            description="Add Variable",
            button_style='success',
            icon='plus',
            layout=widgets.Layout(width='150px')
        )
        add_button.on_click(self._on_add_variable)
        
        # Clear all button
        clear_button = widgets.Button(
            description="Clear All",
            button_style='warning',
            icon='trash',
            layout=widgets.Layout(width='100px')
        )
        clear_button.on_click(self._on_clear_variables)
        
        # Variable counter
        self.variable_count_label = widgets.HTML(
            value="<b>Variables: 0</b>",
            layout=widgets.Layout(margin='5px 0')
        )
        
        # Add 2 empty variable widgets by default
        self.variable_widgets = []
        for i in range(2):
            new_widget = self._create_variable_widget()
            self.variable_widgets.append(new_widget)
        
        # Controls row
        controls = widgets.HBox([
            add_button,
            clear_button,
            widgets.HTML("&nbsp;" * 5),
            self.variable_count_label
        ])
        
        # Headers for variable inputs
        headers = widgets.HTML("""
            <div style='margin: 10px 0; font-weight: bold; color: #555;'>
                <span style='display: inline-block; width: 150px;'>Variable Name</span>
                <span style='display: inline-block; width: 120px;'>Type</span>
                <span style='display: inline-block; width: 200px;'>Parameters (Min, Max, Step/Categories)</span>
                <span style='display: inline-block; width: 150px;'>Description</span>
                <span style='display: inline-block; width: 40px;'></span>
            </div>
        """)
        
        # Initialize display
        self._update_variable_display()
        
        return widgets.VBox([
            controls,
            headers,
            widgets.HTML("<hr style='margin: 5px 0;'>"),
            self.variable_list
        ])
    
    def create_sample_size_configurator(self) -> widgets.Widget:
        """Create sample size configuration interface."""
        # Sample size slider
        self.sample_size_slider = widgets.IntSlider(
            value=25,
            min=4,
            max=100,
            step=1,
            description="Sample Size:",
            style=self.INPUT_STYLE,
            layout=widgets.Layout(width='400px')
        )
        
        # Minimum size label
        self.min_size_label = widgets.HTML(
            value="<i>Minimum recommended: 4 samples</i>",
            layout=widgets.Layout(margin='5px 0')
        )
        
        return widgets.VBox([
            self.sample_size_slider,
            self.min_size_label
        ])
    
    def create_seed_configurator(self) -> widgets.Widget:
        """Create random seed configuration interface."""
        self.random_seed = widgets.IntText(
            value=42,
            description="Random Seed:",
            style=self.INPUT_STYLE,
            layout=widgets.Layout(width='200px')
        )
        
        # Generate new seed button
        new_seed_button = widgets.Button(
            description="New Seed",
            button_style='',
            icon='refresh',
            layout=widgets.Layout(width='100px')
        )
        
        # Add explanation text
        seed_explanation = widgets.HTML(
            value="<i style='color: #666; font-size: 11px;'>Seed controls randomization - same seed produces identical results</i>",
            layout=widgets.Layout(margin='5px 0px 0px 10px')
        )
        
        def generate_new_seed(button):
            self.random_seed.value = np.random.randint(1, 10000)
        
        new_seed_button.on_click(generate_new_seed)
        
        return widgets.VBox([
            widgets.HBox([
                self.random_seed,
                new_seed_button
            ]),
            seed_explanation
        ])
    
    def create_advanced_options(self) -> widgets.Widget:
        """Create advanced options collapsible panel."""
        # Advanced options container
        self.advanced_options_container = widgets.VBox([
            widgets.HTML("<i>No advanced options available for this algorithm</i>")
        ])
        
        # Collapsible accordion
        self.advanced_accordion = widgets.Accordion()
        self.advanced_accordion.children = [self.advanced_options_container]
        self.advanced_accordion.set_title(0, 'Advanced Options')
        self.advanced_accordion.selected_index = None  # Initially collapsed
        
        return self.advanced_accordion
    
    def create_generation_controls(self) -> widgets.Widget:
        """Create sample generation control interface."""
        # Generate button
        self.generate_button = widgets.Button(
            description="Generate Samples",
            button_style='primary',
            icon='cogs',
            layout=widgets.Layout(width='200px', height='40px')
        )
        
        # Regenerate button
        self.regenerate_button = widgets.Button(
            description="Regenerate",
            button_style='',
            icon='refresh',
            layout=widgets.Layout(width='120px')
        )
        
        # Store reference to generation controls for event binding
        self._generation_controls = widgets.HBox([
            self.generate_button,
            widgets.HTML("&nbsp;" * 3),
            self.regenerate_button
        ])
        
        return self._generation_controls
    
    def create_progress_section(self) -> widgets.Widget:
        """Create progress indicator section."""
        self.progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            bar_style='info',
            layout=widgets.Layout(width='300px', visibility='hidden')
        )
        
        self.progress_label = widgets.HTML(
            layout=widgets.Layout(visibility='hidden')
        )
        
        return widgets.VBox([
            self.progress_bar,
            self.progress_label
        ])
    
    def create_metrics_section(self) -> widgets.Widget:
        """Create quality metrics display section."""
        self.metrics_display = widgets.HTML()
        return self.metrics_display
    
    def create_summary_section(self) -> widgets.Widget:
        """Create summary statistics section."""
        self.summary_display = widgets.HTML()
        return self.summary_display
    
    def create_protocol_section(self) -> widgets.Widget:
        """Create experimental protocol section."""
        # Randomization options
        self.randomize_order = widgets.Checkbox(
            value=True,
            description="Randomize experimental order",
            style=self.INPUT_STYLE
        )
        
        # Export protocol button
        export_protocol_button = widgets.Button(
            description="Export Protocol",
            button_style='info',
            icon='download',
            layout=widgets.Layout(width='150px')
        )
        
        # Connect button handler
        export_protocol_button.on_click(self._on_export_protocol)
        
        self.protocol_display = widgets.HTML()
        
        return widgets.VBox([
            self.randomize_order,
            export_protocol_button,
            self.protocol_display
        ])
    
    def create_visualization_controls(self) -> widgets.Widget:
        """Create visualization control interface."""
        # Plot type selector
        self.plot_type_dropdown = widgets.Dropdown(
            options=[
                ('Scatter Plot Matrix', 'splom'),
                ('Parallel Coordinates', 'parallel'),
                ('Distribution Plots', 'distributions'),
                ('3D Scatter Plot', '3d_scatter'),
                ('Correlation Heatmap', 'correlation'),
                ('Space-Filling Quality', 'quality')
            ],
            value='splom',
            description="Plot Type:",
            style=self.INPUT_STYLE
        )
        
        # Export plot button with explanatory text
        export_plot_button = widgets.Button(
            description="Export Plot",
            button_style='',
            icon='download',
            layout=widgets.Layout(width='120px')
        )
        
        export_text = widgets.HTML(
            value="<i style='color: #666; font-size: 11px;'>Download link appears below plot</i>",
            layout=widgets.Layout(margin='0px 0px 0px 10px')
        )
        
        # Store reference for connection
        self.export_plot_button = export_plot_button
        
        controls = widgets.VBox([
            self.plot_type_dropdown,
            widgets.HTML("<hr style='margin: 10px 0;'>"),
            widgets.HBox([
                export_plot_button,
                export_text
            ])
        ])
        
        return controls
    
    def create_export_section(self) -> widgets.Widget:
        """Create data export interface."""
        # Export format selector
        self.export_format_dropdown = widgets.Dropdown(
            options=[
                ('CSV (Samples)', 'csv'),
                ('JSON (Configuration)', 'json'),
                ('Excel (All Data)', 'excel')
            ],
            value='csv',
            description="Format:",
            style=self.INPUT_STYLE
        )
        
        # Export button
        export_button = widgets.Button(
            description="Export Data",
            button_style='success',
            icon='download',
            layout=widgets.Layout(width='150px')
        )
        
        # Connect the button click handler
        export_button.on_click(self._on_export_data)
        
        # Download area
        self.export_download_area = widgets.HTML()
        
        return widgets.VBox([
            self.export_format_dropdown,
            export_button,
            self.export_download_area
        ])
    
    def _create_variable_widget(self, variable: Optional[Variable] = None) -> widgets.Widget:
        """Create a single variable input widget."""
        # Variable name input
        name_input = widgets.Text(
            value=variable.name if variable else f"Variable_{len(self.variable_widgets)+1}",
            placeholder="Variable name",
            layout=widgets.Layout(width='150px')
        )
        
        # Variable type selector - default to discrete
        type_selector = widgets.Dropdown(
            options=[
                ('Discrete', 'discrete'),      # Move discrete to top as default
                ('Continuous', 'continuous'),
                ('Categorical', 'categorical')
            ],
            value=variable.type.value if variable else 'discrete',  # Default to discrete
            layout=widgets.Layout(width='120px')
        )
        
        # Parameter inputs using Text widgets to allow comma input
        min_input = widgets.Text(
            value=str(variable.min_value) if variable and variable.min_value else "1.0",
            placeholder="Min",
            layout=widgets.Layout(width='80px')
        )
        
        max_input = widgets.Text(
            value=str(variable.max_value) if variable and variable.max_value else "10.0",
            placeholder="Max",
            layout=widgets.Layout(width='80px')
        )
        
        step_input = widgets.Text(
            value=str(variable.step_size) if variable and variable.step_size else "1.0",
            placeholder="Step",
            layout=widgets.Layout(width='80px')
        )
        
        # Add validation handlers that trigger on Enter or when losing focus
        def create_validator(widget, default_val):
            def on_submit(sender):
                try:
                    # Parse using your existing method that handles commas
                    parsed_val = self._parse_float(widget.value)
                    widget.value = str(parsed_val)
                except:
                    widget.value = str(default_val)
            return on_submit
        
        # Connect validators to submit events (Enter key)
        min_input.on_submit(create_validator(min_input, 1.0))
        max_input.on_submit(create_validator(max_input, 10.0))
        step_input.on_submit(create_validator(step_input, 1.0))
        
        categories_input = widgets.Text(
            value=','.join(variable.categories) if variable and variable.categories else "A,B,C",
            placeholder="cat1,cat2,cat3",
            layout=widgets.Layout(width='200px')
        )
        
        # Description input
        desc_input = widgets.Text(
            value=variable.description if variable else "",
            placeholder="Description (optional)",
            layout=widgets.Layout(width='150px')
        )
        
        # Remove button
        remove_button = widgets.Button(
            description="",
            button_style='danger',
            icon='times',
            layout=widgets.Layout(width='40px')
        )
        
        # Parameter container (changes based on type)
        param_container = widgets.HBox()
        
        # Update parameters based on type selection
        def update_parameters(change):
            var_type = change['new']
            if var_type == 'continuous':
                param_container.children = [min_input, max_input]
            elif var_type == 'discrete':
                param_container.children = [min_input, max_input, step_input]
            else:  # categorical
                param_container.children = [categories_input]
            
            # Trigger update of minimum samples when variable type changes
            self._update_variable_display()
        
        type_selector.observe(update_parameters, names='value')
        
        # Initialize parameters
        update_parameters({'new': type_selector.value})
        
        # Create widget layout
        widget_row = widgets.HBox([
            name_input,
            type_selector,
            param_container,
            desc_input,
            remove_button
        ], layout=widgets.Layout(margin='2px 0'))
        
        # Store references for data extraction
        widget_row.name_input = name_input
        widget_row.type_selector = type_selector
        widget_row.min_input = min_input
        widget_row.max_input = max_input
        widget_row.step_input = step_input
        widget_row.categories_input = categories_input
        widget_row.desc_input = desc_input
        
        # Remove button functionality
        def remove_variable(button):
            self._remove_variable_widget(widget_row)
        
        remove_button.on_click(remove_variable)
        
        return widget_row
    
    def _on_add_variable(self, button):
        """Handle add variable button click."""
        new_widget = self._create_variable_widget()
        self.variable_widgets.append(new_widget)
        self._update_variable_display()
    
    def _on_clear_variables(self, button):
        """Handle clear all variables button click."""
        self.variable_widgets.clear()
        self._update_variable_display()
    
    def _remove_variable_widget(self, widget_to_remove):
        """Remove a specific variable widget."""
        if widget_to_remove in self.variable_widgets:
            self.variable_widgets.remove(widget_to_remove)
            self._update_variable_display()
    
    def _update_variable_display(self):
        """Update the variable list display."""
        if self.variable_widgets:
            self.variable_list.children = self.variable_widgets
        else:
            self.variable_list.children = [
                widgets.HTML("<i>No variables defined. Click 'Add Variable' to start.</i>")
            ]
        
        # Update counter
        count = len(self.variable_widgets)
        min_samples = max(count ** 2, 4)
        self.variable_count_label.value = f"<b>Variables: {count}</b> (Min samples: {min_samples})"
        
        # Update minimum sample size if slider exists
        if hasattr(self, 'sample_size_slider') and count > 0:
            self.sample_size_slider.min = min_samples
            if self.sample_size_slider.value < min_samples:
                self.sample_size_slider.value = min_samples
            # Update the label too
            if hasattr(self, 'min_size_label'):
                self.min_size_label.value = f"<i>Minimum recommended: {min_samples} samples</i>"
    
    def get_variables_from_widgets(self) -> List[Variable]:
        """Extract variables from current widget state."""
        variables = []
        
        for widget in self.variable_widgets:
            try:
                name = widget.name_input.value.strip()
                if not name:
                    continue
                
                var_type = VariableType(widget.type_selector.value)
                description = widget.desc_input.value.strip()
                
                if var_type == VariableType.CONTINUOUS:
                    variable = Variable(
                        name=name,
                        type=var_type,
                        description=description,
                        min_value=self._parse_float(widget.min_input.value),
                        max_value=self._parse_float(widget.max_input.value)
                    )
                
                elif var_type == VariableType.DISCRETE:
                    variable = Variable(
                        name=name,
                        type=var_type,
                        description=description,
                        min_value=self._parse_float(widget.min_input.value),
                        max_value=self._parse_float(widget.max_input.value),
                        step_size=self._parse_float(widget.step_input.value)
                    )
                
                elif var_type == VariableType.CATEGORICAL:
                    categories = [cat.strip() for cat in widget.categories_input.value.split(',') 
                                 if cat.strip()]
                    variable = Variable(
                        name=name,
                        type=var_type,
                        description=description,
                        categories=categories
                    )
                
                # Validate variable
                is_valid, error_msg = variable.validate()
                if is_valid:
                    variables.append(variable)
                else:
                    print(f"Warning: Invalid variable '{name}': {error_msg}")
            
            except Exception as e:
                print(f"Warning: Error processing variable: {str(e)}")
        
        return variables
    
    def get_sampling_parameters(self) -> Dict[str, Any]:
        """Get sampling parameters from UI."""
        params = {
            'n_samples': self.sample_size_slider.value if hasattr(self, 'sample_size_slider') else 25,
            'random_state': self.random_seed.value if hasattr(self, 'random_seed') else 42
        }
        
        # Add advanced options if available
        if hasattr(self, 'advanced_options_container'):
            for child in self.advanced_options_container.children:
                if hasattr(child, 'value'):
                    params[child.description.replace(':', '')] = child.value
        
        return params
    
    def get_selected_plot_type(self) -> str:
        """Get currently selected plot type."""
        if hasattr(self, 'plot_type_dropdown'):
            return self.plot_type_dropdown.value
        return 'splom'
    
    def show_progress(self, container: widgets.Widget, message: str):
        """Show progress indicator."""
        if hasattr(self, 'progress_bar') and hasattr(self, 'progress_label'):
            self.progress_bar.layout.visibility = 'visible'
            self.progress_label.layout.visibility = 'visible'
            self.progress_label.value = f"<i>{message}</i>"
            self.progress_bar.value = 50  # Indeterminate progress
    
    def hide_progress(self, container: widgets.Widget):
        """Hide progress indicator."""
        if hasattr(self, 'progress_bar') and hasattr(self, 'progress_label'):
            self.progress_bar.layout.visibility = 'hidden'
            self.progress_label.layout.visibility = 'hidden'
    
    def update_data_table(self, container: widgets.Widget, data: pd.DataFrame):
        """Update data table display."""
        # Find the Output widget in the container
        output_widget = None
        if hasattr(container, 'children') and len(container.children) > 1:
            output_widget = container.children[1]  # Second child should be the Output widget
        
        if output_widget and hasattr(output_widget, 'clear_output'):
            with output_widget:
                output_widget.clear_output(wait=True)
                
                if len(data) > 0:
                    # Show more rows and make it scrollable
                    display_data = data.head(50)  # Show first 50 rows
                    
                    display(display_data.style.set_table_attributes(
                        'style="font-size: 12px; width: 100%;"'
                    ).format(precision=3))
                    
                    if len(data) > 50:
                        print(f"\nShowing first 50 of {len(data)} samples. Full data available for export.")
                else:
                    print("No samples generated yet")
    
    def update_summary_statistics(self, container: widgets.Widget, data: pd.DataFrame, 
                                 variables: List[Variable]):
        """Update summary statistics display."""
        if not hasattr(self, 'summary_display'):
            return
        
        if data.empty:
            self.summary_display.value = "<i>No data available</i>"
            return
        
        # Generate summary HTML
        html_parts = []
        html_parts.append("<div style='font-family: monospace;'>")
        html_parts.append(f"<p><b>Total Samples:</b> {len(data)}</p>")
        html_parts.append(f"<p><b>Variables:</b> {len(variables)}</p>")
        
        # Variable-wise statistics
        html_parts.append("<h4>Variable Statistics:</h4>")
        html_parts.append("<table style='border-collapse: collapse; margin: 10px 0;'>")
        html_parts.append("<tr style='border-bottom: 1px solid #ddd;'><th style='padding: 5px; text-align: left;'>Variable</th><th style='padding: 5px;'>Type</th><th style='padding: 5px;'>Min</th><th style='padding: 5px;'>Max</th><th style='padding: 5px;'>Unique</th></tr>")
        
        for var in variables:
            if var.name in data.columns:
                col_data = data[var.name]
                if var.type in [VariableType.CONTINUOUS, VariableType.DISCRETE]:
                    numeric_data = pd.to_numeric(col_data, errors='coerce')
                    min_val = f"{numeric_data.min():.3f}"
                    max_val = f"{numeric_data.max():.3f}"
                else:
                    min_val = "N/A"
                    max_val = "N/A"
                
                unique_count = col_data.nunique()
                
                html_parts.append(f"<tr style='border-bottom: 1px solid #eee;'>")
                html_parts.append(f"<td style='padding: 5px;'>{var.name}</td>")
                html_parts.append(f"<td style='padding: 5px;'>{var.type.value}</td>")
                html_parts.append(f"<td style='padding: 5px;'>{min_val}</td>")
                html_parts.append(f"<td style='padding: 5px;'>{max_val}</td>")
                html_parts.append(f"<td style='padding: 5px;'>{unique_count}</td>")
                html_parts.append(f"</tr>")
        
        html_parts.append("</table>")
        html_parts.append("</div>")
        
        self.summary_display.value = ''.join(html_parts)
    
    def update_metrics_display(self, container: widgets.Widget, metrics: Dict[str, Any]):
        """Update quality metrics display."""
        if not hasattr(self, 'metrics_display'):
            return
        
        if not metrics:
            self.metrics_display.value = "<i>No metrics available</i>"
            return
        
        # Generate metrics HTML
        html_parts = []
        html_parts.append("<div style='font-family: monospace;'>")
        
        # Basic metrics
        if 'sample_count' in metrics:
            html_parts.append(f"<p><b>Samples Generated:</b> {metrics['sample_count']}</p>")
        
        # Space-filling metrics
        if 'min_distance' in metrics:
            html_parts.append("<h4>Space-Filling Quality:</h4>")
            html_parts.append("<ul>")
            html_parts.append(f"<li><b>Minimum Distance:</b> {metrics['min_distance']:.4f}</li>")
            if 'mean_distance' in metrics:
                html_parts.append(f"<li><b>Mean Distance:</b> {metrics['mean_distance']:.4f}</li>")
            if 'distance_uniformity' in metrics:
                html_parts.append(f"<li><b>Distance Uniformity:</b> {metrics['distance_uniformity']:.4f}</li>")
            html_parts.append("</ul>")
        
        # Coverage metrics
        if 'overall_coverage' in metrics:
            html_parts.append("<h4>Design Space Coverage:</h4>")
            html_parts.append(f"<p><b>Overall Coverage:</b> {metrics['overall_coverage']:.1%}</p>")
            
            if 'coverage_per_variable' in metrics:
                html_parts.append("<ul>")
                for var_name, coverage in metrics['coverage_per_variable'].items():
                    html_parts.append(f"<li><b>{var_name}:</b> {coverage:.1%}</li>")
                html_parts.append("</ul>")
        
        # Correlation metrics
        if 'correlation_stats' in metrics:
            corr_stats = metrics['correlation_stats']
            html_parts.append("<h4>Correlation Analysis:</h4>")
            html_parts.append("<ul>")
            html_parts.append(f"<li><b>Max Correlation:</b> {corr_stats['max_correlation']:.3f}</li>")
            html_parts.append(f"<li><b>Mean Correlation:</b> {corr_stats['mean_correlation']:.3f}</li>")
            html_parts.append("</ul>")
        
        html_parts.append("</div>")
        
        self.metrics_display.value = ''.join(html_parts)
    
    def update_advanced_options(self, algorithm_name: str):
        """Update advanced options based on selected algorithm."""
        if not hasattr(self, 'advanced_options_container'):
            return
        
        # Clear existing options
        options_widgets = []
        
        if algorithm_name == "Latin Hypercube Sampling":
            optimization_dropdown = widgets.Dropdown(
                options=[('None', None), ('Random-CD', 'random-cd'), ('Lloyd', 'lloyd')],
                value=None,
                description="Optimization:",
                style=self.INPUT_STYLE
            )
            options_widgets.append(optimization_dropdown)
        
        elif algorithm_name in ["Sobol Sequences", "Halton Sequences"]:
            scramble_checkbox = widgets.Checkbox(
                value=True,
                description="Apply scrambling",
                style=self.INPUT_STYLE
            )
            options_widgets.append(scramble_checkbox)
        
        # Set content
        if options_widgets:
            self.advanced_options_container.children = options_widgets
        else:
            self.advanced_options_container.children = [
                widgets.HTML(f"<i>No advanced options available for {algorithm_name}</i>")
            ]
    
    def update_from_configuration(self, config: Dict[str, Any]):
        """Update GUI from loaded configuration."""
        # Update algorithm selection
        if 'algorithm' in config and hasattr(self, 'algorithm_dropdown'):
            if config['algorithm'] in [opt for opt, _ in self.algorithm_dropdown.options]:
                self.algorithm_dropdown.value = config['algorithm']
        
        # Update variables
        if 'variables' in config:
            self.variable_widgets.clear()
            for var_dict in config['variables']:
                var = Variable.from_dict(var_dict)
                widget = self._create_variable_widget(var)
                self.variable_widgets.append(widget)
            self._update_variable_display()
    
    def create_download_link(self, content: str, filename: str, description: str) -> str:
        """Create a download link for file content."""
        # Encode content as base64
        b64_content = base64.b64encode(content.encode()).decode()
        
        # Create download link HTML
        download_html = f"""
        <p>{description}</p>
        <a download="{filename}" href="data:text/plain;base64,{b64_content}" 
           style="background-color: #007bff; color: white; padding: 10px 15px; 
                  text-decoration: none; border-radius: 4px; display: inline-block;">
            📥 Download {filename}
        </a>
        """
        
        return download_html