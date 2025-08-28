"""
Main application controller for the Design of Experiments Voila application.
Manages tab structure, user interactions, and coordination between modules.
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import json
from datetime import datetime

# Local imports
from data_manager import DataManager
from sampling_algorithms import SamplingEngine
from gui_components import GUIComponents
from plot_manager import PlotManager
from utils import ValidationUtils, Constants


class DoEApplication:
    """Main application controller for Design of Experiments interface."""
    
    def __init__(self):
        """Initialize the DoE application with all modules."""
        # Initialize core modules
        self.data_manager = DataManager()
        self.sampling_engine = SamplingEngine()
        self.gui_components = GUIComponents()
        self.plot_manager = PlotManager()
        self.validator = ValidationUtils()
        
        # Application state
        self.current_samples = None
        self.current_figure = None
        self.current_algorithm = "Latin Hypercube Sampling"
        self.quality_metrics = {}
        
        # Create main interface
        self.main_interface = self._create_main_interface()

        # Bind generate button after GUI is created
        if hasattr(self.gui_components, 'generate_button'):
            self.gui_components.generate_button.on_click(self._on_generate_samples)

        # Delay event handler setup to ensure widgets exist
        import time
        time.sleep(0.1)  # Small delay
            
        # Setup event handlers
        self._setup_event_handlers()
        
    def _create_main_interface(self):
        """Create the main tabbed interface."""
        # Create individual tabs
        tab1 = self._create_algorithm_tab()
        tab2 = self._create_sampling_tab()
        tab3 = self._create_results_tab()
        tab4 = self._create_visualization_tab()
        tab5 = self._create_export_tab()
        
        # Create tab widget
        tabs = widgets.Tab()
        tabs.children = [tab1, tab2, tab3, tab4, tab5]
        
        # Set tab titles
        tab_titles = [
            "Algorithm & Variables",
            "Sample Generation", 
            "Results & Design",
            "Visualizations",
            "Export & Import"
        ]
        for i, title in enumerate(tab_titles):
            tabs.set_title(i, title)
        
        # Create main layout with header
        header = self._create_header()
        
        main_layout = widgets.VBox([
            header,
            tabs
        ])
        
        return main_layout
    
    def _create_header(self):
        """Create application header with title and status."""
        title = widgets.HTML(
            value="<h2 style='text-align: center; color: #2E4057;'>Design of Experiments Application</h2>",
            layout=widgets.Layout(margin='10px 0')
        )
        
        self.status_label = widgets.HTML(
            value="<p style='text-align: center; color: #666;'>Ready - Select algorithm and define variables to begin</p>",
            layout=widgets.Layout(margin='5px 0')
        )
        
        return widgets.VBox([title, self.status_label])
    
    def _create_algorithm_tab(self):
        """Create Tab 1: Algorithm Selection & Variable Configuration."""
        # Algorithm selection
        algorithm_section = self.gui_components.create_algorithm_selector()
        
        # Variable configuration  
        variable_section = self.gui_components.create_variable_configurator()
        
        # Combine sections
        tab_content = widgets.VBox([
            widgets.HTML("<h3>Algorithm Selection</h3>"),
            algorithm_section,
            widgets.HTML("<hr>"),
            widgets.HTML("<h3>Variable Configuration</h3>"),
            variable_section
        ], layout=widgets.Layout(padding='20px'))
        
        return tab_content
    
    def _create_sampling_tab(self):
        """Create Tab 2: Sample Generation."""
        # Sample size configuration
        size_section = self.gui_components.create_sample_size_configurator()
        
        # Random seed input
        seed_section = self.gui_components.create_seed_configurator()
        
        # Advanced options
        advanced_section = self.gui_components.create_advanced_options()
        
        # Generation controls
        control_section = self.gui_components.create_generation_controls()
        
        # Connect the generate button event here
        if hasattr(self.gui_components, 'generate_button'):
            self.gui_components.generate_button.on_click(self._on_generate_samples)
        
        # Progress and status
        self.progress_section = self.gui_components.create_progress_section()
        
        # Quality metrics display
        self.metrics_section = self.gui_components.create_metrics_section()
        
        tab_content = widgets.VBox([
            widgets.HTML("<h3>Sample Configuration</h3>"),
            size_section,
            seed_section,
            advanced_section,
            widgets.HTML("<hr>"),
            widgets.HTML("<h3>Generate Samples</h3>"),
            control_section,
            self.progress_section,
            widgets.HTML("<hr>"),
            widgets.HTML("<h3>Quality Metrics</h3>"),
            self.metrics_section
        ], layout=widgets.Layout(padding='20px'))
        
        return tab_content
    
    def _create_results_tab(self):
        """Create Tab 3: Results & Experimental Design."""
        # Data table section with larger display area
        self.table_section = widgets.VBox([
            widgets.HTML("<b>Generated Samples</b>"),
            widgets.Output(layout=widgets.Layout(height='400px', overflow='auto', border='1px solid #ccc'))
        ])
        
        # Summary statistics
        self.summary_section = self.gui_components.create_summary_section()
        
        # Experimental protocol
        self.protocol_section = self.gui_components.create_protocol_section()
        
        tab_content = widgets.VBox([
            widgets.HTML("<h3>Generated Samples</h3>"),
            self.table_section,
            widgets.HTML("<hr>"),
            widgets.HTML("<h3>Summary Statistics</h3>"),
            self.summary_section,
            widgets.HTML("<hr>"),
            widgets.HTML("<h3>Experimental Protocol</h3>"),
            self.protocol_section
        ], layout=widgets.Layout(padding='20px'))
        
        return tab_content
    
    def _create_visualization_tab(self):
        """Create Tab 4: Design Space Visualization."""
        # Visualization controls
        viz_controls = self.gui_components.create_visualization_controls()
        
        # Plot display area
        self.plot_area = widgets.Output(layout=widgets.Layout(height='500px'))
        
        tab_content = widgets.VBox([
            widgets.HTML("<h3>Visualization Options</h3>"),
            viz_controls,  # Make sure this line exists
            widgets.HTML("<hr>"),
            widgets.HTML("<h3>Design Space Plots</h3>"),
            self.plot_area
        ], layout=widgets.Layout(padding='20px'))
        
        return tab_content
    
    def _create_export_tab(self):
        """Create Tab 5: Data Export & Import."""
        # Export section
        export_section = self.gui_components.create_export_section()
        
        # Remove this line completely:
        # validation_section = self.gui_components.create_validation_section()
        
        tab_content = widgets.VBox([
            widgets.HTML("<h3>Export Data</h3>"),
            export_section,
            # Remove the validation section from here too
        ], layout=widgets.Layout(padding='20px'))
        
        scrollable_tab = widgets.VBox(
            children=[tab_content], 
            layout=widgets.Layout(
                height='600px', 
                overflow='auto'
            )
        )
        
        return scrollable_tab
    
    def _setup_event_handlers(self):
        """Setup event handlers for user interactions."""
        # Algorithm selection handler
        if hasattr(self.gui_components, 'algorithm_dropdown'):
            self.gui_components.algorithm_dropdown.observe(
                self._on_algorithm_change, names='value'
            )
        
        # Sample generation handler - bind after GUI components are created
        if hasattr(self.gui_components, 'generate_button'):
            self.gui_components.generate_button.on_click(self._on_generate_samples)
        
        # Visualization handler
        if hasattr(self.gui_components, 'plot_type_dropdown'):
            self.gui_components.plot_type_dropdown.observe(
                self._on_plot_type_change, names='value'
            )

        # Connect update plot button
        if hasattr(self.gui_components, 'update_plot_button'):
            self.gui_components.update_plot_button.on_click(lambda x: self._update_visualizations())

        # Connect export plot button
        if hasattr(self.gui_components, 'export_plot_button'):
            self.gui_components.export_plot_button.on_click(self._on_export_plot)
    
    def _on_algorithm_change(self, change):
        """Handle algorithm selection change."""
        self.current_algorithm = change['new']
        self._update_status(f"Algorithm changed to: {self.current_algorithm}")
        
        # Update advanced options based on algorithm
        self.gui_components.update_advanced_options(self.current_algorithm)
    
    def _on_generate_samples(self, button):
        """Handle sample generation request."""
        try:
            # Get variables from GUI
            variables = self.gui_components.get_variables_from_widgets()
            
            if not variables:
                self._update_status("Error: Please define at least one variable", error=True)
                return
            
            # Update data manager with current variables
            self.data_manager.clear_all_variables()
            for var in variables:
                success, msg = self.data_manager.add_variable(var)
                if not success:
                    self._update_status(f"Error: {msg}", error=True)
                    return
            
            # Get sampling parameters
            params = self.gui_components.get_sampling_parameters()
            
            # Show progress
            self._show_progress("Generating samples...")
            
            # Generate samples
            self.current_samples = self.sampling_engine.generate_samples(
                variables=variables,
                algorithm=self.current_algorithm,
                **params
            )
            
            # Calculate quality metrics
            self.quality_metrics = self.sampling_engine.calculate_quality_metrics(
                self.current_samples, variables
            )

            # Update displays
            self._update_results_display()
            self._update_metrics_display()
            self._update_visualizations()
            
            self._hide_progress()
            self._update_status(f"Generated {len(self.current_samples)} samples successfully")
            
            # Add this new code here - after successful generation
            if self.current_samples is not None and not self.current_samples.empty:
                self.gui_components.set_current_data(
                    samples=self.current_samples,
                    variables=variables,
                    metrics=self.quality_metrics,
                    algorithm=self.current_algorithm,
                    random_seed=params.get('random_state', 42)
                )
            
        except Exception as e:
            self._hide_progress()
            self._update_status(f"Error generating samples: {str(e)}", error=True)
            import traceback
            traceback.print_exc()
    
    def _on_plot_type_change(self, change):
        """Handle visualization type change."""
        if self.current_samples is not None:
            self._update_visualizations()
    
    def _show_progress(self, message):
        """Show progress indicator."""
        if hasattr(self, 'progress_section'):
            self.gui_components.show_progress(self.progress_section, message)
    
    def _hide_progress(self):
        """Hide progress indicator."""
        if hasattr(self, 'progress_section'):
            self.gui_components.hide_progress(self.progress_section)
    
    def _update_status(self, message, error=False):
        """Update application status message."""
        color = "#d32f2f" if error else "#666"
        self.status_label.value = f"<p style='text-align: center; color: {color};'>{message}</p>"
    
    def _update_results_display(self):
        """Update the results table and summary."""
        if self.current_samples is not None:
            # Update the table section title with algorithm name
            if hasattr(self, 'table_section'):
                algorithm_display = self.current_algorithm.replace('_', ' ')  # Clean up name
                self.table_section.children = (
                    widgets.HTML(f"<b>Generated Samples with {algorithm_display}</b>"),
                    self.table_section.children[1]  # Keep the existing output widget
                )
            
            # Update data table
            self.gui_components.update_data_table(
                self.table_section, 
                self.current_samples
            )
            
            # Update summary statistics
            variables = self.data_manager.get_variables()
            self.gui_components.update_summary_statistics(
                self.summary_section,
                self.current_samples,
                variables
            )
    
    def _update_metrics_display(self):
        """Update quality metrics display."""
        if self.quality_metrics:
            self.gui_components.update_metrics_display(
                self.metrics_section,
                self.quality_metrics
            )
    
    def _update_visualizations(self):
        """Update visualization plots."""
        if self.current_samples is not None:
            if not hasattr(self, 'plot_area'):
                return
                
            with self.plot_area:
                clear_output(wait=True)
                
                try:
                    plot_type = self.gui_components.get_selected_plot_type()
                    variables = self.data_manager.get_variables()
                    
                    fig = self.plot_manager.create_plot(
                        plot_type=plot_type,
                        data=self.current_samples,
                        variables=variables,
                        algorithm=self.current_algorithm
                    )
                    
                    if fig:
                        fig.show()  # Use show() instead of display() in Voila
                        self.current_figure = fig
                except Exception as e:
                    print(f"Error creating plot: {str(e)}")

    def _on_export_plot(self, button):
        """Handle plot export."""
        if not hasattr(self, 'current_figure') or self.current_figure is None:
            # Show error in plot area temporarily
            with self.plot_area:
                clear_output(wait=True)
                display(widgets.HTML("<p style='color: red;'>No plot to export. Generate a plot first.</p>"))
            return
        
        try:
            # Show processing message in plot area
            with self.plot_area:
                clear_output(wait=True)
                display(self.current_figure)
                display(widgets.HTML("<p style='color: #007bff;'>Creating image file...</p>"))
            
            # Export as HTML
            html_content = self.current_figure.to_html(include_plotlyjs=True)
            
            # Create download link
            import base64
            b64_content = base64.b64encode(html_content.encode()).decode()
            
            download_html = f"""
            <p style='color: green;'>Plot exported successfully!</p>
            <a download="doe_plot.html" href="data:text/html;base64,{b64_content}" 
               style="background-color: #28a745; color: white; padding: 8px 12px; 
                      text-decoration: none; border-radius: 4px; display: inline-block; margin-top: 5px;">
                Download Plot (HTML)
            </a>
            """
            
            # Show plot with download link below it
            with self.plot_area:
                clear_output(wait=True)
                display(self.current_figure)
                display(widgets.HTML(download_html))
                
        except Exception as e:
            with self.plot_area:
                clear_output(wait=True)
                display(self.current_figure)
                display(widgets.HTML(f"<p style='color: red;'>Export failed: {str(e)}</p>"))
    
    def get_status(self):
        """Get current application status."""
        return {
            'variables_count': len(self.data_manager.get_variables()),
            'samples_generated': len(self.current_samples) if self.current_samples is not None else 0,
            'current_algorithm': self.current_algorithm,
            'has_quality_metrics': bool(self.quality_metrics)
        }
    
    def get_current_tab(self):
        """Get currently selected tab index."""
        # This would need to be implemented with tab observation
        return 0
    
    def export_configuration(self):
        """Export current application configuration."""
        config = {
            'timestamp': datetime.now().isoformat(),
            'algorithm': self.current_algorithm,
            'variables': [var.to_dict() for var in self.data_manager.get_variables()],
            'samples': self.current_samples.to_dict('records') if self.current_samples is not None else None,
            'quality_metrics': self.quality_metrics,
            'version': Constants.APPLICATION_VERSION
        }
        return json.dumps(config, indent=2, default=str)
    
    def import_configuration(self, config_json):
        """Import application configuration from JSON."""
        try:
            config = json.loads(config_json)
            
            # Restore algorithm
            self.current_algorithm = config.get('algorithm', 'Latin Hypercube Sampling')
            
            # Restore variables
            variables = config.get('variables', [])
            success, msg = self.data_manager.set_variables(variables)
            
            # Restore samples if available
            if config.get('samples'):
                self.current_samples = pd.DataFrame(config['samples'])
            
            # Restore quality metrics
            self.quality_metrics = config.get('quality_metrics', {})
            
            # Update displays
            self.gui_components.update_from_configuration(config)
            self._update_results_display()
            self._update_metrics_display()
            self._update_visualization_controls()
            self._update_visualizations()
            
            self._update_status("Configuration imported successfully")
            return True
            
        except Exception as e:
            self._update_status(f"Error importing configuration: {str(e)}", error=True)
            return False