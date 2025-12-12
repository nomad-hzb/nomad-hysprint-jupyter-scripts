"""
Simplified Application Controller
Main orchestrator for the JV Analysis Dashboard with cleaner organization.
"""

__author__ = "Edgar Nandayapa"
__institution__ = "Helmholtz-Zentrum Berlin"
__created__ = "August 2025"

import ipywidgets as widgets
from IPython.display import display, clear_output, Markdown
import os
import io
import base64
import zipfile
import requests
import json
import sys
from utils import save_combined_excel_data
from resizable_plot_utility import ResizablePlotManager
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
import plotly.graph_objects as go
import pandas as pd

# Add parent directory for shared modules
parent_dir = os.path.dirname(os.getcwd())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the new organized modules
from gui_components import AuthenticationUI, FilterUI, PlotUI, SaveUI, ColorSchemeSelector, InfoUI
from data_manager import DataManager
from plot_manager import plotting_string_action
from utils import save_full_data_frame
from batch_selection import create_batch_selection
from error_handler import ErrorHandler

# Import shared modules
try:
    from api_calls import get_all_measurements_except_JV, get_ids_in_batch
except ImportError:
    print("Warning: Some API modules not available")


class SimpleAuthManager:
    """Simplified authentication manager"""
    
    def __init__(self, base_url, api_endpoint):
        self.base_url = base_url
        self.api_endpoint = api_endpoint
        self.url = f"{base_url}{api_endpoint}"
        self.current_token = None
        self.current_user_info = None
        self.status_callback = None
        self.api_client = self  # Compatibility
    
    def set_status_callback(self, callback):
        self.status_callback = callback
    
    def _update_status(self, message, color=None):
        if self.status_callback:
            self.status_callback(message, color)
    
    def authenticate_with_credentials(self, username, password):
        if not username or not password:
            raise ValueError("Username and Password are required.")
        
        auth_dict = dict(username=username, password=password)
        token_url = f"{self.url}/auth/token"
        
        response = requests.get(token_url, params=auth_dict, timeout=10)
        response.raise_for_status()
        
        token_data = response.json()
        if 'access_token' not in token_data:
            raise ValueError("Access token not found in response.")
        
        self.current_token = token_data['access_token']
        return self.current_token
    
    def authenticate_with_token(self, token=None):
        if token is None:
            token = os.environ.get('NOMAD_CLIENT_ACCESS_TOKEN')
            if not token:
                raise ValueError("Token not found in environment variable.")
        
        self.current_token = token
        return self.current_token
    
    def verify_token(self):
        if not self.current_token:
            raise ValueError("No token available for verification.")
        
        verify_url = f"{self.url}/users/me"
        headers = {'Authorization': f'Bearer {self.current_token}'}
        
        verify_response = requests.get(verify_url, headers=headers, timeout=10)
        verify_response.raise_for_status()
        
        self.current_user_info = verify_response.json()
        return self.current_user_info
    
    def is_authenticated(self):
        return self.current_token is not None and self.current_user_info is not None
    
    def clear_authentication(self):
        self.current_token = None
        self.current_user_info = None
    
    def get_user_display_name(self):
        if not self.current_user_info:
            return 'Unknown User'
        return self.current_user_info.get('name', self.current_user_info.get('username', 'Unknown User'))


class JVAnalysisApp:
    """Simplified main application controller"""
    
    def __init__(self):
        # Initialize core components
        self.auth_manager = SimpleAuthManager("https://nomad-hzb-se.de", "/nomad-oasis/api/v1")
        self.data_manager = DataManager(self.auth_manager)
        
        # Application state
        self.is_conditions = False
        self.global_plot_data = {'figs': [], 'names': [], 'workbook': None}
        
        # Initialize UI components
        self._init_ui_components()
        self._create_tabs()
        self._setup_callbacks()
        self._auto_authenticate()
    
    def _init_ui_components(self):
        """Initialize UI components with proper data manager integration"""
        self.auth_ui = AuthenticationUI(self.auth_manager)
        self.filter_ui = FilterUI()
        self.plot_ui = PlotUI()
        self.save_ui = SaveUI()
        self.color_selector = ColorSchemeSelector()
        self.info_ui = InfoUI()  # ADD this line
    
        # Give FilterUI access to DataManager for preview functionality
        self.filter_ui.data_manager = self.data_manager
            
        # Tab-specific widgets
        self.batch_selection_container = widgets.Output()
        self.load_status_output = widgets.Output(
            layout=widgets.Layout(border='1px solid #eee', padding='10px', 
                                margin='10px 0 0 0', min_height='100px')
        )
        
        # Variables tab widgets
        self.default_variables = widgets.Dropdown(
            options=['all', 'Batch name', 'Variation'],
            value='Variation',
            description='Defaults:',
            style={'description_width': 'initial'}
        )
        
        self.dynamic_content = widgets.Output()
        self.results_content = widgets.Output(layout={'width': '400px', 'height': '500px', 'overflow': 'scroll'})
        self.read_output = widgets.Output()
        self.download_content = widgets.Output()
        self.show_other_measurements = widgets.Output()
        
        self.download_button = widgets.Button(
            description='Download JV',
            button_style='info',
            layout=widgets.Layout(min_width='150px'),
            disabled=True
        )
        
        self.download_curves_button = widgets.Button(
            description='Download Curves',
            button_style='info',
            layout=widgets.Layout(min_width='150px'),
            disabled=True
        )
    
    def _create_tabs(self):
        """Create tab system"""
        self.select_upload_tab = widgets.VBox([
            widgets.HTML("<h3>Select Upload</h3>"),
            widgets.HTML("<p><i>Select one or multiple batches</i></p>"),
            self.batch_selection_container,
            self.load_status_output
        ])
        
        self.add_variables_tab = widgets.VBox([
            widgets.HTML("<h3>Add Variable Names</h3>"),
            self.dynamic_content,
            widgets.HTML("<h3>Download Data</h3>"),
            widgets.VBox([self.download_content]),
            widgets.HTML("<h3>Other Measurements</h3>"),
            widgets.VBox([self.show_other_measurements])
        ])
        
        # Create plot tab with color selector between controls and plots
        plot_tab_content = widgets.VBox([
            self.plot_ui.get_widget(),
            widgets.HTML("<hr style='margin: 20px 0;'>"),  # Separator
            self.color_selector.get_widget(),
            widgets.HTML("<hr style='margin: 20px 0;'>"),  # Another separator
            widgets.HTML("<h3>Generated Plots</h3>"),
            self.plot_ui.plotted_content  # Now plots appear after color selection
        ])
        
        self.tabs = widgets.Tab()
        self.tabs.children = [
            self.select_upload_tab,
            self.add_variables_tab,
            self.filter_ui.get_widget(),
            plot_tab_content,  # CHANGE this line
            self.save_ui.get_widget()
        ]
        
        tab_labels = ['Select Upload', 'Add Variable Names', 'Select Filters', 'Select Plots', 'Save Results']
        for i, title in enumerate(tab_labels):
            self.tabs.set_title(i, title)
    
    def _setup_callbacks(self):
        """Setup all callbacks"""
        self.auth_ui.set_success_callback(self._on_auth_success)
        self.filter_ui.set_apply_callback(self._on_apply_filters)
        self.plot_ui.set_plot_callback(self._on_create_plots)
        self.save_ui.set_save_callbacks(self._on_save_plots, self._on_save_data, self._on_save_all)
        self.default_variables.observe(self._on_change_default_variables, names=['value'])
        self.download_button.on_click(self._download_jv_data)
        self.download_curves_button.on_click(self._download_curves_data)
    
    def _auto_authenticate(self):
        """Auto-authenticate based on environment"""
        is_hub_environment = bool(os.environ.get('JUPYTERHUB_USER'))
        
        if is_hub_environment:
            self.auth_ui.auth_method_selector.value = 'Token (from ENV)'
            self.auth_ui.local_auth_box.layout.display = 'none'
        else:
            self.auth_ui.auth_method_selector.value = 'Username/Password'
            self.auth_ui.local_auth_box.layout.display = 'flex'
        
        self.auth_ui._on_auth_button_clicked(None)
    
    def _on_auth_success(self):
        """Handle successful authentication"""
        self.tabs.selected_index = 0
        self.auth_ui.close_settings()
        self._init_batch_selection()
    
    def _init_batch_selection(self):
        """Initialize batch selection after authentication"""
        with self.batch_selection_container:
            clear_output(wait=True)
            
            if not self.auth_manager.is_authenticated():
                print("Please authenticate first before loading batch data.")
                return
            
            try:
                url = self.auth_manager.url
                token = self.auth_manager.current_token
                batch_selection_widget = create_batch_selection(url, token, self._load_data_from_selection)
                display(batch_selection_widget)
            except Exception as e:
                ErrorHandler.log_error("initializing batch selection", e, self.batch_selection_container)
    
    def _load_data_from_selection(self, batch_selector):
        """Load data from batch selection with user feedback"""
        batch_ids = list(batch_selector.value) if batch_selector.value else []
        
        success = self.data_manager.load_batch_data(batch_ids, self.load_status_output)
        
        if success:
            # Set data for sample selection in filter UI
            data = self.data_manager.get_data()
            self.filter_ui.set_sample_data(data)
            
            # Show initial data summary
            with self.load_status_output:
                print(f"\nData Loading Summary:")
                print(f"   Total records loaded: {len(data['jvc'])}")
                print(f"   Unique samples: {data['jvc']['sample'].nunique()}")
                print(f"   Unique cells: {data['jvc'].groupby('sample')['cell'].nunique().sum()}")
                print(f"   Batches: {data['jvc']['batch'].nunique()}")
                print(f"\nData ready for variable assignment and filtering!")
            
            self._enable_tab(1)
            self.tabs.selected_index = 1
            self._make_variables_menu()
    
    def _make_variables_menu(self):
        """Create variables menu using DataManager's summary"""
        unique_vals = self.data_manager.get_unique_values()
        data = self.data_manager.get_data()
        
        variables_markdown = f"""
# Add variable names
There are {len(unique_vals)} samples found.
If you tested specific variables or conditions for each sample, please write them down below.
"""
        
        results_markdown = self.data_manager.generate_summary_statistics(data['jvc'])
        
        with self.dynamic_content:
            clear_output(wait=True)
            display(Markdown(variables_markdown))
            display(self.default_variables)
            
            widgets_table, text_widgets_dict = self._create_widgets_table(unique_vals)
            retrieve_button = widgets.Button(
                description="Confirm variables",
                button_style='success',
                layout=widgets.Layout(min_width='150px')
            )
            retrieve_button.on_click(lambda b: self._on_retrieve_clicked(text_widgets_dict))
            
            information_group = widgets.HBox([widgets_table, self.results_content])
            display(information_group)
            button_group = widgets.HBox([retrieve_button, self.read_output])
            display(button_group)
        
        # Show download buttons
        with self.download_content:
            clear_output(wait=True)
            self.download_button.disabled = True
            self.download_curves_button.disabled = True
            download_box = widgets.HBox([self.download_button, self.download_curves_button])
            display(download_box)
            display(widgets.HTML("<p><i>Download buttons will be enabled after confirming variables.</i></p>"))
        
        with self.results_content:
            clear_output()
            display(Markdown(results_markdown))
        
        with self.read_output:
            clear_output()
            print("⚠️ Variables not loaded")
    
    def _create_widgets_table(self, elements_list):
        """Create widgets table for variable input"""
        rows = []
        text_widgets = {}
        
        for item in elements_list:
            item_split = item.split("&")
            batch, variable = "", item
            if len(item_split) >= 2:
                batch, variable = item_split[0], "&".join(item_split[1:])
            
            default_value = ""
            if self.default_variables.value == "Batch name":
                default_value = batch if batch else "_".join(item.split("_")[:-1])
            elif self.default_variables.value == "Variation":
                default_value = variable
            
            label = widgets.Label(value=variable)
            text_input = widgets.Text(value=default_value, placeholder='Variable e.g. 1000 rpm')
            row = widgets.HBox([label, text_input])
            rows.append(row)
            text_widgets[item] = text_input
        
        return widgets.VBox(rows), text_widgets
    
    def _on_retrieve_clicked(self, text_widgets_dict):
        """Handle variable retrieval and condition assignment"""
        self.is_conditions = True
        
        # Create conditions_dict from the text widget values
        data = self.data_manager.get_data()
        if data and 'jvc' in data:
            conditions_dict = {}
            
            # For each unique identifier, get the user's input for the condition
            for identifier in data['jvc']['identifier'].unique():
                if identifier in text_widgets_dict:
                    # Use the user's input from the text widget
                    user_condition = text_widgets_dict[identifier].value.strip()
                    if user_condition:
                        conditions_dict[identifier] = user_condition
                    else:
                        # Fallback to variation from identifier if user didn't enter anything
                        if identifier and '&' in str(identifier):
                            variation = str(identifier).split('&', 1)[1]
                            conditions_dict[identifier] = variation
                        else:
                            conditions_dict[identifier] = "Unknown"
                else:
                    # Fallback for identifiers not in text_widgets_dict
                    if identifier and '&' in str(identifier):
                        variation = str(identifier).split('&', 1)[1]
                        conditions_dict[identifier] = variation
                    else:
                        conditions_dict[identifier] = "Unknown"
        
        # Apply the conditions
        success = self.data_manager.apply_conditions(conditions_dict)
        
        if success:
            # Update FilterUI with the new condition data
            updated_data = self.data_manager.get_data()
            self.filter_ui.set_sample_data(updated_data)
            
            with self.read_output:
                clear_output()
                print("✅ Variables loaded successfully")
            
            with self.download_content:
                clear_output(wait=True)
                display(widgets.HTML("<h3>Download Data</h3>"))
                self.download_button.disabled = False
                self.download_curves_button.disabled = False
                download_box = widgets.HBox([self.download_button, self.download_curves_button])
                display(download_box)
                display(widgets.HTML("<p><i>✅ Download buttons are now enabled!</i></p>"))
            
            self._show_measurements_table()
            self._enable_tab(2)
            self.tabs.selected_index = 2
        else:
            with self.read_output:
                clear_output()
                print("❌ Error loading variables")
    
    def _on_change_default_variables(self, change):
        """Handle default variables change"""
        self._make_variables_menu()
    
    def _download_jv_data(self, e=None):
        """Download JV data as CSV"""
        jvc_data, _ = self.data_manager.get_export_data()
        if jvc_data is not None:
            jvc_csv = jvc_data.to_csv(index=False)
            self.save_ui.trigger_download(jvc_csv, 'export_jvc.csv', 'text/plain')
        else:
            print("No JV data available for download")
    
    def _download_curves_data(self, e=None):
        """Download curves data as CSV"""
        _, curves_data = self.data_manager.get_export_data()
        if curves_data is not None:
            curves_csv = curves_data.to_csv(index=False)
            self.save_ui.trigger_download(curves_csv, 'export_curves.csv', 'text/plain')
        else:
            print("No curves data available for download")
    
    def _show_measurements_table(self):
        """Show other measurements table"""
        try:
            data = self.data_manager.get_data()
            if not data or "jvc" not in data:
                return
            
            with self.show_other_measurements:
                clear_output(wait=True)
                print("Loading measurements data...")
                
                batch_ids_value = list(data['jvc']['batch'].unique())
                if not batch_ids_value:
                    print("No batch IDs found in loaded data.")
                    return
                
                url = self.auth_manager.url
                token = self.auth_manager.current_token
                
                try:
                    sample_ids = get_ids_in_batch(url, token, batch_ids_value)
                    measurements_data = get_all_measurements_except_JV(url, token, sample_ids)
                    
                    import pandas as pd
                    df = pd.DataFrame()
                    
                    def make_clickable(r):
                        base_url = self.auth_manager.base_url
                        if "SEM" in r[1]["entry_type"]:
                            return f'<a href="{base_url}/nomad-oasis/gui/entry/id/{r[1]["entry_id"]}/data/data/images:0/image_preview/preview" rel="noopener noreferrer" target="_blank">{r[1]["entry_type"].split("_")[-1]}</a>'
                        return f'<a href="{base_url}/nomad-oasis/gui/entry/id/{r[1]["entry_id"]}/data/data" rel="noopener noreferrer" target="_blank">{r[1]["entry_type"].split("_")[-1]}</a>'
                    
                    for key, value in measurements_data.items():
                        if value:
                            df[key] = pd.Series([make_clickable(r) for r in value])
                    
                    if df.empty:
                        print("No additional measurements found.")
                    else:
                        display(widgets.HTML("<h3>Additional Measurements</h3>"))
                        display(widgets.HTML(df.to_html(escape=False)))
                
                except AssertionError:
                    print("No additional measurements found for the selected batches.")
                except Exception as api_error:
                    print(f"Could not load additional measurements: {api_error}")
        
        except Exception as e:
            ErrorHandler.log_error("displaying measurements", e, self.show_other_measurements)
    
    def _on_apply_filters(self, b):
        """Handle filter application using sample-based filtering"""
        data = self.data_manager.get_data()
        if not data or "jvc" not in data:
            with self.filter_ui.main_output:
                print("No data loaded. Please load data first.")
            return

        # Get selected samples and filter values
        selected_items = self.filter_ui.get_selected_items()
        filter_values = self.filter_ui.get_filter_values()
        direction_value = self.filter_ui.get_direction_value()
    
        with self.filter_ui.main_output:
            clear_output(wait=True)
            try:
                # Apply the filters
                filtered_df, omitted_df, filter_params = self.data_manager.apply_filters(
                    filter_values, direction_value, selected_items, verbose=True
                )
                
                # Show results with detailed breakdown
                original_count = len(data["jvc"])
                final_count = len(filtered_df)
                total_excluded = original_count - final_count
                
                # Calculate the breakdown of exclusions
                if selected_items is None:
                    # No sample selection applied - all samples used
                    sample_selection_excluded = 0
                    filter_condition_excluded = total_excluded
                else:
                    # Sample selection was applied
                    sample_selection_excluded = len(omitted_df[omitted_df['filter_reason'].str.contains('sample/cell not selected', na=False)])
                    filter_condition_excluded = len(omitted_df[~omitted_df['filter_reason'].str.contains('sample/cell not selected', na=False)])
                
                print(f"\nFiltering Results:")
                print(f"   Original dataset: {original_count} records")
                print(f"   Excluded by sample selection: {sample_selection_excluded} records")
                print(f"   Excluded by filter conditions: {filter_condition_excluded} records")
                print(f"   Final dataset: {final_count} records")
                print(f"   Retention rate: {(final_count/original_count)*100:.1f}%")
                
                # Show which selected samples were removed by filter conditions (not by selection)
                if len(omitted_df) > 0:
                    # Only show samples that were removed by JV parameter filters, not by sample selection
                    jv_filtered_samples = omitted_df[~omitted_df['filter_reason'].str.contains('sample/cell not selected', na=False)]
                    
                    if len(jv_filtered_samples) > 0:
                        print(f"\n📋 Selected samples removed by filter conditions:")
                        
                        # Group by sample and show reasons
                        sample_filter_reasons = jv_filtered_samples.groupby(['sample', 'cell'])['filter_reason'].first()
                        
                        for (sample, cell), reason in sample_filter_reasons.items():
                            # Clean up the reason string - remove trailing comma and space
                            clean_reason = reason.rstrip(', ')
                            print(f"   • {sample}_{cell}: {clean_reason}")
                
                if final_count > 0:
                    print(f"\n✅ Filtering complete! Proceed to plotting tab.")
                    self._enable_tab(3)
                else:
                    print(f"\n⚠️ No data available for plotting. Please adjust filters.")
                
            except Exception as e:
                ErrorHandler.log_error("applying filters", e, self.filter_ui.main_output)
    
    def _on_create_plots(self, b):
        """Handle plot creation"""
        data = self.data_manager.get_data()
        if not data or 'filtered' not in data:
            with self.plot_ui.plotted_content:
                print("Please apply filters first in the 'Select Filters' tab.")
            return
        
        filtered_data = data.get('filtered')
        if filtered_data is None or filtered_data.empty:
            with self.plot_ui.plotted_content:
                print("No data remains after filtering. Please adjust your filters.")
            return
        
        plot_selections = self.plot_ui.get_plot_selections()
        max_categories = 8  # Default estimate
        
        # Estimate how many colors we might need based on the data
        if hasattr(filtered_data, 'condition'):
            max_categories = max(max_categories, filtered_data['condition'].nunique())
        if hasattr(filtered_data, 'status'):
            max_categories = max(max_categories, filtered_data['status'].nunique())
        
        sampling_method = self.color_selector.sampling_dropdown.value
        selected_colors = self.color_selector.get_colors(num_colors=max_categories, sampling=sampling_method)

        # Show processing message immediately
        with self.plot_ui.plotted_content:
            clear_output(wait=True)
            display(widgets.HTML("""
            <div style="text-align: center; padding: 40px; background-color: #f8f9fa; border-radius: 8px; border: 2px solid #007bff;">
                <div style="font-size: 24px; margin-bottom: 15px;">🔄</div>
                <h3 style="color: #007bff; margin-bottom: 10px;">Creating Plots...</h3>
                <p style="color: #6c757d;">Please be patient while we generate your visualizations.</p>
            </div>
            """))
        
        # GET SELECTED COLORS - ADD this line:
        sampling_method = self.color_selector.sampling_dropdown.value
        selected_colors = self.color_selector.get_colors(num_colors=None, sampling=sampling_method)
        
        try:
            # Create workbook for Excel export with proper analysis sheets
            wb = openpyxl.Workbook()
            wb.remove(wb.active)  # Remove default sheet
            
            # Add main data sheet first
            main_sheet = wb.create_sheet(title='All_data')
            for r in dataframe_to_rows(filtered_data, index=True, header=True):
                main_sheet.append(r)
            
            # Create analysis sheets for each boxplot
            from utils import save_combined_excel_data
            
            # Variable name mapping (PCE -> PCE(%))
            variable_mapping = {
                'PCE': 'PCE(%)',
                'Voc': 'Voc(V)', 
                'Jsc': 'Jsc(mA/cm2)',
                'FF': 'FF(%)',
                'Voc x FF': 'Voc x FF(V%)',
                'R_ser': 'R_series(Ohmcm2)',
                'R_shu': 'R_shunt(Ohmcm2)',
                'V_mpp': 'V_mpp(V)',
                'J_mpp': 'J_mpp(mA/cm2)',
                'P_mpp': 'P_mpp(mW/cm2)'
            }
            
            # Process each boxplot for Excel export
            for plot_type, option1, option2 in plot_selections:
                if plot_type in ['Boxplot', 'Boxplot (omitted)']:  # Only process boxplots for Excel
                    # Extract parameters
                    var_y = option1  # e.g., 'PCE'
                    var_x = option2  # e.g., 'by Variable'
                    name_y = variable_mapping.get(var_y, var_y + '(%)')  # e.g., 'PCE(%)'
                    
                    # Determine grouping column
                    if var_x == 'by Variable':
                        grouping_col = 'condition'
                    elif var_x == 'by Batch':
                        grouping_col = 'batch_for_plotting' if 'batch_for_plotting' in filtered_data.columns else 'batch'
                    elif var_x == 'by Sample':
                        grouping_col = 'sample'
                    elif var_x == 'by Cell':
                        grouping_col = 'cell'
                    elif var_x == 'by Scan Direction':
                        grouping_col = 'direction'
                    else:
                        grouping_col = 'condition'  # default
                    
                    # Calculate statistical summary
                    stats_df = filtered_data.groupby(grouping_col)[name_y].describe()
                    
                    # Get omitted data and filter parameters from data manager
                    omitted_data = self.data_manager.get_omitted_data()
                    filter_params = self.data_manager.get_filter_parameters()
                    
                    # Prepare filtered info (omitted data and filter parameters)
                    filtered_info = (omitted_data if omitted_data is not None else pd.DataFrame(), filter_params)
                    
                    # Save to Excel
                    wb = save_combined_excel_data(
                        "",
                        wb,
                        filtered_data,
                        filtered_info,
                        grouping_col,
                        name_y,
                        var_y,
                        stats_df
                    )
            
            # Use the precisely matched curves data instead of all curves
            filtered_curves = data.get('filtered_curves', pd.DataFrame())
            if filtered_curves.empty:
                # Fallback to creating filtered curves if not available
                filtered_curves = self._create_filtered_curves_data(filtered_data, data["curves"])
            
            # For JV curve plots, we want ALL curves data for complete plots
            all_curves_for_plotting = data["curves"]  # Use complete curves data
            
            # Create properly filtered curves for working/rejected cell plots
            filtered_curves_for_jv_plots = self._create_matching_curves_from_filtered_jv(filtered_data, data["curves"])

            # Prepare data tuple with ALL curves for complete dataset access
            jv_data = (
                filtered_data,           # filtered data for analysis
                data["jvc"],            # complete original data for reference  
                data["curves"]          # CHANGE: Use complete original curves, not filtered_curves
            )
            
            support_data = (
                data.get('junk', pd.DataFrame()),
                self.data_manager.get_filter_parameters(), 
                self.is_conditions, 
                os.getcwd(), 
                filtered_curves["sample"].unique().tolist() if 'sample' in filtered_curves.columns else []
            )
            
            # Create plots using the plot manager - ADD color_scheme parameter:
            figs, names = plotting_string_action(
                plot_selections, 
                jv_data, 
                support_data, 
                is_voila=True,
                color_scheme=selected_colors  # ADD this line
            )
            
            # Store plot data
            self.global_plot_data['figs'] = figs
            self.global_plot_data['names'] = names
            self.global_plot_data['workbook'] = wb
            
            # Display plots using resizable plot utility
            with self.plot_ui.plotted_content:
                clear_output(wait=True)
                ResizablePlotManager.display_plots_resizable(figs, names, self.plot_ui.plotted_content)
                print("Proceed to the next tab to save your results.")
                self._enable_tab(4)
        
        except Exception as e:
            with self.plot_ui.plotted_content:
                clear_output(wait=True)
                display(widgets.HTML(f"""
                <div style="text-align: center; padding: 40px; background-color: #f8d7da; border-radius: 8px;">
                    <div style="font-size: 24px; margin-bottom: 15px;">❌</div>
                    <h3 style="color: #dc3545;">Plot Creation Failed</h3>
                    <p>Error: {str(e)}</p>
                </div>
                """))
            ErrorHandler.handle_plot_error(e, self.plot_ui.plotted_content)

    def _create_matching_curves_from_filtered_jv(self, filtered_jv_data, original_curves_data):
        """Create curves data that exactly matches filtered JV data using sample_id"""
        
        if filtered_jv_data.empty:
            return pd.DataFrame()
        
        # Get unique sample_id + cell + direction + ilum combinations from filtered JV
        filtered_combinations = set()
        for _, row in filtered_jv_data.iterrows():
            combination = (row['sample_id'], row['cell'], row['direction'], row['ilum'])
            filtered_combinations.add(combination)
        
        # Filter curves data to match exactly
        def should_include_curve(curve_row):
            if 'sample_id' not in curve_row:
                return False
            combination = (curve_row['sample_id'], curve_row['cell'], curve_row['direction'], curve_row['ilum'])
            return combination in filtered_combinations
        
        matching_curves = original_curves_data[original_curves_data.apply(should_include_curve, axis=1)].copy()

        return matching_curves
    
    def _create_filtered_curves_data(self, filtered_jv_data, original_curves_data):
        """Create curves data that matches filtered JV data with debugging"""
        
        # Get sample-cell combinations from filtered JV data
        filtered_combinations = set()
        for _, row in filtered_jv_data.iterrows():
            filtered_combinations.add((row['sample'], row['cell']))
        
        # Try exact matching first
        def should_include_curve(row):
            return (row['sample'], row['cell']) in filtered_combinations
        
        filtered_curves = original_curves_data[
            original_curves_data.apply(should_include_curve, axis=1)
        ].copy()
        
        print(f"DEBUG: Exact matching found {len(filtered_curves)} curve records")
        
        # If exact matching fails, try alternative sample name matching
        if len(filtered_curves) == 0:
            print("DEBUG: Trying alternative sample name matching...")
            
            # Extract clean sample names from both datasets for comparison
            jv_clean_samples = set()
            for _, row in filtered_jv_data.iterrows():
                # Extract the core sample name (like C-11)
                clean_name = row['sample'].split('_')[-1] if '_' in row['sample'] else row['sample']
                jv_clean_samples.add((clean_name, row['cell']))
            
            def should_include_curve_alt(row):
                curve_clean = row['sample'].split('_')[-1] if '_' in row['sample'] else row['sample']
                return (curve_clean, row['cell']) in jv_clean_samples
            
            filtered_curves = original_curves_data[
                original_curves_data.apply(should_include_curve_alt, axis=1)
            ].copy()
            
            print(f"DEBUG: Alternative matching found {len(filtered_curves)} curve records")
        
        # Always return a DataFrame, even if empty
        if filtered_curves is None or len(filtered_curves) == 0:
            print("DEBUG: No matching curves found, returning empty DataFrame")
            return original_curves_data.iloc[0:0].copy()  # Return empty DataFrame with same structure
        
        return filtered_curves
    
    def _on_save_plots(self, b):
        """Handle plots saving"""
        if not self.global_plot_data.get('figs'):
            with self.save_ui.download_output:
                print("No plots have been created yet.")
            return
        
        try:
            zip_content = self.save_ui.create_plots_zip(
                self.global_plot_data['figs'], 
                self.global_plot_data['names']
            )
            
            b64 = base64.b64encode(zip_content).decode()
            js_code = f"""
            var link = document.createElement('a');
            link.href = 'data:application/zip;base64,{b64}';
            link.download = 'plots.zip';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            """
            
            with self.save_ui.download_output:
                display(widgets.HTML(f"<button onclick=\"{js_code}\">Click to download all plots</button>"))
                print("Download initiated. If the download doesn't start automatically, click the button above.")
        
        except Exception as e:
            ErrorHandler.log_error("saving plots", e, self.save_ui.download_output)
    
    def _on_save_data(self, b):
        """Handle data saving"""
        if not self.global_plot_data.get('workbook'):
            with self.save_ui.download_output:
                print("No data has been processed yet.")
            return
        
        try:
            excel_buffer = io.BytesIO()
            self.global_plot_data['workbook'].save(excel_buffer)
            excel_buffer.seek(0)
            
            b64 = base64.b64encode(excel_buffer.getvalue()).decode()
            js_code = f"""
            var link = document.createElement('a');
            link.href = 'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}';
            link.download = 'collected_data.xlsx';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            """
            
            with self.save_ui.download_output:
                display(widgets.HTML(f"<button onclick=\"{js_code}\">Click to download data</button>"))
                print("Download initiated. If the download doesn't start automatically, click the button above.")
        
        except Exception as e:
            ErrorHandler.log_error("saving data", e, self.save_ui.download_output)
    
    def _on_save_all(self, b):
        """Handle saving all files"""
        if not self.global_plot_data.get('figs') or not self.global_plot_data.get('workbook'):
            with self.save_ui.download_output:
                print("No plots or data have been created yet.")
            return
        
        try:
            # Create combined zip
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                # Add plots
                for fig, name in zip(self.global_plot_data['figs'], self.global_plot_data['names']):
                    try:
                        html_str = fig.to_html(include_plotlyjs='cdn')
                        zip_file.writestr(name, html_str)
                    except Exception as e:
                        print(f"Error saving {name}: {e}")
                
                # Add Excel file
                excel_buffer = io.BytesIO()
                self.global_plot_data['workbook'].save(excel_buffer)
                excel_buffer.seek(0)
                zip_file.writestr("collected_data.xlsx", excel_buffer.getvalue())
            
            zip_buffer.seek(0)
            b64 = base64.b64encode(zip_buffer.getvalue()).decode()
            js_code = f"""
            var link = document.createElement('a');
            link.href = 'data:application/zip;base64,{b64}';
            link.download = 'results.zip';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            """
            
            with self.save_ui.download_output:
                display(widgets.HTML(f"<button onclick=\"{js_code}\">Click to download all files</button>"))
                print("Download initiated. If the download doesn't start automatically, click the button above.")
        
        except Exception as e:
            ErrorHandler.log_error("saving all files", e, self.save_ui.download_output)
    
    def _enable_tab(self, tab_index):
        """Enable a specific tab (placeholder for future styling)"""
        pass
    
    def get_dashboard(self):
        """Get the main dashboard widget"""
        app_layout = widgets.Layout(
            max_width="1200px",
            margin="0 auto",
            padding='15px'
        )
        
        # Create header with title and info buttons
        header = widgets.HBox([
            widgets.HTML("<h1 style='margin: 0; flex-grow: 1;'>JV Analysis Dashboard</h1>"),
            self.info_ui.get_widget()
        ], layout=widgets.Layout(
            justify_content='space-between',
            align_items='flex-start',
            margin='0 0 20px 0'
        ))
        
        return widgets.VBox([
            header,  # Header with What's New and Manual buttons
            self.auth_ui.get_widget(),
            self.tabs
        ], layout=app_layout)

    def get_current_color_scheme(self):
        """Get currently selected color scheme"""
        if hasattr(self, 'color_selector'):
            return self.color_selector.get_colors()
        else:
            # Default colors if no selector
            return [
                'rgba(93, 164, 214, 0.7)', 'rgba(255, 144, 14, 0.7)', 
                'rgba(44, 160, 101, 0.7)', 'rgba(255, 65, 54, 0.7)', 
                'rgba(207, 114, 255, 0.7)', 'rgba(127, 96, 0, 0.7)',
                'rgba(255, 140, 184, 0.7)', 'rgba(79, 90, 117, 0.7)'
            ]