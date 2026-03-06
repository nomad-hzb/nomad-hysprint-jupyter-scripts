"""
Sample Data Explorer - Main Application Module

This module provides the main interactive data analysis interface for exploring
sample measurement data from NOMAD OASIS. It combines GUI components, data 
management, and plotting capabilities into a cohesive Jupyter-based application.

Key Features:
    - Interactive batch and sample selection
    - Multi-axis data visualization (X, Y, Color)
    - Material and layer type filtering
    - Process step parameter exploration
    - Real-time plot generation with Plotly

Classes:
    SampleDataExplorer: Main application controller coordinating all components

Usage:
    from sample_data_explorer import SampleDataExplorer
    
    analyzer = SampleDataExplorer(url, token)
    analyzer.display()

Author: HySprint Team
"""

from typing import Dict, List, Optional
from IPython.display import clear_output

# Import API functions
from api_calls import (
    get_ids_in_batch, get_sample_description, get_all_eqe, 
    get_processing_steps, get_batch_ids, get_sample_entry_links
)

# Import modules
from data_loader import HySprintDataLoader
from gui_components import GUIManager
from utils import DebugTools, ParameterManager, ProcessStepManager
from data_manager import DataManager
from plot_manager import PlotManager
from natsort import natsorted
import pandas as pd


class SampleDataExplorer:
    """Main analyzer class that coordinates all modules."""
    
    def __init__(self, url: str, token: str):
        """
        Initialize the analyzer.
        
        Args:
            url: API base URL
            token: Authentication token
        """
        self.url = url
        self.token = token
        
        # Initialize managers
        self.gui = GUIManager()
        
        # Setup batch selection using batch_selection.py
        self.gui.setup_batch_selection(url, token, self._on_load_batches)
        
        self.debug = DebugTools()
        self.param_manager = ParameterManager()
        self.process_manager = ProcessStepManager()
        self.data_loader = HySprintDataLoader(url, token, get_all_eqe)
        self.data_manager = DataManager(self.data_loader, self.param_manager, self.debug)
        self.plot_manager = PlotManager(self.gui.plot_widget, self.gui.stats_output)
        
        # Application state
        self.current_batches = []
        self.current_sample_ids = []
        self.current_variation = {}
        self.processing_steps = []
        self.process_display_to_id = {}
        
        # Connect callbacks
        self._connect_callbacks()
    
    def _connect_callbacks(self):
        """Connect GUI callbacks to handler methods."""
        self.gui.connect_callbacks({
            'x_data_source': self._on_x_data_source_selected,
            'y_data_source': self._on_y_data_source_selected,
            'color_data_source': self._on_color_data_source_selected,
            'x_material': self._on_x_material_selected,
            'y_material': self._on_y_material_selected,
            'color_material': self._on_color_material_selected,
            'create_plot': self._on_create_plot,
            'toggle_varying': self._on_toggle_varying_only,
            'download': self._on_download_data,
        })
    
    def _update_status(self, message: str):
        """Update status message."""
        with self.gui.status_output:
            clear_output()
            print(message)
    
    def _initialize_batch_options(self):
        """Initialize batch options from API."""
        try:
            all_batch_ids = get_batch_ids(self.url, self.token)
            self.gui.batch_selector.options = natsorted(all_batch_ids)
            self._update_status("✓ Batch options loaded. Select batches to continue.")
        except Exception as e:
            self._update_status(f"❌ Error loading batches: {str(e)}")
    
    def _filter_batches(self, change):
        """Filter batches based on search term."""
        search_term = change['new'].lower()
        try:
            all_batch_ids = get_batch_ids(self.url, self.token)
            if search_term:
                filtered = [b for b in all_batch_ids if search_term in b.lower()]
            else:
                filtered = all_batch_ids
            self.gui.batch_selector.options = natsorted(filtered)
        except Exception as e:
            self._update_status(f"❌ Error filtering batches: {str(e)}")
    
    def _on_load_batches(self, button):
        """Handle batch loading."""
        if not self.gui.batch_selector.value:
            self._update_status("⚠️ Please select at least one batch.")
            return
        
        # Clear old data
        self.data_manager.current_results = {}
        self.data_manager.current_metadata = {}

        self.gui.plot_widget.data = []
        self.gui.plot_widget.update_layout(title='Select data and click "Create Plot"')
        for sel in [self.gui.x_param_selector, self.gui.y_param_selector, self.gui.color_param_selector]:
            sel.options = []
            sel.disabled = True
        for sel in [self.gui.x_material_selector, self.gui.y_material_selector, self.gui.color_material_selector]:
            sel.options = ['All']
            sel.disabled = True
        
        self._update_status(f"⏳ Loading {len(self.gui.batch_selector.value)} batches...")
        
        try:
            # Get batch data
            self.current_batches = list(self.gui.batch_selector.value)
            self.current_sample_ids = get_ids_in_batch(self.url, self.token, self.current_batches)
            
            if not self.current_sample_ids:
                self._update_status("❌ No samples found in selected batches.")
                return
            
            self.current_variation = get_sample_description(self.url, self.token, self.current_sample_ids)
            self.processing_steps = get_processing_steps(self.url, self.token, self.current_sample_ids)

            self.data_manager.sample_entry_links = get_sample_entry_links(
                self.url, self.token, self.current_sample_ids
            )
            
            # Extract process types
            process_types = self.process_manager.extract_process_types(
                self.processing_steps, self.debug
            )
            
            self.process_display_to_id = {display: original for display, original in process_types}
            display_names = [display for display, _ in process_types]
            
            # Update data source selectors
            all_options = ['Results'] + sorted(display_names)
            
            self.gui.x_data_source_selector.options = all_options
            self.gui.x_data_source_selector.disabled = False
            
            self.gui.y_data_source_selector.options = all_options
            self.gui.y_data_source_selector.disabled = False
            # NOTE: do NOT set .value = 'Results' here — data not loaded yet
            
            self.gui.color_data_source_selector.options = ['None'] + all_options
            self.gui.color_data_source_selector.disabled = False
            self.gui.color_data_source_selector.value = 'None'

            # Load all data for summary FIRST
            self._update_status(f"⏳ Loading all data for summary...")
            self.data_manager.load_all_data_for_summary(self.current_sample_ids, self.current_variation)
            
            # Now set default Y to Results — current_results is populated
            self.gui.y_data_source_selector.value = 'Results'
            
            # Generate parameter summary
            self.data_manager.generate_parameter_summary(self.gui.param_summary_output)
            
            self._update_status(
                f"✓ Loaded {len(self.current_sample_ids)} samples with "
                f"{len(display_names)} process types. Select data sources."
            )
                
        except Exception as e:
            self._update_status(f"❌ Error loading batches: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _on_x_data_source_selected(self, change):
        """Handle X data source selection."""
        if not change['new']:
            return
        self._load_data_for_source(change['new'], 'x')
    
    def _on_y_data_source_selected(self, change):
        """Handle Y data source selection."""
        if not change['new']:
            return
        self._load_data_for_source(change['new'], 'y')
    
    def _on_color_data_source_selected(self, change):
        """Handle Color data source selection."""
        if not change['new'] or change['new'] == 'None':
            return
        self._load_data_for_source(change['new'], 'color')
    
    def _load_data_for_source(self, data_source: str, param_type: str):
        """Load data for a selected data source."""
        print(f"\n[DEBUG] ===== _load_data_for_source START =====")
        self.debug.print_debug(f"data_source={data_source}, param_type={param_type}")
        self._update_status(f"⏳ Loading data for {param_type.upper()} from: {data_source}...")

        
        try:
            if data_source == 'Results':
                self.debug.print_debug(f"Results branch selected")
                # Load results if not already loaded
                if not self.data_manager.current_results:
                    self.debug.print_debug(f"Loading results from API...")
                    from api_calls import get_all_eqe
                    self.data_manager.current_results = {}
                    
                    # Map measurement types to their data extraction keys
                    measurement_configs = {
                        'jv_measurement': {
                            'api_type': 'HySprint_JVmeasurement',
                            'data_key': 'jv_curve'  # Extract from jv_curve array
                        },
                        'eqe_measurement': {
                            'api_type': 'HySprint_EQEmeasurement',
                            'data_key': 'eqe_data'
                        },
                        'mpp_tracking': {
                            'api_type': 'HySprint_MPPTracking',
                            'data_key': 'properties'
                        },
                        'simple_mpp_tracking': {
                            'api_type': 'HySprint_SimpleMPPTracking',
                            'data_key': 'properties'
                        },
                        'pl_measurement': {
                            'api_type': 'HySprint_PLmeasurement',
                            'data_key': None  # Use top-level data
                        },
                        'trpl_measurement': {
                            'api_type': 'HySprint_TimeResolvedPhotoluminescence',
                            'data_key': None
                        },
                        'abspl_measurement': {
                            'api_type': 'HySprint_AbsPLMeasurement',
                            'data_key': 'results'  # HySprint_AbsPLResult
                        },
                        'pl_imaging': {
                            'api_type': 'HySprint_PLImaging',
                            'data_key': None
                        },
                        'sem': {
                            'api_type': 'HySprint_SEM',
                            'data_key': None
                        },
                        'uvvis_measurement': {
                            'api_type': 'HySprint_UVvismeasurement',
                            'data_key': None
                        },
                        'pes': {
                            'api_type': 'HySprint_PES',
                            'data_key': None
                        },
                        'cyclic_voltammetry': {
                            'api_type': 'HySprint_CyclicVoltammetry',
                            'data_key': 'properties'
                        },
                        'eis': {
                            'api_type': 'HySprint_ElectrochemicalImpedanceSpectroscopy',
                            'data_key': None
                        },
                        'trspv_measurement': {
                            'api_type': 'HySprint_trSPVmeasurement',
                            'data_key': None
                        },
                        'nmr': {
                            'api_type': 'HySprint_Simple_NMR',
                            'data_key': 'data'
                        },
                    }
                    
                    for measurement_key, config in measurement_configs.items():
                        try:
                            data = get_all_eqe(self.url, self.token, self.current_sample_ids, config['api_type'])
                            if data is not None and isinstance(data, dict) and data:
                                rows = []
                                for sample_id, measurements in data.items():
                                    if measurements and len(measurements) > 0:
                                        measurement_data = measurements[0][0]  # First tuple, first element
                                        
                                        # Extract data based on measurement type
                                        data_key = config['data_key']
                                        
                                        if data_key and data_key in measurement_data:
                                            # Extract from specific key
                                            extracted_data = measurement_data[data_key]
                                            
                                            # Handle different data structures
                                            if isinstance(extracted_data, list) and extracted_data:
                                                # For jv_curve: it's a list of dicts, use first or aggregate
                                                if isinstance(extracted_data[0], dict):
                                                    result_data = extracted_data[0].copy()
                                                else:
                                                    result_data = {data_key: extracted_data}
                                            elif isinstance(extracted_data, dict):
                                                result_data = extracted_data.copy()
                                            else:
                                                result_data = {data_key: extracted_data}
                                            
                                            # ADD TOP-LEVEL FIELDS (datetime, name, description, etc.)
                                            # These are at the measurement level, not inside jv_curve
                                            top_level_fields = ['datetime', 'name', 'description', 'data_file', 'lab_id']
                                            for field in top_level_fields:
                                                if field in measurement_data and field not in result_data:
                                                    result_data[field] = measurement_data[field]
                                            
                                            result_data['sample_id'] = sample_id
                                            rows.append(result_data)
                                        else:
                                            # Use top-level measurement data
                                            measurement_data['sample_id'] = sample_id
                                            rows.append(measurement_data)
                                
                                if rows:
                                    df = pd.DataFrame(rows)
                                    self.data_manager.current_results[measurement_key] = df
                                    self.debug.print_debug(f"Loaded {measurement_key}: {len(df)} rows, {len(df.columns)} columns")
                        except Exception as e:
                            self.debug.print_debug(f"Error loading {measurement_key}: {e}")
                            pass
                else:
                    self.debug.print_debug(f"Results already loaded, {len(self.data_manager.current_results)} measurement types")
                
                # Show measurement types in material selector
                measurement_types_list = list(self.data_manager.current_results.keys())
                self.debug.print_debug(f"Available measurement types: {measurement_types_list}")
                
                measurement_display_names = []
                for mt in measurement_types_list:
                    if mt == 'jv_measurement':
                        measurement_display_names.append('JV')
                    elif mt == 'eqe_measurement':
                        measurement_display_names.append('EQE')
                    elif mt == 'mpp_tracking':
                        measurement_display_names.append('MPP')
                    elif mt == 'simple_mpp_tracking':
                        measurement_display_names.append('Simple MPP')
                    elif mt == 'pl_measurement':
                        measurement_display_names.append('PL')
                    elif mt == 'trpl_measurement':
                        measurement_display_names.append('TRPL')
                    elif mt == 'abspl_measurement':
                        measurement_display_names.append('AbsPL')
                    elif mt == 'pl_imaging':
                        measurement_display_names.append('PL Imaging')
                    elif mt == 'sem':
                        measurement_display_names.append('SEM')
                    elif mt == 'uvvis_measurement':
                        measurement_display_names.append('UV-Vis')
                    elif mt == 'pes':
                        measurement_display_names.append('PES')
                    elif mt == 'cyclic_voltammetry':
                        measurement_display_names.append('CV')
                    elif mt == 'eis':
                        measurement_display_names.append('EIS')
                    elif mt == 'trspv_measurement':
                        measurement_display_names.append('trSPV')
                    elif mt == 'nmr':
                        measurement_display_names.append('NMR')
                    else:
                        measurement_display_names.append(mt.replace('_', ' ').title())
                
                self.debug.print_debug(f"Display names: {measurement_display_names}")
                self.debug.print_debug(f"About to call _set_material_selector_options with {len(sorted(measurement_display_names))} options")
                self._set_material_selector_options(param_type, sorted(measurement_display_names), data_source)

                if sorted(measurement_display_names):
                    first_measurement = sorted(measurement_display_names)[0]
                    self.debug.print_debug(f"Auto-selecting first measurement type: {first_measurement}")
                    if param_type == "x":
                        self.gui.x_material_selector.value = first_measurement
                    elif param_type == "y":
                        self.gui.y_material_selector.value = first_measurement
                    else:
                        self.gui.color_material_selector.value = first_measurement
                else:
                    # Fallback: show all parameters
                    params = self.data_manager.get_parameter_options(
                        self.data_manager.current_results,
                        f"{param_type}_parameters",
                        is_results=True
                    )
                    self._update_parameter_selector(param_type, params, is_results=True)
            
            else:
                # Process step data source
                self.debug.print_debug(f"Process step branch: {data_source}")
                self.debug.print_debug(f"About to load metadata for {data_source}")
                
                metadata_df = self.data_manager.load_data_for_source(
                    data_source, self.current_sample_ids, 
                    self.current_variation, self.process_manager
                )
                self.debug.print_debug(f"Metadata loaded, shape: {metadata_df.shape if metadata_df is not None else 'None'}")
                
                # Extract materials
                materials = self.data_manager.extract_materials(metadata_df, data_source)
                self.debug.print_debug(f"Materials extracted: {materials}")
                self._set_material_selector_options(param_type, materials, data_source)
                
                # Update parameter options
                self.debug.print_debug(f"About to get parameter options for process step")
                params = self.data_manager.get_parameter_options(
                    {data_source: metadata_df},
                    f'{param_type}_parameters',
                    is_results=False
                )
                self.debug.print_debug(f"Got {len(params)} parameters: {params[:5] if params else []}")
                self._update_parameter_selector(param_type, params, is_results=False)
            
            # Rebuild merged data
            self.debug.print_debug(f"About to rebuild merged data")
            self._rebuild_merged_data()
            self.debug.print_debug(f"===== _load_data_for_source END =====\n")
                    
        except Exception as e:
            self._update_status(f"❌ Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
        
    def _set_material_selector_options(self, param_type: str, materials: list, data_source: str):
        """Set material selector options and immediately populate parameters."""
        print(f"\n[DEBUG] === _set_material_selector_options START ===")
        print(f"  param_type={param_type}")
        print(f"  data_source={data_source}")
        print(f"  materials={materials}")
        print(f"  materials empty? {not materials}")
        self.debug.print_debug(f"param_type={param_type}, data_source={data_source}, materials={materials}")
    
        if param_type == 'x':
            selector = self.gui.x_material_selector
        elif param_type == 'y':
            selector = self.gui.y_material_selector
        else:
            selector = self.gui.color_material_selector
    
        if materials:
            if data_source == 'Results':
                # For Results: materials list = measurement display names (JV, EQE, etc.)
                selector.options = materials
                selector.value = materials[0]
                selector.disabled = False
                if param_type == 'x':
                    self.data_manager._x_material_is_all = False
                elif param_type == 'y':
                    self.data_manager._y_material_is_all = False
                else:
                    self.data_manager._color_material_is_all = False
    
                # Directly populate parameters — do NOT rely on observer firing,
                # because value may already equal materials[0] from a previous run.
                self.debug.print_debug(f"Results: directly calling _filter_results_parameters({param_type}, {materials[0]})")
                self._filter_results_parameters(param_type, materials[0])
    
            else:
                # For process steps: add 'All' option
                selector.options = ['All'] + materials
                selector.value = 'All'
                selector.disabled = False
                if param_type == 'x':
                    self.data_manager._x_material_is_all = True
                elif param_type == 'y':
                    self.data_manager._y_material_is_all = True
                else:
                    self.data_manager._color_material_is_all = True
    
                # Populate parameters with 'All' selected (includes Material Type option)
                from utils import ProcessStepManager
                pm = ProcessStepManager()
                measurement_type = pm.map_display_to_measurement_type(data_source)
                if measurement_type and measurement_type in self.data_manager.current_metadata:
                    metadata_df = self.data_manager.current_metadata[measurement_type]
                    params = self.data_manager.get_parameter_options(
                        {data_source: metadata_df},
                        f'{param_type}_parameters',
                        is_results=False
                    )
                    self._update_parameter_selector(param_type, params, is_results=False)
        else:
            selector.options = ['All']
            selector.value = 'All'
            selector.disabled = True
            if param_type == 'x':
                self.data_manager._x_material_is_all = True
            elif param_type == 'y':
                self.data_manager._y_material_is_all = True
            else:
                self.data_manager._color_material_is_all = True
    
        print(f"  selector.disabled={selector.disabled}")
        print(f"  selector.options={list(selector.options)[:5]}")
        print(f"=== _set_material_selector_options END ===\n")
    
    def _update_parameter_selector(self, param_type: str, params: List[str], is_results: bool):
        """Update parameter selector with available parameters."""
        self.debug.print_if_enabled('show_parameter_flow',
            f"\n=== UPDATE DROPDOWN ===\n"
            f"param_type: {param_type}\n"
            f"is_results: {is_results}\n"
            f"Params count: {len(params)}\n"
            f"Sample params: {params[:15]}\n"
            f"'datetime (JV)' in params: {'datetime (JV)' in params}")
        print(f"\n[DEBUG] === _update_parameter_selector START ===")
        self.debug.print_debug(f"param_type={param_type}, params count={len(params)}, is_results={is_results}")
        self.debug.print_debug(f"Parameters (first 10): {params[:10] if len(params) > 10 else params}")
        
        if param_type == 'x':
            selector = self.gui.x_param_selector
        elif param_type == 'y':
            selector = self.gui.y_param_selector
        else:
            selector = self.gui.color_param_selector
        
        self.debug.print_debug(f"Current selector state: disabled={selector.disabled}, options count={len(selector.options)}")
        
        selector.options = params if param_type != 'color' else ['None'] + params
        selector.disabled = False
        
        if params:
            if param_type == 'x':
                selector.value = params[0]
            elif param_type == 'y':
                eff = next((p for p in params if 'efficiency' in p.lower()), None)
                selector.value = eff if eff else params[0]
            else:  # color
                selector.value = 'None'
        
        self.debug.print_debug(f"After setting: disabled={selector.disabled}, value={selector.value}, options count={len(selector.options)}")
        self.debug.print_debug(f"=== _update_parameter_selector END ===\n")
    
    def _on_x_material_selected(self, change):
        """Handle X material selection."""
        print(f"\n[DEBUG] === _on_x_material_selected ===")
        self.debug.print_debug(f"new value: {change['new']}")
        self.debug.print_debug(f"data source: {self.gui.x_data_source_selector.value}")
        
        if not change['new']:
            return
        
        # ISSUE 3 FIX: Set flag for "All" selection
        self.data_manager._x_material_is_all = (change['new'] == 'All')
        
        # If Results data source, filter parameters by selected measurement type
        if self.gui.x_data_source_selector.value == 'Results':
            self._filter_results_parameters('x', change['new'])
        
        self._rebuild_merged_data()
    
    def _on_y_material_selected(self, change):
        """Handle Y material selection."""
        print(f"\n[DEBUG] === _on_y_material_selected ===")
        self.debug.print_debug(f"new value: {change['new']}")
        self.debug.print_debug(f"data source: {self.gui.y_data_source_selector.value}")
        
        if not change['new']:
            return
        
        # ISSUE 3 FIX: Set flag for "All" selection
        self.data_manager._y_material_is_all = (change['new'] == 'All')
        
        # If Results data source, filter parameters by selected measurement type
        if self.gui.y_data_source_selector.value == 'Results':
            self._filter_results_parameters('y', change['new'])
        
        self._rebuild_merged_data()
    
    def _on_color_material_selected(self, change):
        """Handle Color material selection."""
        print(f"\n[DEBUG] === _on_color_material_selected ===")
        self.debug.print_debug(f"new value: {change['new']}")
        self.debug.print_debug(f"data source: {self.gui.color_data_source_selector.value}")
        
        if not change['new']:
            return
        
        # ISSUE 3 FIX: Set flag for "All" selection
        self.data_manager._color_material_is_all = (change['new'] == 'All')
        
        # If Results data source, filter parameters by selected measurement type
        if self.gui.color_data_source_selector.value == 'Results':
            self._filter_results_parameters('color', change['new'])
        
        self._rebuild_merged_data()

    def _filter_results_parameters(self, param_type: str, measurement_display_name: str):
        """Filter parameters to show only those from the selected measurement type."""
        print(f"\n[DEBUG] === _filter_results_parameters START ===")
        print(f"[DEBUG]   param_type={param_type}, measurement_display_name={measurement_display_name}")
        print(f"[DEBUG]   current_results keys: {list(self.data_manager.current_results.keys())}")
    
        display_to_key = {
            'JV': 'jv_measurement',
            'EQE': 'eqe_measurement',
            'MPP': 'mpp_tracking',
            'Simple MPP': 'simple_mpp_tracking',
            'PL': 'pl_measurement',
            'TRPL': 'trpl_measurement',
            'AbsPL': 'abspl_measurement',
            'PL Imaging': 'pl_imaging',
            'SEM': 'sem',
            'UV-Vis': 'uvvis_measurement',
            'PES': 'pes',
            'CV': 'cyclic_voltammetry',
            'EIS': 'eis',
            'trSPV': 'trspv_measurement',
            'NMR': 'nmr',
        }
    
        measurement_key = display_to_key.get(measurement_display_name)
        print(f"[DEBUG]   measurement_key resolved to: {measurement_key}")
    
        if measurement_key and measurement_key in self.data_manager.current_results:
            df = self.data_manager.current_results[measurement_key]
            print(f"[DEBUG]   DataFrame shape: {df.shape}")
            print(f"[DEBUG]   DataFrame columns: {list(df.columns)}")
            print(f"[DEBUG]   DataFrame head:\n{df.head(2).to_string()}")
            single_measurement = {measurement_key: df}
            params = self.data_manager.get_parameter_options(
                single_measurement,
                f'{param_type}_parameters',
                is_results=True
            )
        else:
            print(f"[DEBUG]   Measurement key not found in current_results — showing all")
            params = self.data_manager.get_parameter_options(
                self.data_manager.current_results,
                f'{param_type}_parameters',
                is_results=True
            )
    
        print(f"[DEBUG]   Final params count: {len(params)}")
        print(f"[DEBUG]   Final params: {params}")
        print(f"[DEBUG] === _filter_results_parameters END ===\n")
        self._update_parameter_selector(param_type, params, is_results=True)
    
    def _rebuild_merged_data(self):
        """Rebuild merged data with current selections."""
        invalid_samples = self.data_manager.rebuild_merged_data(
            self.gui.x_data_source_selector.value,
            self.gui.y_data_source_selector.value,
            self.gui.color_data_source_selector.value,
            self.gui.x_material_selector.value,
            self.gui.y_material_selector.value,
            self.gui.color_material_selector.value,
            group_by_subbatch=self.gui.group_by_subbatch.value
        )
        
        # Show warning if invalid sample IDs were found
        if invalid_samples:
            with self.gui.status_output:
                print(f"⚠️  {len(invalid_samples)} samples excluded (non-standard ID pattern)")
        
        # Update parameter summary
        self.data_manager.generate_parameter_summary(self.gui.param_summary_output)
    
    def _on_toggle_varying_only(self, change):
        """Handle toggle for showing only varying parameters."""
        if self.data_manager.merged_data is None:
            return
        
        if change['new']:
            self._apply_varying_filter()
            self.gui.show_varying_only.button_style = 'success'
            self.gui.show_varying_only.icon = 'check'
        else:
            # Reload full parameter lists
            self._reload_all_parameters()
            self.gui.show_varying_only.button_style = ''
            self.gui.show_varying_only.icon = ''
    
    def _apply_varying_filter(self):
        """Filter to show only varying parameters."""
        if self.data_manager.merged_data is None:
            return
        
        current_x = list(self.gui.x_param_selector.options)
        current_y = list(self.gui.y_param_selector.options)
        current_color = [c for c in self.gui.color_param_selector.options if c != 'None']
        
        varying_x = self.param_manager.filter_to_varying_only(current_x, self.data_manager.merged_data)
        varying_y = self.param_manager.filter_to_varying_only(current_y, self.data_manager.merged_data)
        varying_color = self.param_manager.filter_to_varying_only(current_color, self.data_manager.merged_data)
        
        self.gui.x_param_selector.options = varying_x
        self.gui.y_param_selector.options = varying_y
        self.gui.color_param_selector.options = ['None'] + varying_color
        
        # Update values if needed
        if self.gui.x_param_selector.value not in varying_x and varying_x:
            self.gui.x_param_selector.value = varying_x[0]
        if self.gui.y_param_selector.value not in varying_y and varying_y:
            self.gui.y_param_selector.value = varying_y[0]
        if self.gui.color_param_selector.value not in ['None'] + varying_color:
            self.gui.color_param_selector.value = 'None'
    
    def _reload_all_parameters(self):
        """Reload full parameter lists (undo varying filter)."""
        # Trigger reload by re-selecting current data sources
        if self.gui.x_data_source_selector.value:
            self._load_data_for_source(self.gui.x_data_source_selector.value, 'x')
        if self.gui.y_data_source_selector.value:
            self._load_data_for_source(self.gui.y_data_source_selector.value, 'y')
        if self.gui.color_data_source_selector.value and self.gui.color_data_source_selector.value != 'None':
            self._load_data_for_source(self.gui.color_data_source_selector.value, 'color')
    
    def _on_create_plot(self, button):
        """Handle plot creation."""
        if self.data_manager.merged_data is None or self.data_manager.merged_data.empty:
            self._update_status("❌ No data available for plotting.")
            return
        
        x_param = self.gui.x_param_selector.value
        y_param = self.gui.y_param_selector.value
        color_param = self.gui.color_param_selector.value
        aggregation = self.gui.jv_aggregation_selector.value
        
        if not x_param or not y_param:
            self._update_status("⚠️ Please select both X and Y parameters.")
            return
        
        try:
            # Prepare plot data
            plot_type = self.gui.plot_type_selector.value
            plot_df, x_col, y_col, color_col = self.plot_manager.prepare_plot_data(
                self.data_manager.merged_data, x_param, y_param, color_param, aggregation,
                plot_type=plot_type,
                color_data_source=self.gui.color_data_source_selector.value,
                source_material_columns=getattr(self.data_manager, '_source_material_columns', {})
            )
            
            if plot_df.empty:
                self._update_status("❌ No valid data points for selected parameters.")
                return
            
            # Create plot
            color_col = None
            if color_param and color_param != 'None':
                # ISSUE 3 FIX: Handle "Material Type" for color
                if color_param == "Material Type":
                    if 'layer_material_name' in plot_df.columns:
                        color_col = 'layer_material_name'
                    elif 'layer_material' in plot_df.columns:
                        color_col = 'layer_material'
                    else:
                        material_cols = [col for col in plot_df.columns if 'material' in col.lower() and 'layer' in col.lower()]
                        if material_cols:
                            color_col = material_cols[0]
                else:
                    color_col = self.plot_manager.extract_column_name(color_param)
            
            plot_type = self.gui.plot_type_selector.value
            colorscale = self.gui.colorscale_selector.value
            
            boxplot_warning = ""
            if plot_type == 'Boxplot' and pd.api.types.is_numeric_dtype(plot_df[x_col]):
                boxplot_warning = (
                    f"\n⚠️  Boxplot requires a categorical X axis.\n"
                    f"   '{x_param}' is numeric — falling back to Scatter.\n"
                    f"   Suggestion: use 'Material Type' or a categorical process\n"
                    f"   parameter as X for a meaningful boxplot."
                )
                self.plot_manager.create_scatter_plot(
                    plot_df, x_col, y_col, color_col, x_param, y_param, colorscale
                )
                self.plot_manager.register_click_handler(
                    self.data_manager.sample_entry_links,
                    self.gui.click_output
                )
            elif plot_type == 'Boxplot':
                self.plot_manager.create_box_plot(
                    plot_df, x_col, y_col, color_col, x_param, y_param, colorscale
                )
                self.plot_manager.register_click_handler(
                    self.data_manager.sample_entry_links,
                    self.gui.click_output
                )
            else:
                self.plot_manager.create_scatter_plot(
                    plot_df, x_col, y_col, color_col, x_param, y_param, colorscale
                )
                self.plot_manager.register_click_handler(
                    self.data_manager.sample_entry_links,
                    self.gui.click_output
                )
            
            # Display statistics
            self.plot_manager.display_statistics(plot_df, x_col, y_col, x_param, y_param)

            # Add subbatch grouping info to status
            status_msg = f"✓ Plot created with {len(plot_df)} data points!"
            if self.gui.group_by_subbatch.value:
                status_msg += " (grouped by subbatch)"
                
            # Debug output
            if self.gui.debug_checkbox.value:
                with self.gui.status_output:
                    clear_output()
                    print(f"✓ Plot created successfully!{boxplot_warning}\n\nDEBUG INFO:\n...")
                    print(f"\nDEBUG INFO:")
                    print(f"Data shape: {plot_df.shape}")
                    print(f"X: {x_col}, Y: {y_col}")
                    print(plot_df.head())
            else:
                self._update_status(status_msg)  # Instead of the hardcoded message
            
        except Exception as e:
            self._update_status(f"❌ Error creating plot: {str(e)}")
            import traceback
            traceback.print_exc()

    def _on_download_data(self, button):
        """Download the currently plotted data as CSV via browser."""
        with self.gui.download_output:
            from IPython.display import clear_output, Javascript, display as ipy_display
            clear_output()
    
            if self.data_manager.merged_data is None or self.data_manager.merged_data.empty:
                print("⚠️ No data loaded to download.")
                return
    
            x_param = self.gui.x_param_selector.value
            y_param = self.gui.y_param_selector.value
            color_param = self.gui.color_param_selector.value
            aggregation = self.gui.jv_aggregation_selector.value
    
            if not x_param or not y_param:
                print("⚠️ Please select X and Y parameters first.")
                return
    
            try:
                plot_df, x_col, y_col, _ = self.plot_manager.prepare_plot_data(
                    self.data_manager.merged_data, x_param, y_param, color_param, aggregation,
                    plot_type=self.gui.plot_type_selector.value
                )
                
                from datetime import datetime
                import base64
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"plot_data_{timestamp}.csv"
                
                # Convert to CSV and encode
                csv_string = plot_df.to_csv(index=False)
                b64 = base64.b64encode(csv_string.encode()).decode()
                
                # JavaScript to trigger browser download
                js_code = f"""
                (function() {{
                    var csvContent = atob('{b64}');
                    var blob = new Blob([csvContent], {{ type: 'text/csv;charset=utf-8;' }});
                    var link = document.createElement('a');
                    var url = URL.createObjectURL(blob);
                    link.setAttribute('href', url);
                    link.setAttribute('download', '{filename}');
                    link.style.visibility = 'hidden';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                }})();
                """
                
                print(f"✓ Downloading {filename} ({len(plot_df)} rows, {len(plot_df.columns)} columns)...")
                ipy_display(Javascript(js_code))
                
            except Exception as e:
                print(f"❌ Download error: {e}")
                import traceback
                traceback.print_exc()
    
    def create_interface(self):
        """Create and return the complete interface."""
        return self.gui.create_layout()
    
    def display(self):
        """Display the complete interface in Jupyter notebook."""
        from IPython.display import display
        interface = self.create_interface()
        display(interface)
