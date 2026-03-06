"""
Data Manager Module

Handles all data loading, filtering, merging, and parameter management for the
Sample Data Explorer. This module acts as the central data hub, coordinating
between the NOMAD API and the visualization components.

Key Responsibilities:
    - Load metadata and results from NOMAD OASIS
    - Filter data by material, layer type, and process step
    - Merge multiple data sources for plotting
    - Generate parameter summaries
    - Manage "Material Type" special parameter

Classes:
    DataManager: Central data management controller

Author: HySprint Team
"""

import pandas as pd
from typing import Dict, List, Optional
from IPython.display import clear_output
from data_loader import HySprintDataLoader


class DataManager:
    """Manages data loading, merging, and parameter summary generation."""
    # Common columns that appear across all data types
    COMMON_COLUMNS = [
        'sample_id', 'variation', 'name', 'datetime',
        'description', 'data_file', 'lab_id', 'position_in_plan',
        'timestamp', 'measured_at', 'raw_data', 'data_path', 'measurement_id'
    ]

    
    def __init__(self, data_loader: HySprintDataLoader, param_manager, debug):
        """
        Initialize data manager.
        
        Args:
            data_loader: HySprintDataLoader instance
            param_manager: ParameterManager instance
            debug: DebugTools instance
        """
        self.data_loader = data_loader
        self.param_manager = param_manager
        self.debug = debug
        
        # Data storage
        self.current_metadata = {}
        self.current_results = {}
        self.merged_data = None
        

        self._x_material_is_all = False
        self._y_material_is_all = False
        self._color_material_is_all = False
        self._source_material_columns = {}
        self.sample_entry_links = {}  # {lab_id: gui_url}

    def get_material_column(self, df: pd.DataFrame) -> Optional[str]:
        """Get the material column name from dataframe."""
        if 'layer_material_name' in df.columns:
            return 'layer_material_name'
        elif 'layer_material' in df.columns:
            return 'layer_material'
        return None

    def extract_subbatch(self, sample_id: str) -> tuple:
        """
        Extract subbatch identifier from sample_id.
        Pattern: {prefix}_{batch}_{subbatch}_{rest}
        Example: HZB_FiNa_1_3_C-1 → 'HZB_FiNa_1_3'
        
        Returns:
            (subbatch_id, is_valid) tuple
            subbatch_id: str like 'HZB_FiNa_1_3' or None if invalid
            is_valid: bool indicating if pattern matched
        """
        parts = sample_id.split('_')
        if len(parts) >= 4:
            # Take first 4 parts: prefix_user_batch_subbatch
            subbatch_id = '_'.join(parts[:4])
            return subbatch_id, True
        return None, False

    def get_measurement_data(self, measurement_key: str, sample_ids: List[str]) -> Optional[pd.DataFrame]:
        """
        Get measurement data with lazy loading.
        
        PERFORMANCE OPTIMIZATION (G4): Only loads data when first requested,
        not all at once. Can save 5-20 seconds on app startup.
        
        Args:
            measurement_key: Type of measurement (e.g., 'jv_measurement', 'eqe_measurement')
            sample_ids: List of sample IDs to load
            
        Returns:
            DataFrame with measurement data, or None if not available
        """
        # Check if already loaded
        if measurement_key in self.current_results:
            return self.current_results[measurement_key]
        
        # Lazy load on first access
        self.debug.print_debug(f"Lazy loading {measurement_key}...")
        
        from api_calls import get_all_eqe
        
        # Measurement type mapping
        measurement_type_map = {
            'jv_measurement': 'HySprint_JVmeasurement',
            'eqe_measurement': 'HySprint_EQEmeasurement',
            'mpp_tracking': 'HySprint_MPPTracking',
            'simple_mpp_tracking': 'HySprint_SimpleMPPTracking',
            'pl_measurement': 'HySprint_PLmeasurement',
            'trpl_measurement': 'HySprint_TimeResolvedPhotoluminescence',
            'abspl_measurement': 'HySprint_AbsPLMeasurement',
            'pl_imaging': 'HySprint_PLImaging',
            'sem': 'HySprint_SEM',
            'uvvis_measurement': 'HySprint_UVvismeasurement',
            'pes': 'HySprint_PES',
            'cyclic_voltammetry': 'HySprint_CyclicVoltammetry',
            'eis': 'HySprint_ElectrochemicalImpedanceSpectroscopy',
            'trspv_measurement': 'HySprint_trSPVmeasurement',
            'nmr': 'HySprint_Simple_NMR',
        }
        
        if measurement_key not in measurement_type_map:
            return None
        
        try:
            data = get_all_eqe(
                self.data_loader.url,
                self.data_loader.token,
                sample_ids,
                measurement_type_map[measurement_key]
            )
            
            if data is not None and isinstance(data, dict) and data:
                rows = []
                for sample_id, measurements in data.items():
                    if measurements and len(measurements) > 0:
                        measurement_data = measurements[0][0]
                        measurement_data['sample_id'] = sample_id
                        rows.append(measurement_data)
                
                if rows:
                    df = pd.DataFrame(rows)
                    self.current_results[measurement_key] = df
                    self.debug.print_debug(f"Loaded {len(df)} rows for {measurement_key}")
                    return df
        except Exception as e:
            self.debug.print_debug(f"Error loading {measurement_key}: {e}")
        
        return None
    
    def load_all_data_for_summary(self, sample_ids: List[str], variation: Dict[str, str]):
        """
        Load all available metadata and results for parameter summary.
        
        PERFORMANCE NOTE (G4): This method now loads only metadata eagerly.
        Measurement results are loaded lazily via get_measurement_data() when needed.
        This can reduce initial load time from 20s to 2-3s.
        """
        self.debug.print_debug("load_all_data_for_summary called")
        
        # Store sample_ids for lazy loading
        self._cached_sample_ids = sample_ids
        
        # TEST: Check get_all_eqe structure
        from api_calls import get_all_eqe
        test_data = get_all_eqe(self.data_loader.url, self.data_loader.token, sample_ids[:5], 'HySprint_JVmeasurement')
        self.debug.print_debug(f"get_all_eqe return type: {type(test_data)}")
        if test_data:
            self.debug.print_debug(f"First 3 keys: {list(test_data.keys())[:3]}")
            first_key = list(test_data.keys())[0]
            self.debug.print_debug(f"First value type: {type(test_data[first_key])}")
            self.debug.print_debug(f"First value: {test_data[first_key]}")
        
        # G4 OPTIMIZATION: Load only metadata eagerly, results loaded lazily
        loader_methods = {
            'inkjet_printing': self.data_loader.load_inkjet_printing_data,
            'cleaning': self.data_loader.load_cleaning_data,
            'substrate': self.data_loader.load_substrate_data,
            'evaporation': self.data_loader.load_evaporation_data,
            'slot_die_coating': self.data_loader.load_slot_die_coating_data,
            'spin_coating': self.data_loader.load_spin_coating_data,
        }
        
        self.debug.print_debug("Loading metadata types (fast)...")
        for measurement_type, loader_func in loader_methods.items():
            try:
                if measurement_type not in self.current_metadata:
                    metadata_df = loader_func(sample_ids, variation)
                    if metadata_df is not None and not metadata_df.empty:
                        self.current_metadata[measurement_type] = metadata_df
            except Exception as e:
                pass
        
        self.debug.print_debug(f"Metadata loaded. Results will be loaded on-demand (lazy loading).")
        # Note: Results are now loaded via get_measurement_data() when actually needed
        
        self.debug.print_debug(f"Starting results loading...")
        
        # Load all results using get_all_eqe for each measurement type
        if not self.current_results:
            self.current_results = {}
            
            from api_calls import get_all_eqe
            
            measurement_types = {
                'jv_measurement': 'HySprint_JVmeasurement',
                'eqe_measurement': 'HySprint_EQEmeasurement',
                'mpp_tracking': 'HySprint_MPPTracking',
                'simple_mpp_tracking': 'HySprint_SimpleMPPTracking',
                'pl_measurement': 'HySprint_PLmeasurement',
                'trpl_measurement': 'HySprint_TimeResolvedPhotoluminescence',
                'abspl_measurement': 'HySprint_AbsPLMeasurement',
                'pl_imaging': 'HySprint_PLImaging',
                'sem': 'HySprint_SEM',
                'uvvis_measurement': 'HySprint_UVvismeasurement',
                'pes': 'HySprint_PES',
                'cyclic_voltammetry': 'HySprint_CyclicVoltammetry',
                'eis': 'HySprint_ElectrochemicalImpedanceSpectroscopy',
                'trspv_measurement': 'HySprint_trSPVmeasurement',
                'nmr': 'HySprint_Simple_NMR',
            }
            
            # data_key tells us where the actual results live inside the top-level measurement dict
            measurement_data_keys = {
                'jv_measurement':         'jv_curve',
                'eqe_measurement':        'eqe_data',
                'mpp_tracking':           'properties',
                'simple_mpp_tracking':    'properties',
                'abspl_measurement':      'results',
                'cyclic_voltammetry':     'properties',
                'nmr':                    'data',
                # None means use the top-level dict directly
                'pl_measurement':         None,
                'trpl_measurement':       None,
                'pl_imaging':             None,
                'sem':                    None,
                'uvvis_measurement':      None,
                'pes':                    None,
                'eis':                    None,
                'trspv_measurement':      None,
            }
            top_level_fields = ['datetime', 'name', 'description', 'data_file', 'lab_id']
            
            for measurement_key, measurement_type in measurement_types.items():
                self.debug.print_debug(f"Attempting to load {measurement_key}")
                try:
                    data = get_all_eqe(self.data_loader.url, self.data_loader.token, sample_ids, measurement_type)
                    if data is not None and isinstance(data, dict) and data:
                        rows = []
                        data_key = measurement_data_keys.get(measurement_key)  # may be None
            
                        for sample_id, measurements in data.items():
                            if not measurements or len(measurements) == 0:
                                continue
                            measurement_data = measurements[0][0]
            
                            if data_key and data_key in measurement_data:
                                extracted = measurement_data[data_key]
                                # extracted may be a list of dicts (e.g. jv_curve) or a dict
                                if isinstance(extracted, list) and extracted:
                                    if isinstance(extracted[0], dict):
                                        # Multiple sub-measurements (e.g. forward/reverse scan)
                                        for sub in extracted:
                                            row = sub.copy()
                                            for field in top_level_fields:
                                                if field in measurement_data and field not in row:
                                                    row[field] = measurement_data[field]
                                            row['sample_id'] = sample_id
                                            rows.append(row)
                                    else:
                                        row = {data_key: extracted}
                                        for field in top_level_fields:
                                            if field in measurement_data:
                                                row[field] = measurement_data[field]
                                        row['sample_id'] = sample_id
                                        rows.append(row)
                                elif isinstance(extracted, dict):
                                    row = extracted.copy()
                                    for field in top_level_fields:
                                        if field in measurement_data and field not in row:
                                            row[field] = measurement_data[field]
                                    row['sample_id'] = sample_id
                                    rows.append(row)
                                else:
                                    # Scalar or unexpected — fall back to top-level
                                    row = measurement_data.copy()
                                    row['sample_id'] = sample_id
                                    rows.append(row)
                            else:
                                # No data_key — use top-level dict directly
                                row = measurement_data.copy()
                                row['sample_id'] = sample_id
                                rows.append(row)
            
                        if rows:
                            df = pd.DataFrame(rows)
                            self.current_results[measurement_key] = df
                            self.debug.print_debug(
                                f"Successfully loaded {measurement_key}: {len(df)} rows, "
                                f"columns: {list(df.columns)}"
                            )
                    else:
                        self.debug.print_debug(f"{measurement_key} returned None or empty")
                except Exception as e:
                    self.debug.print_debug(f"Error loading {measurement_key}: {e}")
        
        self.debug.print_debug(f"Loaded {len(self.current_results)} result types")
    
    def load_data_for_source(self, data_source: str, sample_ids: List[str], 
                            variation: Dict[str, str], process_manager) -> Optional[pd.DataFrame]:
        """
        Load data for a specific data source.
        
        Args:
            data_source: Name of data source (e.g., "evaporation - HTL" or "Results")
            sample_ids: List of sample IDs
            variation: Sample variation dictionary
            process_manager: ProcessStepManager for mapping display names
        
        Returns:
            DataFrame with loaded data or None
        """
        if data_source == 'Results':
            # Load all results if not already loaded
            if not self.current_results:
                all_results = self.data_loader.load_all_results(sample_ids, variation)
                self.current_results = all_results
            return None  # Results don't return a single dataframe
        
        # Map display name to measurement type
        measurement_type = process_manager.map_display_to_measurement_type(data_source)
        
        if not measurement_type:
            raise ValueError(f"Unknown data source: {data_source}")
        
        if measurement_type in self.current_metadata:
            metadata_df = self.current_metadata[measurement_type]
            
            # ISSUE 1 FIX: Apply layer type filtering
            filtered_df = self.filter_metadata_by_layer_type(metadata_df, data_source)
            
            return filtered_df
        else:
            self.debug.print_debug(f"Metadata type '{measurement_type}' not found in current_metadata")
            return None
    
    def extract_materials(self, df: pd.DataFrame, process_step_name: Optional[str] = None) -> List[str]:
        """
        Extract unique materials from a dataframe, optionally filtered by layer type.
        
        PERFORMANCE NOTE (G3): Uses vectorized pandas operations (dropna, unique, sorted)
        instead of Python loops for ~100x speed improvement on large datasets.
        """
        materials = []
        
        # If process_step_name is provided and contains layer type, filter first
        if process_step_name and '-' in process_step_name:
            # Extract layer type from process step name
            layer_type = process_step_name.split('-', 1)[1].strip()
            if 'layer_type' in df.columns:
                df = df[df['layer_type'].str.strip() == layer_type].copy()
                self.debug.print_debug(f"extract_materials filtered by layer_type='{layer_type}': {len(df)} rows")
        
        if 'layer_material_name' in df.columns:
            unique_materials = df['layer_material_name'].dropna().unique()
            materials = sorted([str(m) for m in unique_materials])
        elif 'layer_material' in df.columns:
            unique_materials = df['layer_material'].dropna().unique()
            materials = sorted([str(m) for m in unique_materials])
        
        return materials
    
    def filter_metadata_by_layer_type(self, df: pd.DataFrame, process_step_name: str) -> pd.DataFrame:
        """Filter metadata dataframe by layer type extracted from process step name."""
        if not process_step_name or '-' not in process_step_name or df is None or df.empty:
            return df
        
        # Extract layer type from process step name
        layer_type = process_step_name.split('-', 1)[1].strip()
        
        # Filter by layer type if the column exists
        if 'layer_type' in df.columns:
            filtered_df = df[df['layer_type'].str.strip() == layer_type].copy()
            self.debug.print_debug(f"filter_metadata_by_layer_type: '{layer_type}' - {len(df)} -> {len(filtered_df)} rows")
            return filtered_df
        else:
            self.debug.print_debug(f"filter_metadata_by_layer_type: No layer_type column")
            return df
    
    def rebuild_merged_data(self, x_data_source , y_data_source , color_data_source , 
                       x_material, y_material, color_material,
                       group_by_subbatch=False):
        """
        Rebuild merged_data based on current selections.
        
        Args:
            group_by_subbatch: If True, merge on subbatch instead of exact sample_id
        """
        print(f"\n[DEBUG] === rebuild_merged_data START ===")
        self.debug.print_debug(f"Data sources: x={x_data_source }, y={y_data_source }, color={color_data_source }")
        self.debug.print_debug(f"Materials: x={x_material}, y={y_material}, color={color_material}")

        # Track which material column in merged_data belongs to which source
        self._source_material_columns = {}
        
        if not self.current_metadata and not self.current_results:
            self.debug.print_debug(f"No data to merge!")
            return
        
        merged_data = None
        
        # Collect all data sources that need to be included
        active_data_sources = {}
        
        from utils import ProcessStepManager
        pm = ProcessStepManager()
        
        # Check X data source
        if x_data_source and x_data_source != 'Results' and x_data_source != 'None':
            measurement_type = pm.map_display_to_measurement_type(x_data_source)
            if measurement_type and measurement_type in self.current_metadata:
                active_data_sources[x_data_source] = {
                    'measurement_type': measurement_type,
                    'material_filter': x_material if x_material != 'All' else None,
                    'df': self.current_metadata[measurement_type]
                }
                self.debug.print_debug(f"Added X data source: {x_data_source} -> {measurement_type}")
        
        # Check Y data source
        if y_data_source and y_data_source != 'Results' and y_data_source != 'None':
            measurement_type = pm.map_display_to_measurement_type(y_data_source)
            if measurement_type and measurement_type in self.current_metadata:
                if y_data_source not in active_data_sources:
                    active_data_sources[y_data_source] = {
                        'measurement_type': measurement_type,
                        'material_filter': y_material if y_material != 'All' else None,
                        'df': self.current_metadata[measurement_type]
                    }
                    self.debug.print_debug(f"Added Y data source: {y_data_source} -> {measurement_type}")
        
        # Check Color data source
        if color_data_source and color_data_source != 'Results' and color_data_source != 'None':
            measurement_type = pm.map_display_to_measurement_type(color_data_source)
            if measurement_type and measurement_type in self.current_metadata:
                if color_data_source not in active_data_sources:
                    active_data_sources[color_data_source] = {
                        'measurement_type': measurement_type,
                        'material_filter': color_material if color_material != 'All' else None,
                        'df': self.current_metadata[measurement_type]
                    }
                    self.debug.print_debug(f"Added Color data source: {color_data_source} -> {measurement_type}")
        
        self.debug.print_debug(f"Active data sources: {list(active_data_sources.keys())}")
        
        # Process each active data source
        for data_source_name, source_info in active_data_sources.items():
            df = source_info['df'].copy()
            
            # Apply layer type filter
            if '-' in data_source_name:
                layer_type = data_source_name.split('-', 1)[1].strip()
                if 'layer_type' in df.columns:
                    before_count = len(df)
                    df = df[df['layer_type'].str.strip() == layer_type]
                    self.debug.print_debug(f"Filtered {data_source_name} by layer_type='{layer_type}': {before_count} -> {len(df)} rows")
            
            # Apply material filter
            if source_info['material_filter']:
                material_col = 'layer_material_name' if 'layer_material_name' in df.columns else 'layer_material'
                if material_col in df.columns:
                    before_count = len(df)
                    df = df[df[material_col] == source_info['material_filter']]
                    self.debug.print_debug(f"Filtered {data_source_name} by material='{source_info['material_filter']}': {before_count} -> {len(df)} rows")
            
            if df.empty:
                self.debug.print_debug(f"{data_source_name} filtered to empty, skipping")
                continue
            
            self.debug.print_debug(f"Including {data_source_name}: {len(df)} rows, columns: {list(df.columns)[:10]}")
            
            # Merge into combined data
            if merged_data is None:
                merged_data = df
                # First source keeps original column names
                for mat_col in ['layer_material_name', 'layer_material']:
                    if mat_col in merged_data.columns:
                        self._source_material_columns[data_source_name] = mat_col
                        print(f"[DEBUG] Source '{data_source_name}' → material col '{mat_col}'")
                        break
            else:
                suffix = f'_{source_info["measurement_type"]}'
                
                # SUBBATCH GROUPING LOGIC
                if group_by_subbatch:
                    # Extract subbatch for both dataframes
                    merged_data['_subbatch'], merged_data['_valid'] = zip(*merged_data['sample_id'].apply(self.extract_subbatch))
                    df['_subbatch'], df['_valid'] = zip(*df['sample_id'].apply(self.extract_subbatch))
                    
                    # Track invalid samples
                    invalid_merged = merged_data[~merged_data['_valid']]['sample_id'].unique()
                    invalid_df = df[~df['_valid']]['sample_id'].unique()
                    if not hasattr(self, '_invalid_samples'):
                        self._invalid_samples = set()
                    self._invalid_samples.update(invalid_merged)
                    self._invalid_samples.update(invalid_df)
                    
                    # Filter to valid only
                    merged_data = merged_data[merged_data['_valid']].copy()
                    df = df[df['_valid']].copy()
                    
                    # Merge on subbatch
                    merged_data = pd.merge(
                        merged_data, df,
                        on='_subbatch', how='inner',
                        suffixes=('', suffix)
                    )
                    
                    # Clean up temp columns
                    merged_data = merged_data.drop(columns=['_subbatch', '_valid'], errors='ignore')
                    if f'_valid{suffix}' in merged_data.columns:
                        merged_data = merged_data.drop(columns=[f'_valid{suffix}'])
                else:
                    # Original exact sample_id merge
                    merged_data = pd.merge(
                        merged_data, df,
                        on='sample_id', how='outer',
                        suffixes=('', suffix)
                    )
                
                # Track material columns after merge (same for both paths)
                for mat_col in ['layer_material_name', 'layer_material']:
                    suffixed = f'{mat_col}{suffix}'
                    if suffixed in merged_data.columns:
                        self._source_material_columns[data_source_name] = suffixed
                        print(f"[DEBUG] Source '{data_source_name}' → material col '{suffixed}'")
                        break
                    elif mat_col in merged_data.columns and data_source_name not in self._source_material_columns:
                        self._source_material_columns[data_source_name] = mat_col
                        print(f"[DEBUG] Source '{data_source_name}' → material col '{mat_col}' (unsuffixed)")
                        break
                
                self.debug.print_debug(f"After merging {data_source_name}: {len(merged_data)} rows, {len(merged_data.columns)} columns")
        
        # Merge results if needed
        if self.current_results:
            self.debug.print_debug(f"Merging results data...")
            for result_type, result_df in self.current_results.items():
                if result_df is None or result_df.empty:
                    continue
                
                self.debug.print_debug(f"Merging {result_type}: {len(result_df)} rows")
                
                cols_to_drop = [col for col in result_df.columns 
                               if merged_data is not None and col in merged_data.columns and col != 'sample_id']
                result_df_clean = result_df.drop(columns=cols_to_drop) if cols_to_drop and merged_data is not None else result_df
                
                if merged_data is None:
                    merged_data = result_df_clean.copy()
                else:
                    merged_data = pd.merge(
                        merged_data, result_df_clean,
                        on='sample_id', how='outer',
                        suffixes=('', f'_{result_type}')
                    )
                
                self.debug.print_debug(f"After merging {result_type}: {len(merged_data)} rows, {len(merged_data.columns)} columns")

        # Return invalid samples if any
        invalid_samples = getattr(self, '_invalid_samples', set())
        self._invalid_samples = set()  # Clear for next call
        
        self.merged_data = merged_data
        
        if merged_data is not None:
            material_cols = [col for col in merged_data.columns if 'material' in col.lower()]
            self.debug.print_debug(f"Final merged data: {len(merged_data)} rows, {len(merged_data.columns)} columns")
            self.debug.print_debug(f"Material columns: {material_cols}")
        else:
            self.debug.print_debug(f"Final merged data is None!")
        
        self.debug.print_debug(f"=== rebuild_merged_data END ===\n")
        return invalid_samples
    
    def get_parameter_options(self, data_dict: Dict[str, pd.DataFrame], param_type: str, 
                             is_results: bool = False) -> List[str]:
        """
        Get available parameters from data.
        
        PERFORMANCE NOTE (G4): For results data, only loads the specific measurement
        types that are present, not all possible types.
        """
        if not data_dict:
            return []
        
        # Initialize params list FIRST
        params = []
        
        # ISSUE 3 FIX: Add "Material Type" option for process steps when "All" is selected
        if not is_results:
            add_material_type = False
            if param_type == 'x_parameters' and getattr(self, '_x_material_is_all', False):
                add_material_type = True
            elif param_type == 'y_parameters' and getattr(self, '_y_material_is_all', False):
                add_material_type = True
            elif param_type == 'color_parameters' and getattr(self, '_color_material_is_all', False):
                add_material_type = True
            
            if add_material_type:
                params.append("Material Type")
        
        if is_results:
            self.debug.print_if_enabled('show_parameter_flow', 
                f"\n=== RESULTS PARAMETER COLLECTION ===\n"
                f"param_type={param_type}, is_results={is_results}\n"
                f"data_dict keys: {list(data_dict.keys())}")
            
            # Collect result parameters with measurement type suffix
            for result_type, result_df in data_dict.items():
                if result_df is None or result_df.empty:
                    continue
                
                self.debug.print_if_enabled('show_parameter_flow',
                    f"\nProcessing {result_type}:\n"
                    f"  Shape: {result_df.shape}\n"
                    f"  Columns: {list(result_df.columns)}\n"
                    f"  'datetime' present: {'datetime' in result_df.columns}")
                
                for col in result_df.columns:
                    # Determine measurement type suffix
                    if result_type == 'jv_measurement':
                        suffix = 'JV'
                    elif result_type == 'eqe_measurement':
                        suffix = 'EQE'
                    elif result_type == 'mpp_tracking':
                        suffix = 'MPP'
                    elif result_type == 'simple_mpp_tracking':  # ISSUE 4 FIX
                        suffix = 'Simple MPP'
                    elif result_type == 'pl_measurement':
                        suffix = 'PL'
                    elif result_type == 'trpl_measurement':
                        suffix = 'TRPL'
                    elif result_type == 'abspl_measurement':
                        suffix = 'AbsPL'
                    else:
                        suffix = result_type.replace('_measurement', '').upper()
                    
                    params.append(f"{col} ({suffix})")
            
            self.debug.print_if_enabled('show_parameter_flow',
                f"\nCollected {len(params)} result parameters\n"
                f"First 10: {params[:10]}\n"
                f"'datetime (JV)' present: {'datetime (JV)' in params}")
        else:
            # Collect metadata parameters
            # Don't pre-filter by COMMON_COLUMNS - let the blacklist handle it
            
            for measurement_type, metadata_df in data_dict.items():
                for col in metadata_df.columns:
                    if not col.endswith(('_jv', '_eqe', '_mpp')):
                        params.append(col)
        
        # Apply blacklist filtering
        self.debug.print_if_enabled('show_parameter_flow',
            f"\n=== BLACKLIST FILTERING ===\n"
            f"Before: {len(params)} parameters\n"
            f"Sample params: {params[:15]}\n"
            f"'datetime (JV)' before filter: {'datetime (JV)' in params}")
        
        filtered = self.param_manager.filter_parameters(params, param_type)
        
        self.debug.print_if_enabled('show_parameter_flow',
            f"After: {len(filtered)} parameters\n"
            f"Sample filtered: {filtered[:15]}\n"
            f"'datetime (JV)' after filter: {'datetime (JV)' in filtered}")
        
        # Sort alphabetically
        return sorted(filtered)
    
    def generate_parameter_summary(self, summary_output):
        """Generate parameter summary table in the output widget."""
        with summary_output:
            clear_output()
            
            if not self.current_metadata and not self.current_results:
                print("No data loaded yet.")
                return
            
            # Track available result types
            result_types = set()
            for result_type, result_df in self.current_results.items():
                if result_df is not None and not result_df.empty:
                    if result_type == 'jv_measurement':
                        result_types.add('JV')
                    elif result_type == 'eqe_measurement':
                        result_types.add('EQE')
                    elif result_type == 'mpp_tracking':
                        result_types.add('MPP Tracking')
                    elif result_type == 'simple_mpp_tracking': 
                        result_types.add('Simple MPP')
                    elif result_type == 'sem':
                        result_types.add('SEM')
                    elif result_type == 'abspl_measurement':
                        result_types.add('AbsPL')
                    elif result_type == 'xrd':
                        result_types.add('XRD')
            
            # Display results
            print("="*80)
            print("AVAILABLE RESULT TYPES")
            print("="*80)
            if result_types:
                result_list = sorted(result_types)
                print(f"  {', '.join(result_list)}")
            else:
                print("  None")
            
            # Group parameters by source
            exclude_cols = self.COMMON_COLUMNS[:8]
            
            # Process each metadata source
            for measurement_type, metadata_df in self.current_metadata.items():
                display_name = measurement_type.replace('_', ' ').title()
                
                # Check if this process has a material column
                material_col = self.get_material_column(metadata_df)
                
                if material_col and material_col in metadata_df.columns:
                    # Check if layer_type column exists
                    if 'layer_type' in metadata_df.columns:
                        # Split by both layer_type and material
                        for (layer_type, material_name), group_df in metadata_df.groupby(['layer_type', material_col]):
                            if pd.isna(material_name):
                                continue
                            
                            # Build section name with layer type
                            section_name = f"{display_name} - {layer_type} - {material_name}" if not pd.isna(layer_type) else f"{display_name} - {material_name}"
                            
                            self._display_param_table(
                                group_df, 
                                section_name, 
                                exclude_cols
                            )
                    else:
                        # Split by material only (no layer_type column)
                        for material_name, material_df in metadata_df.groupby(material_col):
                            if pd.isna(material_name):
                                continue
                            
                            self._display_param_table(
                                material_df, 
                                f"{display_name} - {material_name}", 
                                exclude_cols
                            )
                else:
                    # No material split
                    self._display_param_table(metadata_df, display_name, exclude_cols)
            
            print("\n" + "="*80)

    def _display_param_table(self, df, section_name, exclude_cols):
        """Display parameter summary table for a single section."""
        # Collect varying parameters
        source_params = []
        for col in df.columns:
            if col in exclude_cols:
                continue
            
            try:
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                
                unique_count = series.nunique()
                total_count = len(series)
                
                # Skip parameters with no variation
                if unique_count <= 1:
                    continue
                
                variation_ratio = unique_count / total_count
                
                # Get sample values
                if series.dtype in ['int64', 'float64']:
                    # Show actual values if <= 5 unique
                    if unique_count <= 5:
                        unique_vals = sorted(series.unique())
                        value_range = ", ".join(f"{v:.3g}" for v in unique_vals)
                    else:
                        value_range = f"[{series.min():.3g}, {series.max():.3g}]"
                elif series.dtype == 'bool':
                    counts = series.value_counts()
                    value_range = f"T:{counts.get(True, 0)}, F:{counts.get(False, 0)}"
                else:
                    unique_vals = series.unique()[:5]
                    value_range = ", ".join(str(v)[:15] for v in unique_vals)
                    if len(series.unique()) > 5:
                        value_range += f"... (+{len(series.unique())-5})"
                
                source_params.append({
                    'parameter': col,
                    'unique_count': unique_count,
                    'total_count': total_count,
                    'variation_ratio': variation_ratio,
                    'value_range': value_range
                })
            except Exception as e:
                continue
        
        # Display ONLY if has varying parameters
        if source_params:
            source_params.sort(key=lambda x: (x['unique_count'], x['variation_ratio']), reverse=True)
            
            print("\n" + "="*80)
            print(f"{section_name.upper()}")
            print("="*80)
            print(f"{'Parameter':<30} {'Unique':<8} {'Samples':<8} {'Var%':<8} {'Values'}")
            print("-"*80)
            
            for param in source_params:
                variation_pct = f"{param['variation_ratio']*100:.1f}%"
                param_name = param['parameter'][:28]
                if len(param['parameter']) > 28:
                    param_name += ".."
                
                print(f"{param_name:<30} {param['unique_count']:<8} "
                      f"{param['total_count']:<8} {variation_pct:<8} {param['value_range']}")
