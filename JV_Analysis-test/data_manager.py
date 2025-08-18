"""
Data Management Module
Handles all data loading, processing, filtering, and basic analysis operations.
"""

__author__ = "Edgar Nandayapa"
__institution__ = "Helmholtz-Zentrum Berlin"
__created__ = "August 2025"

import pandas as pd
import numpy as np
import os
import operator
import sys

# Add parent directory for shared modules
parent_dir = os.path.dirname(os.getcwd())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from api_calls import get_ids_in_batch, get_sample_description, get_all_JV
from error_handler import ErrorHandler


def extract_status_from_metadata(data, metadata):
    """
    Extract status from API metadata containing filename
    """
    import re
    
    # Look for filename in metadata
    filename_candidates = [
        metadata.get('mainfile', ''),
        metadata.get('upload_name', ''),
        metadata.get('entry_name', ''),
        metadata.get('filename', ''),
        # Also check in data if filename is stored there
        data.get('data_file', '') if isinstance(data, dict) else '',
    ]
    
    for candidate in filename_candidates:
        if candidate:
            # Extract status from filename like "HZB_JJ_1_B_C-8.JJ_1_B_8_L1_jv.jv.txt"
            status_match = re.search(r'_([LD]\d+)(?:_3min)?_', candidate)
            if not status_match:
                status_match = re.search(r'([LD]\d+)', candidate)
            
            if status_match:
                return status_match.group(1)
    
    return 'N/A'


class DataManager:
    """Main data management class for JV analysis application"""
    
    def __init__(self, auth_manager):
        self.auth_manager = auth_manager
        self.data = {}
        self.unique_vals = []
        self.filtered_data = None
        self.omitted_data = None
        self.filter_parameters = []
        # Store export data
        self.export_jvc_data = None
        self.export_curves_data = None
    
    def load_batch_data(self, batch_ids, output_widget=None):
        """Load data from selected batch IDs"""
        self.data = {}
        
        if not self.auth_manager.is_authenticated():
            ErrorHandler.log_error("Authentication required", output_widget=output_widget)
            return False
        
        if not batch_ids:
            ErrorHandler.log_error("Please select at least one batch to load", output_widget=output_widget)
            return False
        
        try:
            if output_widget:
                with output_widget:
                    print("Loading Data")
                    print(f"Loading data for batch IDs: {batch_ids}")
            
            url = self.auth_manager.url
            token = self.auth_manager.current_token
            
            # Get sample IDs and descriptions
            sample_ids = get_ids_in_batch(url, token, batch_ids)
            identifiers = get_sample_description(url, token, sample_ids)
            
            # Process JV data
            df_jvc, df_cur = self._process_jv_data_for_analysis(sample_ids, output_widget)
            
            # Store data
            self.data["jvc"] = pd.concat([self.data.get("jvc", pd.DataFrame()), df_jvc], ignore_index=True)
            self.data["curves"] = pd.concat([self.data.get("curves", pd.DataFrame()), df_cur], ignore_index=True)
            
            # Process sample information
            self._process_sample_info(identifiers)
            
            # Export data
            self._export_data(df_jvc, df_cur)
            
            # Find unique values
            self.unique_vals = self._find_unique_values()
            
            if output_widget:
                with output_widget:
                    print("Data Loaded Successfully!")
            
            return True
            
        except Exception as e:
            ErrorHandler.handle_data_loading_error(e, output_widget)
            return False
    
    def _process_jv_data_for_analysis(self, sample_ids, output_widget=None):
        """Process JV data for analysis from sample IDs"""
        columns_jvc = ['Voc(V)', 'Jsc(mA/cm2)', 'FF(%)', 'PCE(%)', 'V_mpp(V)', 'J_mpp(mA/cm2)',
                      'P_mpp(mW/cm2)', 'R_series(Ohmcm2)', 'R_shunt(Ohmcm2)', 'sample', 'batch',
                      'condition', 'cell', 'direction', 'ilum', 'status']
        columns_cur = ['index', 'sample', 'batch', 'condition', 'variable', 'cell', 'direction', 'ilum']
        rows_jvc = []
        rows_cur = []
        
        try:
            url = self.auth_manager.url
            token = self.auth_manager.current_token
            
            if output_widget:
                with output_widget:
                    print("Fetching JV data...")
            
            all_jvs = get_all_JV(url, token, sample_ids)
            
            for sid in sample_ids:
                jv_res = all_jvs.get(sid, [])
                if output_widget:
                    with output_widget:
                        print(f"Processing: {sid}")
                
                for jv_data, jv_md in jv_res:
                    status = extract_status_from_metadata(jv_data, jv_md)
                    for c in jv_data["jv_curve"]:
                        file_name = os.path.join("../", jv_md["upload_id"], jv_data.get("data_file"))
                        illum = "Dark" if "dark" in c["cell_name"].lower() else "Light"
                        cell = c["cell_name"][0]
                        direction = "Forward" if "for" in c["cell_name"].lower() else "Reverse"
                        
                        row = [
                            c["open_circuit_voltage"],
                            -c["short_circuit_current_density"],
                            100 * c["fill_factor"],
                            c["efficiency"],
                            c["potential_at_maximum_power_point"],
                            -c["current_density_at_maximun_power_point"],
                            -c["potential_at_maximum_power_point"] * c["current_density_at_maximun_power_point"],
                            c["series_resistance"],
                            c["shunt_resistance"],
                            file_name,
                            file_name.split("/")[1],
                            "w",
                            cell,
                            direction,
                            illum,
                            status  # ADD this line
                        ]
                        rows_jvc.append(row)
                        
                        # Process voltage data
                        row_v = [
                            "_".join(["Voltage (V)", cell, direction, illum]),
                            file_name,
                            file_name.split("/")[1],
                            "w",
                            "Voltage (V)",
                            cell,
                            direction,
                            illum
                        ]
                        row_v.extend(c["voltage"])
                        
                        # Process current density data
                        row_j = [
                            "_".join(["Current Density(mA/cm2)", cell, direction, illum]),
                            file_name,
                            file_name.split("/")[1],
                            "w",
                            "Current Density(mA/cm2)",
                            cell,
                            direction,
                            illum
                        ]
                        row_j.extend(c["current_density"])
                        
                        # Update columns_cur with new indices for both voltage and current
                        max_data_points = max(len(c["voltage"]), len(c["current_density"]))
                        for i in range(max_data_points):
                            if i not in columns_cur:
                                columns_cur.append(i)
                        
                        rows_cur.append(row_v)
                        rows_cur.append(row_j)
            
            df_jvc = pd.DataFrame(rows_jvc, columns=columns_jvc)
            df_cur = pd.DataFrame(rows_cur, columns=columns_cur)
            
            return df_jvc, df_cur
            
        except Exception as e:
            ErrorHandler.handle_data_loading_error(e, output_widget)
            return pd.DataFrame(), pd.DataFrame()
    
    def _process_sample_info(self, identifiers):
        """Process sample information and create identifiers with enhanced deduplication"""
        # Store original sample paths before cleaning
        self.data["jvc"]["original_sample"] = self.data["jvc"]["sample"].copy()
        
        self.data["jvc"]["subbatch"] = self.data["jvc"]["sample"].apply(
            lambda x: x.split('/')[-1].split('.')[0].split('_')[-2]
        )
        
        # Extract human-readable batch name for display using original paths
        def extract_display_batch(sample_path):
            filename = sample_path.split('/')[-1].split('.')[0]
            parts = filename.split('_')
            
            if len(parts) >= 2:
                batch_parts = parts[:-1]
                result = '_'.join(batch_parts)
            else:
                result = parts[0] if parts else "unknown"
            
            return result
        
        # Keep the original batch ID from the file path for actual grouping
        self.data["jvc"]["batch"] = self.data["jvc"]["sample"].apply(
            lambda x: x.split("/")[1] if "/" in x else "unknown"
        )
        
        # Add display batch name for UI purposes using original paths
        self.data["jvc"]["display_batch"] = self.data["jvc"]["original_sample"].apply(extract_display_batch)
        
        self.data["jvc"]["identifier"] = self.data["jvc"]["sample"].apply(
            lambda x: x.split('/')[-1].split(".")[0]
        )
        
        if identifiers:
            self.data["jvc"]["identifier"] = self.data["jvc"]["identifier"].apply(
                lambda x: f'{"_".join(x.split("_")[:-1])}&{identifiers.get(x, "No variation specified")}'
            )
        else:
            self.data["jvc"]["identifier"] = self.data["jvc"]["sample"].apply(
                lambda x: "_".join(x.split('/')[-1].split(".")[0].split("_")[:-1])
            )
        
        # Clean sample names AFTER extracting display batch
        self.data["jvc"]["sample"] = self.data["jvc"]["sample"].apply(
            lambda x: x.split('/')[-1].split('.')[0].split('_', 4)[-1]
        )
        self.data["curves"]["sample"] = self.data["curves"]["sample"].apply(
            lambda x: x.split('/')[-1].split('.')[0].split('_', 4)[-1]
        )
        
        # Enhanced deduplication strategy
        grouped = self.data['jvc'].groupby(['sample', 'cell', 'direction', 'ilum'])
        
        rows_to_keep = []
        duplicate_issue_count = 0
        
        for name, group in grouped:
            unique_identifiers = group['identifier'].unique()
            
            if len(unique_identifiers) == 1:
                # Simple case: all records have same identifier, keep first
                rows_to_keep.append(group.iloc[0].name)
            else:
                # Complex case: multiple identifiers for same physical measurement
                duplicate_issue_count += 1
                
                # Strategy: prefer non-empty/meaningful identifiers
                best_idx = None
                best_score = -1
                
                for idx, row in group.iterrows():
                    identifier = str(row['identifier'])
                    score = 0
                    
                    # Prefer identifiers with meaningful content after &
                    if '&' in identifier:
                        variation = identifier.split('&', 1)[1]
                        if variation and variation != 'No variation specified' and variation != 'w':
                            score += 10
                            # Prefer certain keywords
                            if any(keyword in variation for keyword in ['BL Printing', 'Slot_SAM', 'Spin_SAM']):
                                score += 5
                    
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                
                # If no clear winner, just take the first
                if best_idx is None:
                    best_idx = group.iloc[0].name
                
                rows_to_keep.append(best_idx)
        
        # Keep only the selected rows
        self.data['jvc'] = self.data['jvc'].loc[rows_to_keep].reset_index(drop=True)
        
        # Also deduplicate curves data to match
        self.data['curves'] = self.data['curves'].drop_duplicates(
            subset=['sample', 'cell', 'direction', 'ilum', 'variable'],
            keep='first'
        ).reset_index(drop=True)
    
    def _export_data(self, df_jvc, df_cur):
        """Store data for potential export"""
        self.export_jvc_data = df_jvc
        self.export_curves_data = df_cur
    
    def _find_unique_values(self):
        """Find unique values in the dataset"""
        try:
            unique_values = self.data["jvc"]["identifier"].unique()
        except:
            unique_values = self.data["jvc"]["sample"].unique()
        
        return unique_values
    
    def apply_conditions(self, conditions_dict):
        """Apply conditions mapping to the data"""
        if "jvc" in self.data:
            # Apply the mapping
            self.data['jvc']['condition'] = self.data['jvc']['identifier'].map(conditions_dict)
            
            # Fill any NaN values with a default
            nan_conditions = self.data['jvc']['condition'].isna().sum()
            if nan_conditions > 0:
                self.data['jvc']['condition'] = self.data['jvc']['condition'].fillna('Unknown')
            
            # Verify that each sample_cell has only one condition
            condition_check = self.data['jvc'].groupby(['sample', 'cell'])['condition'].nunique()
            multiple_conditions = condition_check[condition_check > 1]
            
            if len(multiple_conditions) > 0:
                return False
            
            return True
        return False
    
    def apply_filters(self, filter_list, direction_filter='Both', selected_items=None, verbose=True):
        """Apply filters to the dataframe with improved two-step process"""
        if not self.data or "jvc" not in self.data:
            return None, None, []
        
        # Default filters if none provided
        if not filter_list:
            filter_list = [("PCE(%)", "<", "40"), ("FF(%)", "<", "89"), ("FF(%)", ">", "24"), 
                          ("Voc(V)", "<", "2"), ("Jsc(mA/cm2)", ">", "-30")]
        
        # Operator mapping
        operat = {"<": operator.lt, ">": operator.gt, "==": operator.eq,
                  "<=": operator.le, ">=": operator.ge, "!=": operator.ne}
        
        data = self.data["jvc"].copy()
        
        # Initialize filter reason column
        data['filter_reason'] = ''
        filtering_options = []
        
        # Apply sample/cell selection filter if provided
        sample_selection_filtered_count = 0
        if selected_items:
            original_count = len(data)
            
            def is_selected(row):
                cell_key = f"{row['sample']}_{row['cell']}"
                return cell_key in selected_items
            
            selection_mask = data.apply(is_selected, axis=1)
            data.loc[~selection_mask, 'filter_reason'] += 'sample/cell not selected, '
            
            sample_selection_filtered_count = len(data[~selection_mask])
            filtering_options.append(f'sample/cell selection ({sample_selection_filtered_count} filtered)')
        
        # Apply numeric filters
        for col, op, val in filter_list:
            try:
                mask = operat[op](data[col], float(val))
                before_count = len(data[data['filter_reason'] == ''])
                data.loc[~mask, 'filter_reason'] += f'{col} {op} {val}, '
                after_count = len(data[data['filter_reason'] == ''])
                filtered_by_this_condition = before_count - after_count
                
                if filtered_by_this_condition > 0:
                    filtering_options.append(f'{col} {op} {val} ({filtered_by_this_condition} filtered)')
                
            except (ValueError, KeyError) as e:
                if verbose:
                    print(f"Warning: Could not apply filter {col} {op} {val}: {e}")
        
        # Apply direction filter
        if direction_filter != 'Both' and 'direction' in data.columns:
            before_count = len(data[data['filter_reason'] == ''])
            direction_mask = data['direction'] != direction_filter
            data.loc[direction_mask, 'filter_reason'] += f'direction != {direction_filter}, '
            after_count = len(data[data['filter_reason'] == ''])
            direction_filtered_count = before_count - after_count
            
            if direction_filtered_count > 0:
                filtering_options.append(f'direction == {direction_filter} ({direction_filtered_count} filtered)')
        
        # Separate filtered and omitted data
        omitted = data[data['filter_reason'] != ''].copy()
        filtered = data[data['filter_reason'] == ''].copy()
        
        # Clean up filter reason string
        omitted['filter_reason'] = omitted['filter_reason'].str.rstrip(', ')
        
        # Store results
        self.filtered_data = filtered
        self.omitted_data = omitted
        self.filter_parameters = filtering_options
        
        # Update main data dict
        self.data['filtered'] = filtered
        self.data['junk'] = omitted
        
        return filtered, omitted, filtering_options
    
    def generate_summary_statistics(self, df=None):
        """Generate comprehensive summary statistics"""
        if df is None:
            df = self.data.get("jvc", pd.DataFrame())
        
        if df.empty:
            return "No data available for summary."
        
        try:
            # Basic statistics
            global_mean_PCE = df['PCE(%)'].mean()
            global_std_PCE = df['PCE(%)'].std()
            max_PCE_row = df.loc[df['PCE(%)'].idxmax()]
            
            # Group statistics by sample and batch
            mean_std_PCE_per_sample = df.groupby(['batch', 'sample'])['PCE(%)'].agg(['mean', 'std'])
            highest_mean_PCE_sample = mean_std_PCE_per_sample.idxmax()['mean']
            lowest_mean_PCE_sample = mean_std_PCE_per_sample.idxmin()['mean']
            
            # Highest PCE per sample (including batch info)
            if 'display_batch' in df.columns:
                highest_PCE_per_sample = df.loc[df.groupby(['sample'])['PCE(%)'].idxmax(), ['batch', 'sample', 'cell', 'PCE(%)', 'display_batch']]
                highest_PCE_per_sample = highest_PCE_per_sample.copy()
                highest_PCE_per_sample['display_name'] = highest_PCE_per_sample['display_batch'] + '_' + highest_PCE_per_sample['sample']
                max_PCE_display_name = max_PCE_row.get('display_batch', max_PCE_row.get('batch', '')) + '_' + max_PCE_row['sample']
            else:
                highest_PCE_per_sample = df.loc[df.groupby(['sample'])['PCE(%)'].idxmax(), ['batch', 'sample', 'cell', 'PCE(%)']]
                highest_PCE_per_sample = highest_PCE_per_sample.copy()
                highest_PCE_per_sample['display_name'] = highest_PCE_per_sample['batch'] + '_' + highest_PCE_per_sample['sample']
                max_PCE_display_name = max_PCE_row.get('batch', '') + '_' + max_PCE_row['sample']
            
            # Create detailed markdown table
            markdown_output = f"""
### Summary Statistics

**Global mean PCE(%)**: {global_mean_PCE:.2f} ± {global_std_PCE:.2f}%
**Total measurements**: {len(df)}

#### Best and Worst Samples by Average PCE

| | Sample | Mean PCE(%) | Std PCE(%) |
|---|--------|-------------|------------|
| Best sample | {highest_mean_PCE_sample[1]} | {mean_std_PCE_per_sample.loc[highest_mean_PCE_sample, 'mean']:.2f}% | {mean_std_PCE_per_sample.loc[highest_mean_PCE_sample, 'std']:.2f}% |
| Worst sample | {lowest_mean_PCE_sample[1]} | {mean_std_PCE_per_sample.loc[lowest_mean_PCE_sample, 'mean']:.2f}% | {mean_std_PCE_per_sample.loc[lowest_mean_PCE_sample, 'std']:.2f}% |

#### Top Performing Samples

| Sample | Cell | PCE(%) |
|--------|------|--------|
| **{max_PCE_display_name}** | **{max_PCE_row['cell']}** | **{max_PCE_row['PCE(%)']:.2f}%** |
"""
            
            # Add all samples (sorted by PCE descending)
            all_top_samples = highest_PCE_per_sample.sort_values('PCE(%)', ascending=False)
            for _, row in all_top_samples.iterrows():
                if row['sample'] != max_PCE_row['sample'] or row['batch'] != max_PCE_row['batch']:
                    display_name = row.get('display_name', f"{row.get('batch', '')}_{row['sample']}")
                    markdown_output += f"| {display_name} | {row['cell']} | {row['PCE(%)']:.2f}% |\n"
            
            # Add scan direction comparison if available
            if 'direction' in df.columns:
                forward_pce = df[df['direction'] == 'Forward']['PCE(%)'].mean()
                reverse_pce = df[df['direction'] == 'Reverse']['PCE(%)'].mean()
                
                markdown_output += f"""

#### Performance by Scan Direction

| Direction | Average PCE(%) | Count |
|-----------|----------------|-------|
| Forward | {forward_pce:.2f}% | {len(df[df['direction'] == 'Forward'])} |
| Reverse | {reverse_pce:.2f}% | {len(df[df['direction'] == 'Reverse'])} |
"""
            
            # Add distribution statistics
            markdown_output += f"""

#### Distribution Statistics

| Metric | Value |
|--------|-------|
| Median PCE | {df['PCE(%)'].median():.2f}% |
| Min PCE | {df['PCE(%)'].min():.2f}% |
| Max PCE | {df['PCE(%)'].max():.2f}% |
| 25th Percentile | {df['PCE(%)'].quantile(0.25):.2f}% |
| 75th Percentile | {df['PCE(%)'].quantile(0.75):.2f}% |
"""
            
            return markdown_output
            
        except Exception as e:
            return f"""
### Summary Statistics

**Error generating detailed statistics**: {str(e)}

**Basic Info**:
- Total measurements: {len(df)}
- Best Overall PCE: {df['PCE(%)'].max():.2f}% (Sample: {df.loc[df['PCE(%)'].idxmax(), 'sample']})
- Average PCE: {df['PCE(%)'].mean():.2f}%
"""
    
    # Getter methods
    def get_data(self):
        """Get the loaded data"""
        return self.data
    
    def get_unique_values(self):
        """Get unique values"""
        return self.unique_vals
    
    def get_filtered_data(self):
        """Get filtered data"""
        return self.filtered_data
    
    def get_omitted_data(self):
        """Get omitted data"""
        return self.omitted_data
    
    def get_filter_parameters(self):
        """Get filter parameters"""
        return self.filter_parameters
    
    def has_data(self):
        """Check if data is loaded"""
        return bool(self.data and "jvc" in self.data and not self.data["jvc"].empty)
    
    def get_export_data(self):
        """Get the export data for CSV download"""
        return self.export_jvc_data, self.export_curves_data
    
    def has_export_data(self):
        """Check if export data is available"""
        return self.export_jvc_data is not None and self.export_curves_data is not None