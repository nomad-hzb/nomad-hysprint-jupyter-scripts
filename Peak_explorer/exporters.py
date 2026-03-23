# exporters.py
"""Export utilities for analysis results"""
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
from utils import debug_print
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class ResultExporter:
    """Class to handle export of fitting results and analysis"""
    
    def __init__(self):
        debug_print("ResultExporter initialized", "Exporter")
        # self.export_formats = ['xlsx', 'csv', 'json', 'hdf5']
        
    def export_to_excel(self, fitting_results, timestamps, filename=None):
        """
        Export fitting results to Excel file with multiple sheets
        
        Parameters:
        -----------
        fitting_results : dict
            Dictionary of fitting results from batch fitting
        timestamps : array
            Time values
        filename : str, optional
            Output filename
            
        Returns:
        --------
        str: Path to exported file
        """
        if filename is None:
            filename = f"peak_fitting_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
        # Create Excel writer
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: Peak parameters (main data)
            peak_params_df, stderr_df = self._create_peak_parameters_dataframe(fitting_results)
            peak_params_df.to_excel(writer, sheet_name='Peak_Parameters', index=False)
            
            # Sheet 2: Standard errors
            if not stderr_df.empty:
                stderr_df.to_excel(writer, sheet_name='Standard_Errors', index=False)
            
            # Sheet 3: Fitting quality metrics
            quality_df = self._create_quality_metrics_dataframe(fitting_results)
            quality_df.to_excel(writer, sheet_name='Fitting_Quality', index=False)
            
        return filename

    def export_to_isa_h5(self, fitting_results, h5_path, time_unit='s', h5_mode=None, wavelength_unit='nm'):
        """
        Export fitting results into the existing HDF5 file.

        A timestamped subgroup is created under /fitting_results/ with the name
        fit_YYYYMMDD_HH_MM_SS. The group carries attributes for the time and
        index range covered as well as the creation timestamp. All columns from
        the merged dataframe (peak_params, stderr, quality) are written as
        individual datasets under that group. Each dataset has peak_name and
        model_type attributes.

        Parameters:
        -----------
        fitting_results : dict
            Dictionary of fitting results from batch fitting
        h5_path : str
        """
        import h5py

        debug_print("Exporting fitting results to HDF5: " + h5_path, "Exporter")

        peak_params_df, stderr_df = self._create_peak_parameters_dataframe(fitting_results)
        quality_df = self._create_quality_metrics_dataframe(fitting_results)

        # Merge the three dataframes into one; columns shared between them are kept once
        combined_df = peak_params_df.copy()
        for df in (stderr_df, quality_df):
            new_cols = [c for c in df.columns if c not in combined_df.columns]
            combined_df = combined_df.join(df[new_cols], how='outer') if new_cols else combined_df

        # Build peak_id -> {name, model_type} mapping from the first successful result
        peak_info_map = {}
        for result in fitting_results.values():
            if result and result.get('success'):
                for i, pm in enumerate(result.get('peak_models', [])):
                    pid = f'p{i}'
                    peak_info_map[pid] = {
                        'name': pm.get('name', pid),
                        'model_type': pm.get('type', 'Unknown')
                    }
                break

        debug_print("Peak info map: " + str(peak_info_map), "Exporter")

        # Time and index range from the merged df
        min_time  = float(combined_df['Time'].min())       if 'Time'       in combined_df.columns else float('nan')
        max_time  = float(combined_df['Time'].max())       if 'Time'       in combined_df.columns else float('nan')
        min_index = int(combined_df['Time_Index'].min())   if 'Time_Index' in combined_df.columns else -1
        max_index = int(combined_df['Time_Index'].max())   if 'Time_Index' in combined_df.columns else -1

        now = datetime.now()
        group_name = now.strftime('fit_%Y%m%d_%H_%M_%S')

        with h5py.File(h5_path, 'a') as h5f:
            grp = h5f.require_group(f'fitting_results/{group_name}')

            # Group-level attributes
            grp.attrs['min_time']    = min_time
            grp.attrs['max_time']    = max_time
            grp.attrs['min_index']   = min_index
            grp.attrs['max_index']   = max_index
            grp.attrs['h5_mode']     = h5_mode if h5_mode is not None else 'N/A'
            grp.attrs['created_at']  = now.isoformat()

            # Parameters whose unit equals the x-axis (wavelength / q) unit
            position_params = {'center', 'sigma', 'gamma', 'fwhm',
                               'width', 'width_10pct', 'width_50pct'}
            # Parameters that carry intensity (arbitrary units → '-')
            intensity_params = {'amplitude', 'height', 'area'}

            # One dataset per column
            for col in combined_df.columns:
                data = combined_df[col].values
                if col in grp:
                    del grp[col]
                ds = grp.create_dataset(col, data=data)

                # Store time_unit as attribute on the Time dataset
                if col == 'Time':
                    ds.attrs['time_unit'] = time_unit

                # Resolve peak_id from column name (e.g. "p0_center" -> "p0")
                parts = col.split('_')
                peak_id = parts[0] if len(parts) > 1 and parts[0].startswith('p') and parts[0][1:].isdigit() else None

                # Only store peak_name and model_type if dataset starts with 'p' and contains no strings
                if peak_id is not None and not any(isinstance(v, str) for v in data):
                    info = peak_info_map.get(peak_id, {})
                    ds.attrs['peak_name']   = info.get('name', 'N/A')
                    ds.attrs['model_type']  = info.get('model_type', 'N/A')

                    # Determine unit from base parameter name (strip _stderr suffix)
                    base_param = col[len(peak_id) + 1:]          # e.g. "center" or "center_stderr"
                    base_param = base_param.removesuffix('_stderr')
                    if base_param in position_params:
                        ds.attrs['unit'] = wavelength_unit
                    elif base_param in intensity_params:
                        ds.attrs['unit'] = '-'

        debug_print(f"Saved fitting results to {h5_path} under /fitting_results/{group_name}", "Exporter")


        
    def _create_peak_parameters_dataframe(self, fitting_results):
        """Create comprehensive peak parameters dataframe and separate stderr dataframe"""
        param_data = []
        stderr_data = []

        # debug_print("Fitting_results: " + str(fitting_results), "Exporter")
        
        for idx, result in fitting_results.items():
            if result is None or not result.get('success', False):
                continue
                
            base_row = {
                'Time_Index': result['index'],
                'Time': result['time']
            }
            
            # Extract peak parameters
            parameters = result.get('parameters', {})

            # debug_print("Raw Parameters: " + str(parameters), "Exporter")
            
            # Group parameters by peak
            peak_params = {}
            for param_name, param_data_item in parameters.items():
                if param_name.startswith('p') and '_' in param_name:
                    peak_id = param_name.split('_')[0]
                    param_type = '_'.join(param_name.split('_')[1:])
                    
                    if peak_id not in peak_params:
                        peak_params[peak_id] = {}
                    peak_params[peak_id][param_type] = param_data_item['value']
                    peak_params[peak_id][f'{param_type}_stderr'] = param_data_item.get('stderr', np.nan)

            # debug_print("Peak Parameters" + str(peak_params), "Exporter")

            # Create single row with all peaks' parameters
            if peak_params:
                row = base_row.copy()
                stderr_row = base_row.copy()
                
                # Sort peaks by ID for consistent ordering
                sorted_peaks = sorted(peak_params.keys(), key=lambda x: int(x[1:]))
                
                peak_models = result.get('peak_models', [])
                for peak_id in sorted_peaks:
                    params = peak_params[peak_id]
                    peak_index = int(peak_id[1:])
                    peak_type = peak_models[peak_index]['type'] if peak_index < len(peak_models) else 'Unknown'
                    row[f'{peak_id}_model_type'] = peak_type

                    # debug_print(f"Processing {peak_id} with model {peak_type}", "Exporter")
                    # debug_print("params: " + str(params), "Exporter")
                    
                    # Add main parameters with peak_id prefix
                    for param_type, value in params.items():
                        if not param_type.endswith('_stderr'):
                            # Handle special naming cases
                            if param_type == 'amplitude':
                                col_name = f'{peak_id}_amplitude'
                            elif param_type == 'center':
                                col_name = f'{peak_id}_center'
                            elif param_type == 'sigma':
                                col_name = f'{peak_id}_sigma'
                            elif param_type == 'gamma':  # For Voigt profiles
                                col_name = f'{peak_id}_gamma'
                            else:
                                col_name = f'{peak_id}_{param_type}'
                            
                            row[col_name] = value
                            
                            # Add stderr to separate dataframe
                            stderr_key = f'{param_type}_stderr'
                            if stderr_key in params:
                                stderr_row[f'{col_name}_stderr'] = params[stderr_key]

                    # Calculate derived parameters
                    # todo add other peak_type; refine if condition; some calculations are valid for all types
                    if peak_type == "Gaussian":
                    # if 'amplitude' in params and 'sigma' in params:
                        amplitude = params['amplitude']
                        sigma = params['sigma']
                        amplitude_stderr = params.get('amplitude_stderr', np.nan)
                        sigma_stderr = params.get('sigma_stderr', np.nan)

                        # Calculate height from amplitude (for Gaussian)
                        height = amplitude / (sigma * np.sqrt(2 * np.pi))
                        row[f'{peak_id}_height'] = height
                        # Gaußsche Fehlerfortpflanzungsgesetz.
                        # todo check with Edgar if this is correct
                        stderr_row[f'{peak_id}_height_stderr'] = height * np.sqrt(
                            (amplitude_stderr / amplitude) ** 2 + (sigma_stderr / sigma) ** 2
                        ) if amplitude and sigma else np.nan

                        # Calculate area under curve (integral of Gaussian)
                        row[f'{peak_id}_area'] = amplitude
                        stderr_row[f'{peak_id}_area_stderr'] = amplitude_stderr

                        # Calculate FWHM (Full Width at Half Maximum)
                        # fwhm = 2.355 * sigma
                        # fwhm_stderr = 2.355 * sigma_stderr
                        # row[f'{peak_id}_FWHM'] = fwhm
                        # stderr_row[f'{peak_id}_FWHM_stderr'] = fwhm_stderr

                        # Calculate peak width at different heights
                        row[f'{peak_id}_width_10pct'] = 4.29 * sigma
                        stderr_row[f'{peak_id}_width_10pct_stderr'] = 4.29 * sigma_stderr
                        # row[f'{peak_id}_width_50pct'] = 2.355 * sigma
                        # stderr_row[f'{peak_id}_width_50pct_stderr'] = 2.355 * sigma_stderr
                
                # Add R_squared as the last column
                row['R_Squared'] = result.get('r_squared', np.nan)
                stderr_row['R_Squared'] = np.nan  # R_squared doesn't have stderr
                
                param_data.append(row)
                stderr_data.append(stderr_row)
                
        main_df = pd.DataFrame(param_data).sort_values('Time_Index').reset_index(drop=True)
        stderr_df = pd.DataFrame(stderr_data).sort_values('Time_Index').reset_index(drop=True)
        main_df_amplitude_cols = [col for col in main_df.columns if '_amplitude' in col]
        main_df = main_df.drop(columns=main_df_amplitude_cols)
        stderr_df_amplitude_cols = [col for col in stderr_df.columns if '_amplitude_stderr' in col]
        stderr_df = stderr_df.drop(columns=stderr_df_amplitude_cols, errors='ignore')
        
        return main_df, stderr_df
        
    def _create_quality_metrics_dataframe(self, fitting_results):
        """Create fitting quality metrics dataframe"""
        quality_data = []
        
        for idx, result in fitting_results.items():
            if result is None:
                continue
                
            row = {
                'Time_Index': result['index'],
                'Time': result['time'],
                'Success': result.get('success', False),
                'R_Squared': result.get('r_squared', np.nan),
                'Chi_Squared': result.get('chi_squared', np.nan),
                'Reduced_Chi_Squared': result.get('reduced_chi_squared', np.nan),
                'AIC': result.get('aic', np.nan),
                'BIC': result.get('bic', np.nan)
            }
            
            # Calculate additional metrics if residuals are available
            if result.get('residuals') is not None:
                residuals = result['residuals']
                row['RMSE'] = np.sqrt(np.mean(residuals**2))
                row['MAE'] = np.mean(np.abs(residuals))
                row['Max_Residual'] = np.max(np.abs(residuals))
                
            quality_data.append(row)
            
        return pd.DataFrame(quality_data)
        
    def export_to_csv(self, fitting_results, output_dir=None):
        """Export fitting results to CSV files"""
        if output_dir is None:
            output_dir = f"peak_analysis_csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Export different data types to separate CSV files
        peak_params_df, _ = self._create_peak_parameters_dataframe(fitting_results)
        peak_params_df.to_csv(os.path.join(output_dir, 'peak_parameters.csv'), index=False)
        
        quality_df = self._create_quality_metrics_dataframe(fitting_results)
        quality_df.to_csv(os.path.join(output_dir, 'fitting_quality.csv'), index=False)
        
        return output_dir
        
    def export_to_json(self, fitting_results, filename=None):
        """Export fitting results to JSON format"""
        if filename is None:
            filename = f"peak_fitting_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        
        for idx, result in fitting_results.items():
            if result is None:
                continue
                
            json_result = {
                'index': result['index'],
                'time': result['time'],
                'success': result.get('success', False),
                'r_squared': result.get('r_squared'),
                'chi_squared': result.get('chi_squared'),
                'aic': result.get('aic'),
                'bic': result.get('bic'),
                'parameters': result.get('parameters', {})
            }
            
            # Convert fitted curve and residuals to lists
            if result.get('fitted_curve') is not None:
                json_result['fitted_curve'] = result['fitted_curve'].tolist()
            if result.get('residuals') is not None:
                json_result['residuals'] = result['residuals'].tolist()
                
            json_data[str(idx)] = json_result
            
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
            
        return filename

    # def _create_centers_plot(self, df, output_dir):
    #     """Create peak centers vs time plot"""
    #     fig = go.Figure()
    #
    #     for peak_id in df['Peak_ID'].unique():
    #         peak_data = df[df['Peak_ID'] == peak_id]
    #         fig.add_trace(go.Scatter(
    #             x=peak_data['Time'],
    #             y=peak_data['Center_nm'],
    #             mode='markers+lines',
    #             name=f'{peak_id} Center',
    #             line=dict(width=2),
    #             marker=dict(size=6)
    #         ))
    #
    #     fig.update_layout(
    #         title="Peak Centers vs Time",
    #         xaxis_title="Time (s)",
    #         yaxis_title="Peak Center (nm)",
    #         height=500,
    #         template='plotly_white'
    #     )
    #
    #     fig.write_html(os.path.join(output_dir, 'peak_centers_vs_time.html'))
    #
    # def _create_heights_plot(self, df, output_dir):
    #     """Create peak heights vs time plot"""
    #     fig = go.Figure()
    #
    #     for peak_id in df['Peak_ID'].unique():
    #         peak_data = df[df['Peak_ID'] == peak_id]
    #         if 'Height' in peak_data.columns:
    #             fig.add_trace(go.Scatter(
    #                 x=peak_data['Time'],
    #                 y=peak_data['Height'],
    #                 mode='markers+lines',
    #                 name=f'{peak_id} Height',
    #                 line=dict(width=2),
    #                 marker=dict(size=6)
    #             ))
    #
    #     fig.update_layout(
    #         title="Peak Heights vs Time",
    #         xaxis_title="Time (s)",
    #         yaxis_title="Peak Height",
    #         height=500,
    #         template='plotly_white'
    #     )
    #
    #     fig.write_html(os.path.join(output_dir, 'peak_heights_vs_time.html'))
    #
    # def _create_areas_plot(self, df, output_dir):
    #     """Create peak areas vs time plot"""
    #     fig = go.Figure()
    #
    #     for peak_id in df['Peak_ID'].unique():
    #         peak_data = df[df['Peak_ID'] == peak_id]
    #         if 'Area_Under_Curve' in peak_data.columns:
    #             fig.add_trace(go.Scatter(
    #                 x=peak_data['Time'],
    #                 y=peak_data['Area_Under_Curve'],
    #                 mode='markers+lines',
    #                 name=f'{peak_id} Area',
    #                 line=dict(width=2),
    #                 marker=dict(size=6)
    #             ))
    #
    #     fig.update_layout(
    #         title="Peak Areas vs Time",
    #         xaxis_title="Time (s)",
    #         yaxis_title="Area Under Curve",
    #         height=500,
    #         template='plotly_white'
    #     )
    #
    #     fig.write_html(os.path.join(output_dir, 'peak_areas_vs_time.html'))
    #
    # def _create_fwhm_plot(self, df, output_dir):
    #     """Create FWHM vs time plot"""
    #     fig = go.Figure()
    #
    #     for peak_id in df['Peak_ID'].unique():
    #         peak_data = df[df['Peak_ID'] == peak_id]
    #         if 'FWHM_nm' in peak_data.columns:
    #             fig.add_trace(go.Scatter(
    #                 x=peak_data['Time'],
    #                 y=peak_data['FWHM_nm'],
    #                 mode='markers+lines',
    #                 name=f'{peak_id} FWHM',
    #                 line=dict(width=2),
    #                 marker=dict(size=6)
    #             ))
    #
    #     fig.update_layout(
    #         title="Peak FWHM vs Time",
    #         xaxis_title="Time (s)",
    #         yaxis_title="FWHM (nm)",
    #         height=500,
    #         template='plotly_white'
    #     )
    #
    #     fig.write_html(os.path.join(output_dir, 'peak_fwhm_vs_time.html'))
    #
    # def _create_dashboard_plot(self, df, output_dir):
    #     """Create comprehensive dashboard plot"""
    #     from plotly.subplots import make_subplots
    #
    #     fig = make_subplots(
    #         rows=2, cols=2,
    #         subplot_titles=('Peak Centers', 'Peak Heights', 'FWHM', 'Areas'),
    #         vertical_spacing=0.1,
    #         horizontal_spacing=0.1
    #     )
    #
    #     colors = px.colors.qualitative.Plotly
    #
    #     for i, peak_id in enumerate(df['Peak_ID'].unique()):
    #         peak_data = df[df['Peak_ID'] == peak_id]
    #         color = colors[i % len(colors)]
    #
    #         # Centers
    #         fig.add_trace(go.Scatter(
    #             x=peak_data['Time'], y=peak_data['Center_nm'],
    #             mode='lines+markers', name=f'{peak_id}',
    #             line=dict(color=color), legendgroup=peak_id
    #         ), row=1, col=1)
    #
    #         # Heights
    #         if 'Height' in peak_data.columns:
    #             fig.add_trace(go.Scatter(
    #                 x=peak_data['Time'], y=peak_data['Height'],
    #                 mode='lines+markers', name=f'{peak_id}',
    #                 line=dict(color=color), legendgroup=peak_id, showlegend=False
    #             ), row=1, col=2)
    #
    #         # FWHM
    #         if 'FWHM_nm' in peak_data.columns:
    #             fig.add_trace(go.Scatter(
    #                 x=peak_data['Time'], y=peak_data['FWHM_nm'],
    #                 mode='lines+markers', name=f'{peak_id}',
    #                 line=dict(color=color), legendgroup=peak_id, showlegend=False
    #             ), row=2, col=1)
    #
    #         # Areas
    #         if 'Area_Under_Curve' in peak_data.columns:
    #             fig.add_trace(go.Scatter(
    #                 x=peak_data['Time'], y=peak_data['Area_Under_Curve'],
    #                 mode='lines+markers', name=f'{peak_id}',
    #                 line=dict(color=color), legendgroup=peak_id, showlegend=False
    #             ), row=2, col=2)
    #
    #     fig.update_layout(
    #         title="Peak Parameters Dashboard",
    #         height=700,
    #         template='plotly_white'
    #     )
    #
    #     fig.update_xaxes(title_text="Time (s)")
    #     fig.update_yaxes(title_text="Center (nm)", row=1, col=1)
    #     fig.update_yaxes(title_text="Height", row=1, col=2)
    #     fig.update_yaxes(title_text="FWHM (nm)", row=2, col=1)
    #     fig.update_yaxes(title_text="Area", row=2, col=2)
    #
    #     fig.write_html(os.path.join(output_dir, 'parameters_dashboard.html'))
    #
    # def _create_quality_plot_enhanced(self, df, output_dir):
    #     """Create enhanced fitting quality plot"""
    #     fig = make_subplots(
    #         rows=2, cols=1,
    #         subplot_titles=('R-squared vs Time', 'Chi-squared vs Time'),
    #         vertical_spacing=0.1
    #     )
    #
    #     # Group by time to get unique time points
    #     time_data = df.groupby('Time').agg({
    #         'R_Squared': 'first',
    #         'Chi_Squared': 'first'
    #     }).reset_index()
    #
    #     fig.add_trace(go.Scatter(
    #         x=time_data['Time'],
    #         y=time_data['R_Squared'],
    #         mode='markers+lines',
    #         name='R²',
    #         line=dict(color='blue', width=2)
    #     ), row=1, col=1)
    #
    #     fig.add_trace(go.Scatter(
    #         x=time_data['Time'],
    #         y=time_data['Chi_Squared'],
    #         mode='markers+lines',
    #         name='χ²',
    #         line=dict(color='red', width=2),
    #         showlegend=False
    #     ), row=2, col=1)
    #
    #     fig.update_layout(
    #         title="Fitting Quality Over Time",
    #         height=600,
    #         template='plotly_white'
    #     )
    #
    #     fig.update_xaxes(title_text="Time (s)")
    #     fig.update_yaxes(title_text="R²", row=1, col=1)
    #     fig.update_yaxes(title_text="χ²", row=2, col=1)
    #
    #     fig.write_html(os.path.join(output_dir, 'fitting_quality_enhanced.html'))
    #
    # def _create_correlation_matrix(self, df, output_dir):
    #     """Create parameter correlation matrix"""
    #     # Select numeric columns for correlation
    #     numeric_cols = ['Center_nm', 'Height', 'Area_Under_Curve', 'FWHM_nm', 'Sigma_nm', 'R_Squared']
    #     available_cols = [col for col in numeric_cols if col in df.columns]
    #
    #     if len(available_cols) < 2:
    #         return
    #
    #     # Calculate correlation matrix
    #     corr_matrix = df[available_cols].corr()
    #
    #     # Create heatmap
    #     fig = go.Figure(data=go.Heatmap(
    #         z=corr_matrix.values,
    #         x=corr_matrix.columns,
    #         y=corr_matrix.columns,
    #         colorscale='RdBu',
    #         zmid=0,
    #         text=corr_matrix.round(2).values,
    #         texttemplate="%{text}",
    #         textfont={"size": 12},
    #         colorbar=dict(title="Correlation")
    #     ))
    #
    #     fig.update_layout(
    #         title="Parameter Correlation Matrix",
    #         height=600,
    #         width=700,
    #         template='plotly_white'
    #     )
    #
    #     fig.write_html(os.path.join(output_dir, 'parameter_correlations.html'))