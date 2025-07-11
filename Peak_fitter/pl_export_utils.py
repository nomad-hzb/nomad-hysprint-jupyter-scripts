# pl_export_utils.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class PLExportUtils:
    """Class to handle export of fitting results and analysis"""
    
    def __init__(self):
        self.export_formats = ['xlsx', 'csv', 'json', 'hdf5']
        
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
            filename = f"pl_fitting_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
        # Create Excel writer
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: Peak parameters with calculated values
            peak_params_df = self._create_peak_parameters_dataframe(fitting_results)
            peak_params_df.to_excel(writer, sheet_name='Peak_Parameters', index=False)
            
            # Sheet 2: Summary statistics per time point
            if hasattr(peak_params_df, 'summary_stats'):
                peak_params_df.summary_stats.to_excel(writer, sheet_name='Time_Summary', index=False)
            
            # Sheet 3: Fitting quality metrics
            quality_df = self._create_quality_metrics_dataframe(fitting_results)
            quality_df.to_excel(writer, sheet_name='Fitting_Quality', index=False)
            
            # Sheet 4: Raw fitting data
            raw_df = self._create_raw_data_dataframe(fitting_results)
            raw_df.to_excel(writer, sheet_name='Raw_Data', index=False)
            
            # Sheet 5: Metadata
            metadata_df = self._create_metadata_dataframe(fitting_results, timestamps)
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
        return filename
        
    def _create_summary_dataframe(self, fitting_results):
        """Create summary statistics dataframe"""
        summary_data = []
        
        for idx, result in fitting_results.items():
            if result is None or not result.get('success', False):
                continue
                
            row = {
                'Time_Index': result['index'],
                'Time': result['time'],
                'Fitting_Success': result['success'],
                'R_Squared': result.get('r_squared', np.nan),
                'Chi_Squared': result.get('chi_squared', np.nan),
                'AIC': result.get('aic', np.nan),
                'BIC': result.get('bic', np.nan),
                'Number_of_Parameters': len(result.get('parameters', {}))
            }
            
            summary_data.append(row)
            
        return pd.DataFrame(summary_data)
        
    def _create_peak_parameters_dataframe(self, fitting_results):
        """Create comprehensive peak parameters dataframe"""
        param_data = []
        
        for idx, result in fitting_results.items():
            if result is None or not result.get('success', False):
                continue
                
            base_row = {
                'Time_Index': result['index'],
                'Time': result['time'],
                'R_Squared': result.get('r_squared', np.nan),
                'Chi_Squared': result.get('chi_squared', np.nan)
            }
            
            # Extract peak parameters
            parameters = result.get('parameters', {})
            
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
                    
            # Create rows for each peak with calculated parameters
            for peak_id, params in peak_params.items():
                row = base_row.copy()
                row['Peak_ID'] = peak_id
                
                # Add fitted parameters
                row.update(params)
                
                # Calculate additional parameters
                if 'center' in params:
                    row['Center_nm'] = params['center']
                    
                if 'amplitude' in params and 'sigma' in params:
                    amplitude = params['amplitude']
                    sigma = params['sigma']
                    
                    # Calculate height from amplitude
                    row['Height'] = amplitude / (sigma * np.sqrt(2 * np.pi))
                    
                    # Calculate area under curve (integral of Gaussian)
                    row['Area_Under_Curve'] = amplitude
                    
                    # Calculate FWHM (Full Width at Half Maximum)
                    row['FWHM_nm'] = 2.355 * sigma  # For Gaussian: FWHM = 2.355 * sigma
                    
                    # Calculate peak width at different heights
                    row['Width_at_10_percent'] = 4.29 * sigma  # Width at 10% of peak height
                    row['Width_at_50_percent'] = row['FWHM_nm']  # Same as FWHM
                    
                if 'sigma' in params:
                    row['Sigma_nm'] = params['sigma']
                    
                # Add relative parameters if possible
                if len(peak_params) > 1:
                    centers = [p.get('center', 0) for p in peak_params.values() if 'center' in p]
                    if len(centers) > 1:
                        row['Peak_Separation_from_First'] = row.get('Center_nm', 0) - min(centers)
                        
                param_data.append(row)
                
        df = pd.DataFrame(param_data)
        
        # Add summary statistics for each time point
        if not df.empty:
            summary_stats = []
            for time_idx in df['Time_Index'].unique():
                time_data = df[df['Time_Index'] == time_idx]
                summary_row = {
                    'Time_Index': time_idx,
                    'Time': time_data['Time'].iloc[0],
                    'Total_Peaks': len(time_data),
                    'Total_Area': time_data['Area_Under_Curve'].sum() if 'Area_Under_Curve' in time_data.columns else np.nan,
                    'Average_FWHM': time_data['FWHM_nm'].mean() if 'FWHM_nm' in time_data.columns else np.nan,
                    'Peak_Center_Range': time_data['Center_nm'].max() - time_data['Center_nm'].min() if 'Center_nm' in time_data.columns else np.nan,
                    'Fit_Quality_R2': time_data['R_Squared'].iloc[0] if 'R_Squared' in time_data.columns else np.nan
                }
                summary_stats.append(summary_row)
                
            # Add summary sheet data
            df.summary_stats = pd.DataFrame(summary_stats)
            
        return df
        
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
                'Reduced_Chi_Squared': result.get('chi_squared', np.nan),  # Would need proper calculation
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
        
    def _create_raw_data_dataframe(self, fitting_results):
        """Create raw fitting data dataframe"""
        raw_data = []
        
        for idx, result in fitting_results.items():
            if result is None or not result.get('success', False):
                continue
                
            base_row = {
                'Time_Index': result['index'],
                'Time': result['time']
            }
            
            # Add all parameter values
            for param_name, param_data in result.get('parameters', {}).items():
                row = base_row.copy()
                row['Parameter_Name'] = param_name
                row['Value'] = param_data['value']
                row['Stderr'] = param_data.get('stderr', np.nan)
                row['Min_Bound'] = param_data.get('min', np.nan)
                row['Max_Bound'] = param_data.get('max', np.nan)
                raw_data.append(row)
                
        return pd.DataFrame(raw_data)
        
    def _create_peak_evolution_dataframe(self, fitting_results):
        """Create peak evolution dataframe"""
        evolution_data = []
        
        # Track peak parameters over time
        peak_tracking = {}
        
        for idx, result in fitting_results.items():
            if result is None or not result.get('success', False):
                continue
                
            parameters = result.get('parameters', {})
            
            # Group by peak
            for param_name, param_data in parameters.items():
                if param_name.startswith('p') and '_center' in param_name:
                    peak_id = param_name.split('_')[0]
                    
                    if peak_id not in peak_tracking:
                        peak_tracking[peak_id] = []
                        
                    peak_info = {
                        'Time_Index': result['index'],
                        'Time': result['time'],
                        'Peak_ID': peak_id,
                        'Center': param_data['value'],
                        'Center_Stderr': param_data.get('stderr', np.nan)
                    }
                    
                    # Add other parameters for this peak
                    for other_param in parameters:
                        if other_param.startswith(peak_id + '_') and other_param != param_name:
                            param_type = other_param.split('_', 1)[1]
                            peak_info[param_type] = parameters[other_param]['value']
                            peak_info[f'{param_type}_stderr'] = parameters[other_param].get('stderr', np.nan)
                            
                    peak_tracking[peak_id].append(peak_info)
                    
        # Convert to dataframe
        for peak_id, peak_data in peak_tracking.items():
            evolution_data.extend(peak_data)
            
        return pd.DataFrame(evolution_data)
        
    def _create_metadata_dataframe(self, fitting_results, timestamps):
        """Create metadata dataframe"""
        metadata = {
            'Export_Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Total_Spectra': [len(fitting_results)],
            'Successful_Fits': [sum(1 for r in fitting_results.values() if r and r.get('success', False))],
            'Time_Range_Start': [min(timestamps)],
            'Time_Range_End': [max(timestamps)],
            'Analysis_Software': ['Python PL Analysis App'],
            'Fitting_Library': ['lmfit']
        }
        
        return pd.DataFrame(metadata)
        
    def export_to_csv(self, fitting_results, output_dir=None):
        """Export fitting results to CSV files"""
        if output_dir is None:
            output_dir = f"pl_analysis_csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Export different data types to separate CSV files
        summary_df = self._create_summary_dataframe(fitting_results)
        summary_df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
        
        peak_params_df = self._create_peak_parameters_dataframe(fitting_results)
        peak_params_df.to_csv(os.path.join(output_dir, 'peak_parameters.csv'), index=False)
        
        quality_df = self._create_quality_metrics_dataframe(fitting_results)
        quality_df.to_csv(os.path.join(output_dir, 'fitting_quality.csv'), index=False)
        
        return output_dir
        
    def export_to_json(self, fitting_results, filename=None):
        """Export fitting results to JSON format"""
        if filename is None:
            filename = f"pl_fitting_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
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
        
    def create_summary_plots(self, fitting_results, timestamps, output_dir=None):
        """Create comprehensive summary plots of fitting results"""
        if output_dir is None:
            output_dir = f"pl_analysis_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data for plotting
        peak_params_df = self._create_peak_parameters_dataframe(fitting_results)
        
        if peak_params_df.empty:
            print("No data to plot")
            return output_dir
        
        print(f"Creating plots for {len(peak_params_df)} data points...")
        
        # Plot 1: Peak Centers vs Time
        self._create_centers_plot(peak_params_df, output_dir)
        
        # Plot 2: Peak Heights vs Time  
        self._create_heights_plot(peak_params_df, output_dir)
        
        # Plot 3: Peak Areas vs Time
        self._create_areas_plot(peak_params_df, output_dir)
        
        # Plot 4: FWHM vs Time
        self._create_fwhm_plot(peak_params_df, output_dir)
        
        # Plot 5: Sigma vs Time
        self._create_sigma_plot(peak_params_df, output_dir)
        
        # Plot 6: Fitting Quality vs Time
        self._create_quality_plot_enhanced(peak_params_df, output_dir)
        
        # Plot 7: Peak Summary Dashboard
        self._create_dashboard_plot(peak_params_df, output_dir)
        
        # Plot 8: Parameter Correlations
        self._create_correlation_matrix(peak_params_df, output_dir)
        
        return output_dir
        
    def _create_centers_plot(self, df, output_dir):
        """Create peak centers vs time plot"""
        fig = go.Figure()
        
        for peak_id in df['Peak_ID'].unique():
            peak_data = df[df['Peak_ID'] == peak_id]
            fig.add_trace(go.Scatter(
                x=peak_data['Time'],
                y=peak_data['Center_nm'],
                mode='markers+lines',
                name=f'{peak_id} Center',
                line=dict(width=2),
                marker=dict(size=6)
            ))
            
        fig.update_layout(
            title="Peak Centers vs Time",
            xaxis_title="Time (s)",
            yaxis_title="Peak Center (nm)",
            height=500,
            template='plotly_white'
        )
        
        fig.write_html(os.path.join(output_dir, 'peak_centers_vs_time.html'))
        
    def _create_heights_plot(self, df, output_dir):
        """Create peak heights vs time plot"""
        fig = go.Figure()
        
        for peak_id in df['Peak_ID'].unique():
            peak_data = df[df['Peak_ID'] == peak_id]
            if 'Height' in peak_data.columns:
                fig.add_trace(go.Scatter(
                    x=peak_data['Time'],
                    y=peak_data['Height'],
                    mode='markers+lines',
                    name=f'{peak_id} Height',
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
                
        fig.update_layout(
            title="Peak Heights vs Time",
            xaxis_title="Time (s)",
            yaxis_title="Peak Height",
            height=500,
            template='plotly_white'
        )
        
        fig.write_html(os.path.join(output_dir, 'peak_heights_vs_time.html'))
        
    def _create_areas_plot(self, df, output_dir):
        """Create peak areas vs time plot"""
        fig = go.Figure()
        
        for peak_id in df['Peak_ID'].unique():
            peak_data = df[df['Peak_ID'] == peak_id]
            if 'Area_Under_Curve' in peak_data.columns:
                fig.add_trace(go.Scatter(
                    x=peak_data['Time'],
                    y=peak_data['Area_Under_Curve'],
                    mode='markers+lines',
                    name=f'{peak_id} Area',
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
                
        fig.update_layout(
            title="Peak Areas vs Time",
            xaxis_title="Time (s)",
            yaxis_title="Area Under Curve",
            height=500,
            template='plotly_white'
        )
        
        fig.write_html(os.path.join(output_dir, 'peak_areas_vs_time.html'))
        
    def _create_fwhm_plot(self, df, output_dir):
        """Create FWHM vs time plot"""
        fig = go.Figure()
        
        for peak_id in df['Peak_ID'].unique():
            peak_data = df[df['Peak_ID'] == peak_id]
            if 'FWHM_nm' in peak_data.columns:
                fig.add_trace(go.Scatter(
                    x=peak_data['Time'],
                    y=peak_data['FWHM_nm'],
                    mode='markers+lines',
                    name=f'{peak_id} FWHM',
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
                
        fig.update_layout(
            title="Peak FWHM vs Time",
            xaxis_title="Time (s)",
            yaxis_title="FWHM (nm)",
            height=500,
            template='plotly_white'
        )
        
        fig.write_html(os.path.join(output_dir, 'peak_fwhm_vs_time.html'))
        
    def _create_sigma_plot(self, df, output_dir):
        """Create sigma vs time plot"""
        fig = go.Figure()
        
        for peak_id in df['Peak_ID'].unique():
            peak_data = df[df['Peak_ID'] == peak_id]
            if 'Sigma_nm' in peak_data.columns:
                fig.add_trace(go.Scatter(
                    x=peak_data['Time'],
                    y=peak_data['Sigma_nm'],
                    mode='markers+lines',
                    name=f'{peak_id} Sigma',
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
                
        fig.update_layout(
            title="Peak Sigma vs Time",
            xaxis_title="Time (s)",
            yaxis_title="Sigma (nm)",
            height=500,
            template='plotly_white'
        )
        
        fig.write_html(os.path.join(output_dir, 'peak_sigma_vs_time.html'))
        
    def _create_dashboard_plot(self, df, output_dir):
        """Create comprehensive dashboard plot"""
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Peak Centers', 'Peak Heights', 'FWHM', 'Areas'),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        colors = px.colors.qualitative.Plotly
        
        for i, peak_id in enumerate(df['Peak_ID'].unique()):
            peak_data = df[df['Peak_ID'] == peak_id]
            color = colors[i % len(colors)]
            
            # Centers
            fig.add_trace(go.Scatter(
                x=peak_data['Time'], y=peak_data['Center_nm'],
                mode='lines+markers', name=f'{peak_id}',
                line=dict(color=color), legendgroup=peak_id
            ), row=1, col=1)
            
            # Heights
            if 'Height' in peak_data.columns:
                fig.add_trace(go.Scatter(
                    x=peak_data['Time'], y=peak_data['Height'],
                    mode='lines+markers', name=f'{peak_id}',
                    line=dict(color=color), legendgroup=peak_id, showlegend=False
                ), row=1, col=2)
            
            # FWHM
            if 'FWHM_nm' in peak_data.columns:
                fig.add_trace(go.Scatter(
                    x=peak_data['Time'], y=peak_data['FWHM_nm'],
                    mode='lines+markers', name=f'{peak_id}',
                    line=dict(color=color), legendgroup=peak_id, showlegend=False
                ), row=2, col=1)
            
            # Areas
            if 'Area_Under_Curve' in peak_data.columns:
                fig.add_trace(go.Scatter(
                    x=peak_data['Time'], y=peak_data['Area_Under_Curve'],
                    mode='lines+markers', name=f'{peak_id}',
                    line=dict(color=color), legendgroup=peak_id, showlegend=False
                ), row=2, col=2)
        
        fig.update_layout(
            title="Peak Parameters Dashboard",
            height=700,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Center (nm)", row=1, col=1)
        fig.update_yaxes(title_text="Height", row=1, col=2)
        fig.update_yaxes(title_text="FWHM (nm)", row=2, col=1)
        fig.update_yaxes(title_text="Area", row=2, col=2)
        
        fig.write_html(os.path.join(output_dir, 'parameters_dashboard.html'))
        
    def _create_quality_plot_enhanced(self, df, output_dir):
        """Create enhanced fitting quality plot"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('R-squared vs Time', 'Chi-squared vs Time'),
            vertical_spacing=0.1
        )
        
        # Group by time to get unique time points
        time_data = df.groupby('Time').agg({
            'R_Squared': 'first',
            'Chi_Squared': 'first'
        }).reset_index()
        
        fig.add_trace(go.Scatter(
            x=time_data['Time'],
            y=time_data['R_Squared'],
            mode='markers+lines',
            name='R²',
            line=dict(color='blue', width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=time_data['Time'],
            y=time_data['Chi_Squared'],
            mode='markers+lines',
            name='χ²',
            line=dict(color='red', width=2),
            showlegend=False
        ), row=2, col=1)
        
        fig.update_layout(
            title="Fitting Quality Over Time",
            height=600,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="R²", row=1, col=1)
        fig.update_yaxes(title_text="χ²", row=2, col=1)
        
        fig.write_html(os.path.join(output_dir, 'fitting_quality_enhanced.html'))
        
    def _create_correlation_matrix(self, df, output_dir):
        """Create parameter correlation matrix"""
        # Select numeric columns for correlation
        numeric_cols = ['Center_nm', 'Height', 'Area_Under_Curve', 'FWHM_nm', 'Sigma_nm', 'R_Squared']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) < 2:
            return
            
        # Calculate correlation matrix
        corr_matrix = df[available_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 12},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Parameter Correlation Matrix",
            height=600,
            width=700,
            template='plotly_white'
        )
        
        fig.write_html(os.path.join(output_dir, 'parameter_correlations.html'))
        
    def _create_quality_plot(self, summary_df):
        """Create fitting quality plot"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('R-squared', 'Chi-squared', 'AIC', 'BIC'),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # R-squared
        fig.add_trace(go.Scatter(
            x=summary_df['Time'],
            y=summary_df['R_Squared'],
            mode='markers+lines',
            name='R²',
            line=dict(color='blue')
        ), row=1, col=1)
        
        # Chi-squared
        fig.add_trace(go.Scatter(
            x=summary_df['Time'],
            y=summary_df['Chi_Squared'],
            mode='markers+lines',
            name='χ²',
            line=dict(color='red'),
            showlegend=False
        ), row=1, col=2)
        
        # AIC
        fig.add_trace(go.Scatter(
            x=summary_df['Time'],
            y=summary_df['AIC'],
            mode='markers+lines',
            name='AIC',
            line=dict(color='green'),
            showlegend=False
        ), row=2, col=1)
        
        # BIC
        fig.add_trace(go.Scatter(
            x=summary_df['Time'],
            y=summary_df['BIC'],
            mode='markers+lines',
            name='BIC',
            line=dict(color='orange'),
            showlegend=False
        ), row=2, col=2)
        
        fig.update_layout(
            title="Fitting Quality Metrics Over Time",
            height=600,
            showlegend=True
        )
        
        return fig
        
    def _create_evolution_plot(self, peak_params_df):
        """Create peak evolution plot"""
        if 'Peak_ID' not in peak_params_df.columns:
            return go.Figure()
            
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Peak Centers', 'Peak Amplitudes', 'Peak Widths'),
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        unique_peaks = peak_params_df['Peak_ID'].unique()
        colors = px.colors.qualitative.Plotly
        
        for i, peak_id in enumerate(unique_peaks):
            peak_data = peak_params_df[peak_params_df['Peak_ID'] == peak_id]
            color = colors[i % len(colors)]
            
            # Centers
            if 'center' in peak_data.columns:
                fig.add_trace(go.Scatter(
                    x=peak_data['Time'],
                    y=peak_data['center'],
                    mode='markers+lines',
                    name=f'{peak_id} Center',
                    line=dict(color=color),
                    legendgroup=peak_id
                ), row=1, col=1)
                
            # Amplitudes
            if 'amplitude' in peak_data.columns:
                fig.add_trace(go.Scatter(
                    x=peak_data['Time'],
                    y=peak_data['amplitude'],
                    mode='markers+lines',
                    name=f'{peak_id} Amplitude',
                    line=dict(color=color),
                    legendgroup=peak_id,
                    showlegend=False
                ), row=2, col=1)
                
            # Widths
            if 'sigma' in peak_data.columns:
                fig.add_trace(go.Scatter(
                    x=peak_data['Time'],
                    y=peak_data['sigma'],
                    mode='markers+lines',
                    name=f'{peak_id} Sigma',
                    line=dict(color=color),
                    legendgroup=peak_id,
                    showlegend=False
                ), row=3, col=1)
                
        fig.update_layout(
            title="Peak Parameter Evolution",
            height=800
        )
        
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Wavelength (nm)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=2, col=1)
        fig.update_yaxes(title_text="Sigma", row=3, col=1)
        
        return fig
        
    def _create_correlation_plot(self, peak_params_df):
        """Create parameter correlation plot"""
        if len(peak_params_df) == 0:
            return go.Figure()
            
        # Select numeric columns for correlation
        numeric_cols = peak_params_df.select_dtypes(include=[np.number]).columns
        param_cols = [col for col in numeric_cols if col not in ['Time_Index', 'Time']]
        
        if len(param_cols) < 2:
            return go.Figure()
            
        # Calculate correlation matrix
        corr_matrix = peak_params_df[param_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Parameter Correlation Matrix",
            height=600,
            width=800
        )
        
        return fig
        
    def _create_statistics_plot(self, summary_df):
        """Create statistical summary plot"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('R² Distribution', 'Chi² Distribution', 'Success Rate', 'Parameter Count'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # R-squared distribution
        fig.add_trace(go.Histogram(
            x=summary_df['R_Squared'].dropna(),
            nbinsx=20,
            name='R²',
            showlegend=False
        ), row=1, col=1)
        
        # Chi-squared distribution
        fig.add_trace(go.Histogram(
            x=summary_df['Chi_Squared'].dropna(),
            nbinsx=20,
            name='χ²',
            showlegend=False
        ), row=1, col=2)
        
        # Success rate
        success_counts = summary_df['Fitting_Success'].value_counts()
        fig.add_trace(go.Bar(
            x=success_counts.index.astype(str),
            y=success_counts.values,
            name='Success Rate',
            showlegend=False
        ), row=2, col=1)
        
        # Parameter count distribution
        fig.add_trace(go.Histogram(
            x=summary_df['Number_of_Parameters'],
            nbinsx=10,
            name='Param Count',
            showlegend=False
        ), row=2, col=2)
        
        fig.update_layout(
            title="Statistical Summary",
            height=600,
            showlegend=False
        )
        
        return fig
        
    def create_detailed_report(self, fitting_results, timestamps, wavelengths, 
                            data_matrix, output_dir=None):
        """Create comprehensive analysis report"""
        if output_dir is None:
            output_dir = f"pl_detailed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Export data
        excel_file = self.export_to_excel(fitting_results, timestamps, 
                                        os.path.join(output_dir, 'fitting_results.xlsx'))
        
        # Create plots
        plot_dir = self.create_summary_plots(fitting_results, timestamps, 
                                           os.path.join(output_dir, 'plots'))
        
        # Create HTML report
        html_report = self._create_html_report(fitting_results, timestamps, 
                                             wavelengths, data_matrix, output_dir)
        
        # Create analysis summary
        summary_file = self._create_analysis_summary(fitting_results, timestamps, 
                                                   os.path.join(output_dir, 'analysis_summary.txt'))
        
        return output_dir
        
    def _create_html_report(self, fitting_results, timestamps, wavelengths, 
                          data_matrix, output_dir):
        """Create HTML report with embedded plots"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Photoluminescence Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; }}
                .stats {{ display: flex; justify-content: space-around; }}
                .stat-box {{ background-color: #e9e9e9; padding: 15px; border-radius: 5px; }}
                .plot-container {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Photoluminescence Analysis Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Data Summary</h2>
                <div class="stats">
                    <div class="stat-box">
                        <h3>Total Spectra</h3>
                        <p>{len(fitting_results)}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Successful Fits</h3>
                        <p>{sum(1 for r in fitting_results.values() if r and r.get('success', False))}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Time Range</h3>
                        <p>{min(timestamps):.2f} - {max(timestamps):.2f}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Wavelength Range</h3>
                        <p>{min(wavelengths):.1f} - {max(wavelengths):.1f} nm</p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Fitting Quality</h2>
                <p>Average R²: {np.mean([r.get('r_squared', 0) for r in fitting_results.values() if r and r.get('success', False)]):.4f}</p>
                <iframe src="plots/fitting_quality.html" width="100%" height="600"></iframe>
            </div>
            
            <div class="section">
                <h2>Peak Evolution</h2>
                <iframe src="plots/peak_evolution.html" width="100%" height="800"></iframe>
            </div>
            
            <div class="section">
                <h2>Parameter Correlations</h2>
                <iframe src="plots/parameter_correlations.html" width="100%" height="600"></iframe>
            </div>
            
            <div class="section">
                <h2>Statistical Summary</h2>
                <iframe src="plots/statistics_summary.html" width="100%" height="600"></iframe>
            </div>
            
            <div class="section">
                <h2>Files Generated</h2>
                <ul>
                    <li><a href="fitting_results.xlsx">Complete fitting results (Excel)</a></li>
                    <li><a href="analysis_summary.txt">Analysis summary (Text)</a></li>
                    <li><a href="plots/">Interactive plots (HTML)</a></li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        report_file = os.path.join(output_dir, 'report.html')
        with open(report_file, 'w') as f:
            f.write(html_content)
            
        return report_file
        
    def _create_analysis_summary(self, fitting_results, timestamps, filename):
        """Create text summary of analysis"""
        summary_lines = []
        
        # Header
        summary_lines.append("PHOTOLUMINESCENCE ANALYSIS SUMMARY")
        summary_lines.append("=" * 50)
        summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")
        
        # Data overview
        summary_lines.append("DATA OVERVIEW")
        summary_lines.append("-" * 20)
        summary_lines.append(f"Total spectra: {len(fitting_results)}")
        successful_fits = sum(1 for r in fitting_results.values() if r and r.get('success', False))
        summary_lines.append(f"Successful fits: {successful_fits}")
        summary_lines.append(f"Success rate: {successful_fits/len(fitting_results)*100:.1f}%")
        summary_lines.append(f"Time range: {min(timestamps):.2f} - {max(timestamps):.2f}")
        summary_lines.append("")
        
        # Fitting quality
        r_squared_values = [r.get('r_squared', 0) for r in fitting_results.values() 
                          if r and r.get('success', False)]
        
        if r_squared_values:
            summary_lines.append("FITTING QUALITY")
            summary_lines.append("-" * 20)
            summary_lines.append(f"Average R²: {np.mean(r_squared_values):.4f}")
            summary_lines.append(f"Std R²: {np.std(r_squared_values):.4f}")
            summary_lines.append(f"Min R²: {np.min(r_squared_values):.4f}")
            summary_lines.append(f"Max R²: {np.max(r_squared_values):.4f}")
            summary_lines.append("")
        
        # Peak analysis
        peak_params_df = self._create_peak_parameters_dataframe(fitting_results)
        if not peak_params_df.empty and 'Peak_ID' in peak_params_df.columns:
            summary_lines.append("PEAK ANALYSIS")
            summary_lines.append("-" * 20)
            unique_peaks = peak_params_df['Peak_ID'].unique()
            summary_lines.append(f"Number of peaks tracked: {len(unique_peaks)}")
            
            for peak_id in unique_peaks:
                peak_data = peak_params_df[peak_params_df['Peak_ID'] == peak_id]
                if 'center' in peak_data.columns:
                    centers = peak_data['center'].dropna()
                    if len(centers) > 0:
                        summary_lines.append(f"  {peak_id}:")
                        summary_lines.append(f"    Center range: {centers.min():.1f} - {centers.max():.1f} nm")
                        summary_lines.append(f"    Average center: {centers.mean():.1f} nm")
                        summary_lines.append(f"    Center drift: {centers.max() - centers.min():.1f} nm")
                        
        summary_lines.append("")
        
        # Failed fits
        failed_fits = [r for r in fitting_results.values() if r and not r.get('success', False)]
        if failed_fits:
            summary_lines.append("FAILED FITS")
            summary_lines.append("-" * 20)
            summary_lines.append(f"Number of failed fits: {len(failed_fits)}")
            
            # Group by error type if available
            error_types = {}
            for result in failed_fits:
                error = result.get('error', 'Unknown error')
                error_types[error] = error_types.get(error, 0) + 1
                
            for error, count in error_types.items():
                summary_lines.append(f"  {error}: {count} occurrences")
                
        summary_lines.append("")
        summary_lines.append("END OF SUMMARY")
        
        # Write to file
        with open(filename, 'w') as f:
            f.write('\n'.join(summary_lines))
            
        return filename
        
    def export_for_origin(self, fitting_results, timestamps, wavelengths, filename=None):
        """Export data in format suitable for OriginPro"""
        if filename is None:
            filename = f"pl_data_for_origin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
        # Create comprehensive data table
        data_rows = []
        
        # Header
        header = ['Time', 'Wavelength_nm']
        
        # Add columns for each spectrum
        for i in range(len(timestamps)):
            header.append(f'Intensity_t{i}')
            
        # Add fitted curve columns
        for i in range(len(timestamps)):
            header.append(f'Fitted_t{i}')
            
        # Add residuals columns
        for i in range(len(timestamps)):
            header.append(f'Residual_t{i}')
            
        # Create data matrix
        data_matrix = np.zeros((len(wavelengths), len(header)))
        
        # Fill wavelength column
        for i, wavelength in enumerate(wavelengths):
            data_matrix[i, 1] = wavelength
            
        # Fill time values (repeated for each wavelength)
        for i in range(len(wavelengths)):
            data_matrix[i, 0] = 0  # Placeholder, will be filled with actual time data
            
        # Write to file
        with open(filename, 'w') as f:
            f.write('\t'.join(header) + '\n')
            
            for i, wavelength in enumerate(wavelengths):
                row = [str(timestamps[0]), str(wavelength)]  # Time, Wavelength
                
                # Add intensity data for each time point
                for j, time in enumerate(timestamps):
                    if j < len(fitting_results) and fitting_results.get(j) is not None:
                        # This would need the original data matrix
                        row.append('0')  # Placeholder
                    else:
                        row.append('NaN')
                        
                # Add fitted curves
                for j, time in enumerate(timestamps):
                    result = fitting_results.get(j)
                    if result and result.get('fitted_curve') is not None:
                        if i < len(result['fitted_curve']):
                            row.append(str(result['fitted_curve'][i]))
                        else:
                            row.append('NaN')
                    else:
                        row.append('NaN')
                        
                # Add residuals
                for j, time in enumerate(timestamps):
                    result = fitting_results.get(j)
                    if result and result.get('residuals') is not None:
                        if i < len(result['residuals']):
                            row.append(str(result['residuals'][i]))
                        else:
                            row.append('NaN')
                    else:
                        row.append('NaN')
                        
                f.write('\t'.join(row) + '\n')
                
        return filename