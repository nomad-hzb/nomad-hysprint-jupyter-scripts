# plot_manager.py
"""
Consolidated visualization module
Contains Plotly figure creation and plot management
"""
import config
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.colors import qualitative


class Plotter:
    """Class to handle visualization of photoluminescence data"""

    def __init__(self):
        """Initialize visualization settings"""
        self.colorscale = config.DEFAULT_COLORSCALE
        self.peak_colors = ['green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

    def create_heatmap(self, data_matrix, wavelengths, timestamps, current_time_idx=0, wavelength_unit='nm'):
        """
        Create heatmap visualization of PL data

        Parameters:
        -----------
        data_matrix : array
            Matrix of intensity values (time x wavelength)
        wavelengths : array
            Wavelength values
        timestamps : array
            Time values
        current_time_idx : int
            Index of current time position

        Returns:
        --------
        plotly.graph_objects.Figure: Heatmap figure
        """
        fig = go.Figure()

        # Create heatmap
        fig.add_trace(go.Heatmap(
            z=data_matrix.T,  # Transpose to have wavelength on y-axis
            x=timestamps,
            y=wavelengths,
            colorscale=self.colorscale,
            colorbar=dict(title="Intensity", titleside="right"),
            hovertemplate=f"Time Index: %{{pointNumber[1]}}<br>Time: %{{x:.2f}}s<br>Wavelength: %{{y:.3f}} {wavelength_unit}<br>Intensity: %{{z:.0f}}<extra></extra>",
            name="PL Data"
        ))

        # Add current time position line
        if current_time_idx < len(timestamps):
            current_time = timestamps[current_time_idx]
            fig.add_vline(
                x=current_time,
                line=dict(color="red", width=3),
                annotation_text=f"t = {current_time:.3f}s",
                annotation_position="top",
                annotation=dict(
                    font=dict(color="red", size=12, family="Arial Black"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="red",
                    borderwidth=1
                )
            )

        # Update layout for better visibility
        fig.update_layout(
            title=dict(
                text="Peak Analysis Heatmap",
                font=dict(size=16, family="Arial", color="darkblue")
            ),
            xaxis_title="Time (s)",
            yaxis_title=f"Wavelength ({wavelength_unit})" if wavelength_unit == 'nm' else f"Energy ({wavelength_unit})" if wavelength_unit == 'eV' else f"q ({wavelength_unit})",
            height=500,
            width=800,
            margin=dict(l=70, r=100, t=60, b=60),
            font=dict(size=12),
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.3)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.3)'),
        )

        # Improve axis formatting
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            tickformat='.2f'
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            tickformat='.0f'
        )

        return fig

    def create_spectrum_plot(self, wavelengths, intensities, fit_result=None, wavelength_range=None, wavelength_unit='nm'):
        """
        Create spectrum plot with optional fitting results
        Parameters:
        -----------
        wavelengths : array
            Wavelength values
        intensities : array
            Intensity values
        fit_result : ModelResult, optional
            Fitting result to overlay
        Returns:
        --------
        plotly.graph_objects.Figure: Spectrum plot figure
        """
        if fit_result is not None:
            # Create subplots for main plot and residuals
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Peak Analysis', 'Residuals'),
                row_heights=[0.75, 0.25]
            )
            # Main spectrum plot
            fig.add_trace(go.Scatter(
                x=wavelengths,
                y=intensities,
                mode='lines',
                name='Raw Data',
                line=dict(color='black', width=3),
                hovertemplate=f"Wavelength: %{{x:.3f}} {wavelength_unit}<br>Intensity: %{{y:.0f}}<extra></extra>"
            ), row=1, col=1)
            
            # Fitted curve
            fig.add_trace(go.Scatter(
                x=wavelengths,
                y=fit_result.best_fit,
                mode='lines',
                name='Fitted Curve',
                line=dict(color='red', width=3, dash='dash'),
                hovertemplate=f"Wavelength: %{{x:.3f}} {wavelength_unit}<br>Fitted: %{{y:.0f}}<extra></extra>"
            ), row=1, col=1)
            
            # Individual components if available
            if hasattr(fit_result, 'eval_components'):
                components = fit_result.eval_components()
                color_idx = 0
                for comp_name, comp_values in components.items():
                    if comp_name != 'best_fit':
                        fig.add_trace(go.Scatter(
                            x=wavelengths,
                            y=comp_values,
                            mode='lines',
                            name=comp_name.replace('_', ' ').title(),
                            line=dict(
                                color=self.peak_colors[color_idx % len(self.peak_colors)],
                                width=2,
                                dash='dot'
                            ),
                            hovertemplate=f"{comp_name}<br>Wavelength: %{{x:.3f}} {wavelength_unit}<br>Intensity: %{{y:.0f}}<extra></extra>"
                        ), row=1, col=1)
                        color_idx += 1
                    
            # Residuals
            fig.add_trace(go.Scatter(
                x=wavelengths,
                y=fit_result.residual,
                mode='lines',
                name='Residuals',
                line=dict(color='blue', width=1.5),
                hovertemplate=f"Wavelength: %{{x:.3f}} {wavelength_unit}<br>Residual: %{{y:.0f}}<extra></extra>",
                showlegend=False
            ), row=2, col=1)
            
            fig.update_layout(
                height=config.SPECTRUM_HEIGHT,
                width=config.SPECTRUM_WIDTH,
                template='plotly_white',
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                ),
                hovermode='x unified'
            )
            
            # Add grid to both subplots
            fig.update_xaxes(showgrid=True, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridcolor='lightgray')
            
        else:
            # Simple spectrum plot without fit
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=wavelengths,
                y=intensities,
                mode='lines',
                name='Spectrum',
                line=dict(color='blue', width=3),
                hovertemplate=f"Wavelength: %{{x:.3f}} {wavelength_unit}<br>Intensity: %{{y:.0f}}<extra></extra>"
            ))
            
            fig.update_layout(
                title="Peak Analysis",
                xaxis_title=f"Wavelength ({wavelength_unit})" if wavelength_unit == 'nm' else f"Energy ({wavelength_unit})" if wavelength_unit == 'eV' else f"q ({wavelength_unit})",
                yaxis_title="Intensity",
                height=config.SPECTRUM_HEIGHT,
                width=config.SPECTRUM_WIDTH,
                template='plotly_white',
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridcolor='lightgray'),
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                ),
                hovermode='x unified'
            )

        # Add vertical lines for wavelength range if specified
        if wavelength_range is not None:
            fig.add_vline(
                x=wavelength_range[0], 
                line_dash="dash", 
                line_color="green",
                annotation_text=f"{wavelength_range[0]:.1f} {wavelength_unit}",
                annotation_position="top"
            )
            fig.add_vline(
                x=wavelength_range[1], 
                line_dash="dash", 
                line_color="green",
                annotation_text=f"{wavelength_range[1]:.1f} {wavelength_unit}",
                annotation_position="top"
            )
        
        return fig

    def update_heatmap_line_position(self, fig, timestamps, current_time_idx):
        """
        Update only the red line position on an existing heatmap

        Parameters:
        -----------
        fig : plotly.graph_objects.Figure
            Existing heatmap figure
        timestamps : array
            Time values
        current_time_idx : int
            New time index

        Returns:
        --------
        plotly.graph_objects.Figure: Updated figure
        """
        if current_time_idx >= len(timestamps):
            return fig

        current_time = timestamps[current_time_idx]

        try:
            # Get wavelength range from figure data
            if fig.data and len(fig.data) > 0:
                heatmap_data = fig.data[0]
                if hasattr(heatmap_data, 'y') and heatmap_data.y is not None:
                    y_min = min(heatmap_data.y)
                    y_max = max(heatmap_data.y)
                else:
                    y_min = min(timestamps)
                    y_max = max(timestamps)
            else:
                y_min = min(timestamps)
                y_max = max(timestamps)

            # Clear existing shapes and annotations
            fig.layout.shapes = []
            fig.layout.annotations = []

            # Add new vertical line using add_shape (more reliable)
            fig.add_shape(
                type="line",
                x0=current_time,
                x1=current_time,
                y0=y_min,
                y1=y_max,
                line=dict(color="red", width=3),
            )

            # Add annotation
            fig.add_annotation(
                x=current_time,
                y=y_max,
                text=f"t = {current_time:.3f}s",
                showarrow=False,
                font=dict(color="red", size=12, family="Arial Black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="red",
                borderwidth=1,
                yanchor="bottom"
            )

        except Exception as e:
            print(f"Error in heatmap update: {e}")
            # Return original figure if update fails
            pass

        return fig# visualization/plot_manager.py
"""
Plot manager for Photoluminescence Analysis App
Orchestrates visualization creation and updates
"""
# Plotter class defined above
from exporters import ResultExporter
from utils import debug_print
import config


class PlotManager:
    """Manages plot creation and updates"""
    
    def __init__(self):
        self.visualization = Plotter()
        self.export_utils = ResultExporter()
        
        # Plot storage
        self.heatmap_fig = None
        self.spectrum_fig = None
        
    def create_heatmap(self, data_matrix, wavelengths, timestamps, current_time_idx=0, wavelength_unit='nm'):
        """
        Create heatmap visualization as FigureWidget for efficient updates
        
        Parameters:
        -----------
        data_matrix : array
            Data matrix (time x wavelength)
        wavelengths : array
            Wavelength values
        timestamps : array
            Time values
        current_time_idx : int
            Current time index for position line
            
        Returns:
        --------
        go.FigureWidget: Heatmap figure widget
        """
        debug_print(f"Creating heatmap at time index {current_time_idx}", "PLOT")
        
        import plotly.graph_objects as go
        
        # Create the figure
        fig = self.visualization.create_heatmap(
            data_matrix,
            wavelengths,
            timestamps,
            current_time_idx=current_time_idx,
            wavelength_unit=wavelength_unit
        )
        
        # Convert to FigureWidget for efficient updates
        self.heatmap_fig = go.FigureWidget(fig)
        
        return self.heatmap_fig
    
    def update_heatmap_line(self, timestamps, current_time_idx):
        """
        Update position line on existing heatmap using FigureWidget
        
        Parameters:
        -----------
        timestamps : array
            Time values
        current_time_idx : int
            New time index
            
        Returns:
        --------
        None (updates in place)
        """
        if self.heatmap_fig is None:
            debug_print("Cannot update heatmap line - no heatmap exists", "PLOT")
            return None
        
        try:
            debug_print(f"Updating heatmap line to time index {current_time_idx}", "PLOT")
            
            current_time = timestamps[current_time_idx]
            
            # Update the shape (red line) in place
            # The line is typically the first shape
            with self.heatmap_fig.batch_update():
                self.heatmap_fig.layout.shapes[0].x0 = current_time
                self.heatmap_fig.layout.shapes[0].x1 = current_time
            
            return self.heatmap_fig
            
        except Exception as e:
            debug_print(f"Error updating heatmap line: {e}", "PLOT")
            return None
    
    def create_spectrum_plot(self, wavelengths, intensities, fit_result=None, wavelength_range=None, wavelength_unit='nm'):
        """
        Create spectrum plot with optional fit
        
        Parameters:
        -----------
        wavelengths : array
            Wavelength values
        intensities : array
            Intensity values
        fit_result : ModelResult, optional
            Fitting result to overlay
        wavelength_range : tuple, optional
            (min, max) wavelength range to highlight
            
        Returns:
        --------
        plotly.graph_objects.Figure: Spectrum plot
        """
        debug_print(f"Creating spectrum plot (with_fit={fit_result is not None})", "PLOT")
        
        import plotly.graph_objects as go

        fig = self.visualization.create_spectrum_plot(
            wavelengths,
            intensities,
            fit_result=fit_result,
            wavelength_range=wavelength_range,
            wavelength_unit=wavelength_unit
        )
        
        # Convert to FigureWidget for dynamic updates
        self.spectrum_fig = go.FigureWidget(fig)
        
        return self.spectrum_fig
    
    def create_time_series_plots(self, fitting_results, output_widget=None):
        """
        Create time series plots from fitting results
        
        Parameters:
        -----------
        fitting_results : dict
            Dictionary of fitting results
        output_widget : ipywidgets.Output, optional
            Output widget to display plots
            
        Returns:
        --------
        list: List of created figures
        """
        import pandas as pd
        import plotly.graph_objects as go
        
        debug_print("Creating time series plots", "PLOT")
        
        # Extract peak parameters
        df = self._extract_parameters_for_plotting(fitting_results)
        
        if df is None or df.empty:
            debug_print("No data available for time series plots", "PLOT")
            return []
        
        figures = []
        
        # Create plots for each peak
        peak_columns = [col for col in df.columns if col.startswith('p')]
        peak_ids = list(set([col.split('_')[0] for col in peak_columns]))
        
        # Plot 1: Peak centers vs time
        fig_centers = go.Figure()
        for peak_id in peak_ids:
            center_col = f'{peak_id}_center'
            if center_col in df.columns:
                fig_centers.add_trace(go.Scatter(
                    x=df['time'],
                    y=df[center_col],
                    mode='lines+markers',
                    name=f'{peak_id} center',
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
        
        fig_centers.update_layout(
            title="Peak Centers vs Time",
            xaxis_title="Time (s)",
            yaxis_title="Center (nm)",
            height=400,
            template='plotly_white',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            hovermode='x unified'
        )
        figures.append(fig_centers)
        
        # Plot 2: Peak heights vs time
        fig_heights = go.Figure()
        for peak_id in peak_ids:
            height_col = f'{peak_id}_height'
            if height_col in df.columns:
                fig_heights.add_trace(go.Scatter(
                    x=df['time'],
                    y=df[height_col],
                    mode='lines+markers',
                    name=f'{peak_id} height',
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
        
        fig_heights.update_layout(
            title="Peak Heights vs Time",
            xaxis_title="Time (s)",
            yaxis_title="Height",
            height=400,
            template='plotly_white',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            hovermode='x unified'
        )
        figures.append(fig_heights)
        
        # Plot 3: R-squared vs time
        if 'r_squared' in df.columns:
            fig_quality = go.Figure()
            fig_quality.add_trace(go.Scatter(
                x=df['time'],
                y=df['r_squared'],
                mode='lines+markers',
                name='R²',
                line=dict(width=2, color='blue'),
                marker=dict(size=6)
            ))
            
            fig_quality.update_layout(
                title="Fitting Quality (R²) vs Time",
                xaxis_title="Time (s)",
                yaxis_title="R²",
                height=400,
                template='plotly_white',
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridcolor='lightgray'),
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                ),
                hovermode='x unified'
            )
            figures.append(fig_quality)
        
        # Display in output widget if provided
        if output_widget is not None:
            with output_widget:
                output_widget.clear_output()
                for fig in figures:
                    fig.show(renderer=config.PLOT_RENDERER)
        
        debug_print(f"Created {len(figures)} time series plots", "PLOT")
        return figures
    
    def _extract_parameters_for_plotting(self, fitting_results):
        """Extract parameters from fitting results into DataFrame"""
        import pandas as pd
        import numpy as np
        
        data_rows = []
        
        for idx, result in fitting_results.items():
            if result is None or not result.get('success', False):
                continue
            
            row = {
                'index': result.get('index', idx),
                'time': result['time'],
                'r_squared': result.get('r_squared', np.nan)
            }
            
            # Extract peak parameters
            for param_name, param_data in result.get('parameters', {}).items():
                if param_name.startswith('p'):
                    # Store the parameter value
                    row[param_name] = param_data['value']
                    
                    # Calculate height from amplitude and sigma if needed
                    if 'amplitude' in param_name:
                        peak_id = param_name.split('_')[0]
                        sigma_key = f'{peak_id}_sigma'
                        if sigma_key in result['parameters']:
                            amplitude = param_data['value']
                            sigma = result['parameters'][sigma_key]['value']
                            if sigma > 0:
                                height = amplitude / (sigma * np.sqrt(2 * np.pi))
                                row[f'{peak_id}_height'] = height
            
            data_rows.append(row)
        
        if not data_rows:
            return None
        
        df = pd.DataFrame(data_rows)
        # Sort by time to ensure consecutive lines
        df = df.sort_values('time').reset_index(drop=True)
        return df
    
    def export_plots(self, fitting_results, output_dir):
        """
        Export time series plots to HTML files
        
        Parameters:
        -----------
        fitting_results : dict
            Fitting results
        output_dir : str
            Output directory
        """
        debug_print(f"Exporting plots to {output_dir}", "PLOT")
        
        self.export_utils.export_plots(fitting_results, output_dir)
    
    def get_current_heatmap(self):
        """Get current heatmap figure"""
        return self.heatmap_fig
    
    def get_current_spectrum(self):
        """Get current spectrum figure"""
        return self.spectrum_fig
