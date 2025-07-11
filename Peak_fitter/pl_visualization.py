# pl_visualization.py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.colors import qualitative

class PLVisualization:
    """Class to handle visualization of photoluminescence data"""
    
    def __init__(self):
        self.colorscale = 'Viridis'
        self.peak_colors = qualitative.Plotly
        
    def create_heatmap(self, data_matrix, wavelengths, timestamps, current_time_idx=0):
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
            hovertemplate="Time: %{x:.2f}s<br>Wavelength: %{y:.1f} nm<br>Intensity: %{z:.0f}<extra></extra>",
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
                text="Photoluminescence Heatmap",
                font=dict(size=16, family="Arial", color="darkblue")
            ),
            xaxis_title="Time (s)",
            yaxis_title="Wavelength (nm)",
            height=500,
            width=800,
            margin=dict(l=70, r=100, t=60, b=60),
            font=dict(size=12),
            plot_bgcolor='white'
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
        
    def create_spectrum_plot(self, wavelengths, intensities, fit_result=None):
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
                subplot_titles=('Photoluminescence Spectrum', 'Residuals'),
                row_heights=[0.75, 0.25]
            )
            
            # Main spectrum plot
            fig.add_trace(go.Scatter(
                x=wavelengths,
                y=intensities,
                mode='lines',
                name='Raw Data',
                line=dict(color='black', width=2),
                hovertemplate="Wavelength: %{x:.1f} nm<br>Intensity: %{y:.0f}<extra></extra>"
            ), row=1, col=1)
            
            # Fitted curve
            fig.add_trace(go.Scatter(
                x=wavelengths,
                y=fit_result.best_fit,
                mode='lines',
                name='Fitted Curve',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate="Wavelength: %{x:.1f} nm<br>Fitted: %{y:.0f}<extra></extra>"
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
                            hovertemplate=f"{comp_name}<br>Wavelength: %{{x:.1f}} nm<br>Intensity: %{{y:.0f}}<extra></extra>"
                        ), row=1, col=1)
                        color_idx += 1
                        
            # Residuals
            fig.add_trace(go.Scatter(
                x=wavelengths,
                y=fit_result.residual,
                mode='lines',
                name='Residuals',
                line=dict(color='blue', width=1.5),
                hovertemplate="Wavelength: %{x:.1f} nm<br>Residual: %{y:.0f}<extra></extra>",
                showlegend=False
            ), row=2, col=1)
            
            # Add zero line for residuals
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1, line_width=1)
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=f"Spectrum Analysis (R² = {fit_result.rsquared:.4f})",
                    font=dict(size=16, family="Arial", color="darkblue")
                ),
                height=600,
                width=800,
                margin=dict(l=70, r=50, t=80, b=60),
                font=dict(size=12),
                plot_bgcolor='white',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )
            )
            
            fig.update_xaxes(title_text="Wavelength (nm)", row=2, col=1, showgrid=True)
            fig.update_yaxes(title_text="Intensity", row=1, col=1, showgrid=True)
            fig.update_yaxes(title_text="Residual", row=2, col=1, showgrid=True)
            
        else:
            # Simple single plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=wavelengths,
                y=intensities,
                mode='lines',
                name='Raw Data',
                line=dict(color='darkblue', width=2),
                hovertemplate="Wavelength: %{x:.1f} nm<br>Intensity: %{y:.0f}<extra></extra>"
            ))
            
            fig.update_layout(
                title=dict(
                    text="Photoluminescence Spectrum",
                    font=dict(size=16, family="Arial", color="darkblue")
                ),
                xaxis_title="Wavelength (nm)",
                yaxis_title="Intensity",
                height=500,
                width=800,
                margin=dict(l=70, r=50, t=60, b=60),
                font=dict(size=12),
                plot_bgcolor='white',
                showlegend=False
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
        return fig
        
    def create_time_series_plot(self, timestamps, parameter_values, parameter_name, errors=None):
        """
        Create time series plot of fitting parameters
        
        Parameters:
        -----------
        timestamps : array
            Time values
        parameter_values : array
            Parameter values over time
        parameter_name : str
            Name of the parameter
        errors : array, optional
            Error bars for parameter values
            
        Returns:
        --------
        plotly.graph_objects.Figure: Time series plot
        """
        fig = go.Figure()
        
        if errors is not None:
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=parameter_values,
                error_y=dict(type='data', array=errors),
                mode='markers+lines',
                name=parameter_name,
                line=dict(width=2),
                marker=dict(size=6),
                hovertemplate="Time: %{x:.2f}<br>" + parameter_name + ": %{y:.3f}<extra></extra>"
            ))
        else:
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=parameter_values,
                mode='markers+lines',
                name=parameter_name,
                line=dict(width=2),
                marker=dict(size=6),
                hovertemplate="Time: %{x:.2f}<br>" + parameter_name + ": %{y:.3f}<extra></extra>"
            ))
            
        fig.update_layout(
            title=f"Time Evolution of {parameter_name}",
            xaxis_title="Time",
            yaxis_title=parameter_name,
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
        
    def create_fitting_quality_plot(self, timestamps, r_squared_values, chi_squared_values):
        """
        Create plot showing fitting quality over time
        
        Parameters:
        -----------
        timestamps : array
            Time values
        r_squared_values : array
            R-squared values
        chi_squared_values : array
            Chi-squared values
            
        Returns:
        --------
        plotly.graph_objects.Figure: Fitting quality plot
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('R-squared', 'Chi-squared')
        )
        
        # R-squared plot
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=r_squared_values,
            mode='markers+lines',
            name='R²',
            line=dict(color='blue', width=2),
            marker=dict(size=4),
            hovertemplate="Time: %{x:.2f}<br>R²: %{y:.4f}<extra></extra>"
        ), row=1, col=1)
        
        # Chi-squared plot
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=chi_squared_values,
            mode='markers+lines',
            name='χ²',
            line=dict(color='red', width=2),
            marker=dict(size=4),
            hovertemplate="Time: %{x:.2f}<br>χ²: %{y:.2e}<extra></extra>",
            showlegend=False
        ), row=2, col=1)
        
        fig.update_layout(
            title="Fitting Quality Over Time",
            height=500,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="R²", row=1, col=1)
        fig.update_yaxes(title_text="χ²", row=2, col=1)
        
        return fig
        
    def create_correlation_plot(self, param1_values, param2_values, param1_name, param2_name):
        """
        Create correlation plot between two parameters
        
        Parameters:
        -----------
        param1_values : array
            First parameter values
        param2_values : array
            Second parameter values
        param1_name : str
            First parameter name
        param2_name : str
            Second parameter name
            
        Returns:
        --------
        plotly.graph_objects.Figure: Correlation plot
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=param1_values,
            y=param2_values,
            mode='markers',
            marker=dict(size=6, opacity=0.7),
            hovertemplate=f"{param1_name}: %{{x:.3f}}<br>{param2_name}: %{{y:.3f}}<extra></extra>"
        ))
        
        # Add correlation coefficient
        correlation = np.corrcoef(param1_values, param2_values)[0, 1]
        
        fig.update_layout(
            title=f"Correlation: {param1_name} vs {param2_name} (r = {correlation:.3f})",
            xaxis_title=param1_name,
            yaxis_title=param2_name,
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
        
    def create_peak_evolution_plot(self, timestamps, peak_centers, peak_intensities, peak_widths):
        """
        Create comprehensive peak evolution plot
        
        Parameters:
        -----------
        timestamps : array
            Time values
        peak_centers : dict
            Peak center positions over time
        peak_intensities : dict
            Peak intensities over time
        peak_widths : dict
            Peak widths over time
            
        Returns:
        --------
        plotly.graph_objects.Figure: Peak evolution plot
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Peak Centers', 'Peak Intensities', 'Peak Widths')
        )
        
        color_idx = 0
        for peak_name in peak_centers.keys():
            color = self.peak_colors[color_idx % len(self.peak_colors)]
            
            # Peak centers
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=peak_centers[peak_name],
                mode='markers+lines',
                name=f'{peak_name} Center',
                line=dict(color=color, width=2),
                marker=dict(size=4),
                legendgroup=peak_name,
                hovertemplate=f"Time: %{{x:.2f}}<br>Center: %{{y:.2f}} nm<extra></extra>"
            ), row=1, col=1)
            
            # Peak intensities
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=peak_intensities[peak_name],
                mode='markers+lines',
                name=f'{peak_name} Intensity',
                line=dict(color=color, width=2),
                marker=dict(size=4),
                legendgroup=peak_name,
                showlegend=False,
                hovertemplate=f"Time: %{{x:.2f}}<br>Intensity: %{{y:.0f}}<extra></extra>"
            ), row=2, col=1)
            
            # Peak widths
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=peak_widths[peak_name],
                mode='markers+lines',
                name=f'{peak_name} Width',
                line=dict(color=color, width=2),
                marker=dict(size=4),
                legendgroup=peak_name,
                showlegend=False,
                hovertemplate=f"Time: %{{x:.2f}}<br>Width: %{{y:.2f}} nm<extra></extra>"
            ), row=3, col=1)
            
            color_idx += 1
            
        fig.update_layout(
            title="Peak Evolution Over Time",
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Wavelength (nm)", row=1, col=1)
        fig.update_yaxes(title_text="Intensity", row=2, col=1)
        fig.update_yaxes(title_text="Width (nm)", row=3, col=1)
        
        return fig