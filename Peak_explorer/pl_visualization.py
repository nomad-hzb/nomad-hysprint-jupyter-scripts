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
            hovertemplate="Time Index: %{pointNumber[1]}<br>Time: %{x:.2f}s<br>Wavelength: %{y:.3f} nm<br>Intensity: %{z:.0f}<extra></extra>",
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
                hovertemplate="Wavelength: %{x:.3f} nm<br>Intensity: %{y:.0f}<extra></extra>"
            ), row=1, col=1)

            # Fitted curve
            fig.add_trace(go.Scatter(
                x=wavelengths,
                y=fit_result.best_fit,
                mode='lines',
                name='Fitted Curve',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate="Wavelength: %{x:.3f} nm<br>Fitted: %{y:.0f}<extra></extra>"
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
                            hovertemplate=f"{comp_name}<br>Wavelength: %{{x:.3f}} nm<br>Intensity: %{{y:.0f}}<extra></extra>"
                        ), row=1, col=1)
                        color_idx += 1

            # Residuals
            fig.add_trace(go.Scatter(
                x=wavelengths,
                y=fit_result.residual,
                mode='lines',
                name='Residuals',
                line=dict(color='blue', width=1.5),
                hovertemplate="Wavelength: %{x:.3f} nm<br>Residual: %{y:.0f}<extra></extra>",
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
                hovertemplate="Wavelength: %{x:.3f} nm<br>Intensity: %{y:.0f}<extra></extra>"
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

        return fig