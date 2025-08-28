"""
Plot Manager for the Design of Experiments application.
All visualization functions and plot generation using plotly and matplotlib.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Plotting libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Optional matplotlib for some plots
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from data_manager import Variable, VariableType
from utils import ValidationUtils


class PlotManager:
    """Manages all visualization and plotting functionality."""
    
    def __init__(self):
        """Initialize the plot manager."""
        self.validator = ValidationUtils()
        
        # Color palettes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff6600',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        self.categorical_colors = px.colors.qualitative.Set3
        
        # Default plot styling - remove plot_bgcolor completely
        self.default_layout = {
            'font': {'family': 'Arial, sans-serif', 'size': 12},
            'paper_bgcolor': 'white',
            'margin': {'l': 80, 'r': 80, 't': 80, 'b': 80}
        }
    
    def create_plot(self, plot_type: str, data: pd.DataFrame, 
                   variables: List[Variable], algorithm: str = "", **kwargs) -> Optional[go.Figure]:
        """Create a plot based on type and data."""
        if data.empty or not variables:
            return None
        
        # Remove experiment ID column if present
        plot_data = data.copy()
        if 'Experiment_ID' in plot_data.columns:
            plot_data = plot_data.drop('Experiment_ID', axis=1)
        
        try:
            if plot_type == 'splom':
                return self.create_scatter_matrix(plot_data, variables, algorithm=algorithm, **kwargs)
            elif plot_type == 'parallel':
                return self.create_parallel_coordinates(plot_data, variables, algorithm=algorithm, **kwargs)
            elif plot_type == 'distributions':
                return self.create_distribution_plots(plot_data, variables, algorithm=algorithm, **kwargs)
            elif plot_type == '3d_scatter':
                return self.create_3d_scatter(plot_data, variables, algorithm=algorithm, **kwargs)
            elif plot_type == 'correlation':
                return self.create_correlation_heatmap(plot_data, variables, algorithm=algorithm, **kwargs)
            elif plot_type == 'quality':
                return self.create_quality_plots(plot_data, variables, algorithm=algorithm, **kwargs)
            else:
                print(f"Unknown plot type: {plot_type}")
                return None
        
        except Exception as e:
            print(f"Error creating plot: {str(e)}")
            return None
    
    def create_scatter_matrix(self, data: pd.DataFrame, variables: List[Variable], **kwargs) -> go.Figure:
        """Create a scatter plot matrix (SPLOM) for all variable pairs."""
        algorithm = kwargs.get('algorithm', '')
        color_by = kwargs.get('color_by', None)
        numeric_vars = [var for var in variables 
                       if var.type in [VariableType.CONTINUOUS, VariableType.DISCRETE]]
        
        if len(numeric_vars) < 2:
            return self.create_distribution_plots(data, variables, **kwargs)
        
        # Prepare data for SPLOM
        numeric_columns = [var.name for var in numeric_vars]
        splom_data = data[numeric_columns].copy()
        
        # Convert to numeric
        for col in numeric_columns:
            splom_data[col] = pd.to_numeric(splom_data[col], errors='coerce')
        
        # Color mapping
        color_col = None
        if color_by and color_by in data.columns:
            color_col = data[color_by]
        
        # Create SPLOM
        fig = go.Figure(data=go.Splom(
            dimensions=[{
                'label': var.name,
                'values': splom_data[var.name]
            } for var in numeric_vars],
            text=data.index,
            marker=dict(
                color=self.colors['primary'],  # Fixed color
                line=dict(color='white', width=0.8),
                size=5
            )
        ))
        
        # Medium sizing
        n_vars = len(numeric_vars)
        plot_size = max(180, min(250, 900 // n_vars))
        total_size = plot_size * n_vars
        
        fig.update_layout(
            title={
                'text': f"Scatter Plot Matrix (SPLOM) - {algorithm}",
                'x': 0.5,
                'font': {'size': 16}
            },
            height=min(800, total_size + 100),
            width=min(800, total_size + 100),
            plot_bgcolor='#fafafa',
            **self.default_layout
        )
        
        return fig
    
    def create_parallel_coordinates(self, data: pd.DataFrame, variables: List[Variable], **kwargs) -> go.Figure:
        """Create parallel coordinates plot."""
        algorithm = kwargs.get('algorithm', '')
        color_by = kwargs.get('color_by', None)
        if data.empty or not variables:
            return None
            
        # Prepare dimensions for parallel coordinates
        dimensions = []
        plot_data = data.copy()
        
        for var in variables:
            if var.name not in plot_data.columns:
                continue
            
            col_data = plot_data[var.name]
            
            if var.type == VariableType.CATEGORICAL:
                # Convert categorical to numeric for plotting
                unique_vals = sorted(col_data.unique())
                val_to_num = {val: i for i, val in enumerate(unique_vals)}
                numeric_col = col_data.map(val_to_num)
                
                dimension = dict(
                    label=var.name,
                    values=numeric_col,
                    tickvals=list(range(len(unique_vals))),
                    ticktext=list(unique_vals)
                )
            else:
                # Numeric variables
                numeric_col = pd.to_numeric(col_data, errors='coerce')
                dimension = dict(
                    label=var.name,
                    values=numeric_col,
                    range=[numeric_col.min(), numeric_col.max()]
                )
            
            dimensions.append(dimension)
        
        if not dimensions:
            return None
        
        # Color mapping
        if color_by and color_by in data.columns and color_by != 'None':
            color_data = data[color_by]
            if color_data.dtype == 'object':
                unique_vals = sorted(color_data.unique())
                val_to_num = {val: i for i, val in enumerate(unique_vals)}
                color_values = color_data.map(val_to_num)
            else:
                color_values = pd.to_numeric(color_data, errors='coerce')
        else:
            color_values = [0] * len(data)
        
        # Create parallel coordinates plot
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=color_values,
                colorscale='Viridis',
                showscale=True if color_by and color_by != 'None' else False
            ),
            dimensions=dimensions
        ))
        
        fig.update_layout(
            title=f"Parallel Coordinates Plot - {algorithm}",
            height=600,
            **self.default_layout
        )
        
        return fig
    
    def create_distribution_plots(self, data: pd.DataFrame, variables: List[Variable], **kwargs) -> go.Figure:
        """Create distribution plots for all variables."""
        algorithm = kwargs.get('algorithm', '')
        if data.empty or not variables:
            return None
            
        n_vars = len(variables)
        
        # Calculate subplot layout
        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        # Create subplots
        subplot_titles = [var.name for var in variables]
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=subplot_titles
        )
        
        for i, var in enumerate(variables):
            if var.name not in data.columns:
                continue
                
            row = i // n_cols + 1
            col = i % n_cols + 1
            
            col_data = data[var.name]
            
            if var.type == VariableType.CATEGORICAL:
                # Bar chart for categorical variables
                value_counts = col_data.value_counts()
                fig.add_trace(
                    go.Bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        name=var.name,
                        showlegend=False,
                        marker_color=self.categorical_colors[i % len(self.categorical_colors)]
                    ),
                    row=row, col=col
                )
            
            else:
                # Histogram for continuous/discrete variables
                numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                
                fig.add_trace(
                    go.Histogram(
                        x=numeric_data,
                        name=var.name,
                        showlegend=False,
                        marker_color=self.categorical_colors[i % len(self.categorical_colors)],
                        opacity=0.7,
                        nbinsx=min(30, len(numeric_data.unique()) if var.type == VariableType.DISCRETE else 30)
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title=f"Variable Distributions - {algorithm}",
            height=300 * n_rows,
            showlegend=False,
            **self.default_layout
        )
        
        return fig
    
    def create_3d_scatter(self, data: pd.DataFrame, variables: List[Variable], **kwargs) -> go.Figure:
        """Create 3D scatter plot."""
        algorithm = kwargs.get('algorithm', '')
        
        if data.empty or not variables:
            return None
            
        # Get numeric variables
        numeric_vars = [var for var in variables 
                       if var.type in [VariableType.CONTINUOUS, VariableType.DISCRETE]
                       and var.name in data.columns]

        if len(numeric_vars) < 3:
            return self.create_scatter_matrix(data, variables, **kwargs)
        
        # Select first three numeric variables for axes
        x_var = numeric_vars[0].name
        y_var = numeric_vars[1].name
        z_var = numeric_vars[2].name
        
        # Prepare data
        x_data = pd.to_numeric(data[x_var], errors='coerce')
        y_data = pd.to_numeric(data[y_var], errors='coerce')
        z_data = pd.to_numeric(data[z_var], errors='coerce')
        
        # Create 3D scatter
        fig = go.Figure(data=[go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode='markers',
            marker=dict(
                size=5,
                color=self.colors['primary'],
                line=dict(color='white', width=0.5)
            ),
            text=[f"Sample {i+1}" for i in range(len(data))]
        )])
        
        fig.update_layout(
            title=f"3D Scatter Plot - {algorithm}",
            scene=dict(
                xaxis_title=x_var,
                yaxis_title=y_var,
                zaxis_title=z_var
            ),
            height=600,
            **self.default_layout
        )
        
        return fig
    
    def create_correlation_heatmap(self, data: pd.DataFrame, variables: List[Variable], **kwargs) -> go.Figure:
        """Create correlation heatmap for numeric variables."""
        algorithm = kwargs.get('algorithm', '')
        if data.empty or not variables:
            return None
            
        # Get numeric variables
        numeric_vars = [var for var in variables 
                       if var.type in [VariableType.CONTINUOUS, VariableType.DISCRETE]
                       and var.name in data.columns]
        
        if len(numeric_vars) < 2:
            return self.create_distribution_plots(data, variables, **kwargs)
        
        # Prepare numeric data
        numeric_data = data[[var.name for var in numeric_vars]].copy()
        for col in numeric_data.columns:
            numeric_data[col] = pd.to_numeric(numeric_data[col], errors='coerce')
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=f"Variable Correlation Heatmap - {algorithm}",
            height=min(600, 50 * len(numeric_vars) + 200),
            width=min(600, 50 * len(numeric_vars) + 200),
            **self.default_layout
        )
        
        return fig
    
    def create_quality_plots(self, data: pd.DataFrame, variables: List[Variable], **kwargs) -> go.Figure:
        """Create space-filling quality assessment plots."""
        algorithm = kwargs.get('algorithm', '')
        # Create subplots for different quality metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Distance Distribution",
                "Coverage Assessment", 
                "Uniformity Check",
                "Sample Distribution"
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Convert to numeric for distance calculations
        numeric_data = self._convert_to_numeric_for_quality(data, variables)
        
        if numeric_data.size > 0 and len(data) > 1:
            # 1. Distance distribution
            distances = self._calculate_pairwise_distances(numeric_data)
            fig.add_trace(
                go.Histogram(
                    x=distances,
                    name="Distances",
                    showlegend=False,
                    marker_color=self.colors['primary'],
                    opacity=0.7
                ),
                row=1, col=1
            )
            
            # 2. Coverage assessment (range utilization)
            coverage_data = self._calculate_coverage_per_variable(data, variables)
            if coverage_data:
                fig.add_trace(
                    go.Bar(
                        x=list(coverage_data.keys()),
                        y=list(coverage_data.values()),
                        name="Coverage",
                        showlegend=False,
                        marker_color=self.colors['success']
                    ),
                    row=1, col=2
                )
            
            # 3. Uniformity check (deviation from uniform distribution)
            uniformity_scores = self._calculate_uniformity_scores(data, variables)
            if uniformity_scores:
                fig.add_trace(
                    go.Bar(
                        x=list(uniformity_scores.keys()),
                        y=list(uniformity_scores.values()),
                        name="Uniformity",
                        showlegend=False,
                        marker_color=self.colors['info']
                    ),
                    row=2, col=1
                )
            
            # 4. Sample distribution in first two dimensions
            if numeric_data.shape[1] >= 2:
                fig.add_trace(
                    go.Scatter(
                        x=numeric_data[:, 0],
                        y=numeric_data[:, 1],
                        mode='markers',
                        name="Samples",
                        showlegend=False,
                        marker=dict(
                            size=6,
                            color=self.colors['secondary'],
                            line=dict(color='white', width=1)
                        )
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            title=f"Design Space Quality Assessment - {algorithm}",
            **self.default_layout,
            height=800,
            showlegend=False
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Distance", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        
        fig.update_xaxes(title_text="Variable", row=1, col=2)
        fig.update_yaxes(title_text="Coverage (%)", row=1, col=2)
        
        fig.update_xaxes(title_text="Variable", row=2, col=1)
        fig.update_yaxes(title_text="Uniformity Score", row=2, col=1)
        
        if variables and len(variables) >= 2:
            fig.update_xaxes(title_text=variables[0].name, row=2, col=2)
            fig.update_yaxes(title_text=variables[1].name, row=2, col=2)
        
        return fig
    
    def _convert_to_numeric_for_quality(self, data: pd.DataFrame, 
                                      variables: List[Variable]) -> np.ndarray:
        """Convert data to numeric format for quality calculations."""
        if data.empty:
            return np.array([])
        
        numeric_data = np.zeros((len(data), len(variables)))
        
        for i, var in enumerate(variables):
            if var.name not in data.columns:
                continue
            
            col_data = data[var.name]
            
            if var.type == VariableType.CATEGORICAL:
                # Convert categories to numeric codes
                unique_vals = col_data.unique()
                val_to_num = {val: j for j, val in enumerate(unique_vals)}
                numeric_data[:, i] = [val_to_num[val] for val in col_data]
            else:
                numeric_data[:, i] = pd.to_numeric(col_data, errors='coerce')
        
        # Normalize to [0, 1] for consistent calculations
        for i in range(numeric_data.shape[1]):
            col_min, col_max = numeric_data[:, i].min(), numeric_data[:, i].max()
            if col_max > col_min:
                numeric_data[:, i] = (numeric_data[:, i] - col_min) / (col_max - col_min)
        
        return numeric_data
    
    def _calculate_pairwise_distances(self, points: np.ndarray) -> np.ndarray:
        """Calculate all pairwise distances between points."""
        from scipy.spatial.distance import pdist
        
        if len(points) < 2:
            return np.array([])
        
        return pdist(points)
    
    def _calculate_coverage_per_variable(self, data: pd.DataFrame, 
                                       variables: List[Variable]) -> Dict[str, float]:
        """Calculate coverage percentage for each variable."""
        coverage_data = {}
        
        for var in variables:
            if var.name not in data.columns:
                continue
            
            col_data = data[var.name]
            
            if var.type == VariableType.CONTINUOUS:
                # Convert to numeric first to avoid string subtraction
                numeric_data = pd.to_numeric(col_data, errors='coerce')
                data_range = numeric_data.max() - numeric_data.min()
                total_range = var.max_value - var.min_value
                coverage = (data_range / total_range * 100) if total_range > 0 else 100
                
            elif var.type == VariableType.DISCRETE:
                # Discrete value coverage
                possible_values = set(var.get_discrete_values())
                covered_values = set(col_data.unique())
                coverage = (len(covered_values) / len(possible_values) * 100)
                
            elif var.type == VariableType.CATEGORICAL:
                # Category coverage
                covered_categories = set(col_data.unique())
                total_categories = set(var.categories)
                coverage = (len(covered_categories) / len(total_categories) * 100)
            
            coverage_data[var.name] = coverage
        
        return coverage_data
    
    def _calculate_uniformity_scores(self, data: pd.DataFrame, 
                                   variables: List[Variable]) -> Dict[str, float]:
        """Calculate uniformity scores for each variable."""
        uniformity_scores = {}
        
        for var in variables:
            if var.name not in data.columns:
                continue
            
            col_data = data[var.name]
            
            if var.type == VariableType.CATEGORICAL:
                # Calculate entropy-based uniformity for categorical
                value_counts = col_data.value_counts()
                if len(value_counts) <= 1:
                    uniformity = 1.0
                else:
                    probabilities = value_counts / value_counts.sum()
                    entropy = -np.sum(probabilities * np.log2(probabilities))
                    max_entropy = np.log2(len(value_counts))
                    uniformity = entropy / max_entropy if max_entropy > 0 else 1.0
            
            else:
                # For continuous/discrete: use coefficient of variation of histogram
                numeric_data = pd.to_numeric(col_data, errors='coerce').dropna()
                if len(numeric_data) < 2:
                    uniformity = 1.0
                else:
                    # Create histogram and check uniformity
                    n_bins = min(20, len(numeric_data.unique()) if var.type == VariableType.DISCRETE else 20)
                    hist, _ = np.histogram(numeric_data, bins=n_bins)
                    
                    # Calculate coefficient of variation (lower = more uniform)
                    if np.mean(hist) > 0:
                        cv = np.std(hist) / np.mean(hist)
                        uniformity = 1.0 / (1.0 + cv)  # Convert to uniformity score
                    else:
                        uniformity = 0.0
            
            uniformity_scores[var.name] = uniformity
        
        return uniformity_scores
    
    def get_available_plot_types(self) -> Dict[str, str]:
        """Get available plot types with descriptions."""
        return {
            'splom': 'Scatter Plot Matrix - All variable pairs',
            'parallel': 'Parallel Coordinates - Multi-dimensional view',
            'distributions': 'Distribution Plots - Individual variable histograms',
            '3d_scatter': '3D Scatter Plot - Three-dimensional visualization',
            'correlation': 'Correlation Heatmap - Variable relationships',
            'quality': 'Space-Filling Quality - Design assessment plots'
        }