"""
Resizable Plot Utility for JV Analysis Application
Adds resizable containers to Plotly figures in Jupyter notebooks
"""

__author__ = "Edgar Nandayapa"
__institution__ = "Helmholtz-Zentrum Berlin"
__created__ = "August 2025"

import plotly.graph_objects as go
import plotly.utils
from IPython.display import HTML, display
import json
import uuid
import ipywidgets as widgets


class ResizablePlotWidget:
    """Creates resizable Plotly plots for Jupyter notebooks"""
    
    def __init__(self, fig, title="Plot", initial_width=800, initial_height=600):
        self.fig = fig
        self.title = title
        self.initial_width = initial_width
        self.initial_height = initial_height
        self.plot_id = f"plot_{uuid.uuid4().hex[:8]}"
        
    def display(self):
        """Display the resizable plot"""
        # Update figure layout for responsiveness
        self.fig.update_layout(
            autosize=True,
            margin=dict(l=50, r=50, t=60, b=50)
        )
        
        # Convert figure to JSON
        fig_json = json.dumps(self.fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Create resizable HTML container
        resizable_html = f'''
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        
        <div style="margin: 20px 0;">
            <h4>{self.title}</h4>
            <p style="color: #666; font-size: 12px;">💡 Drag the bottom-right corner to resize the plot</p>
            
            <div id="container-{self.plot_id}" style="
                width: {self.initial_width}px; 
                height: {self.initial_height}px; 
                border: 2px solid #ddd; 
                border-radius: 8px;
                resize: both; 
                overflow: hidden;
                min-width: 400px;
                min-height: 300px;
                max-width: 1400px;
                max-height: 1000px;
                background-color: white;
                position: relative;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                transition: border-color 0.2s;
            ">
                <div id="{self.plot_id}" style="width: 100%; height: 100%;"></div>
                
                <!-- Resize indicator -->
                <div style="
                    position: absolute;
                    bottom: 2px;
                    right: 2px;
                    width: 16px;
                    height: 16px;
                    background: linear-gradient(-45deg, transparent 30%, #999 30%, #999 40%, transparent 40%, transparent 60%, #999 60%, #999 70%, transparent 70%);
                    cursor: se-resize;
                    z-index: 1000;
                    pointer-events: none;
                "></div>
            </div>
        </div>
        
        <style>
        #container-{self.plot_id}:hover {{
            border-color: #007bff;
        }}
        </style>
        
        <script>
        (function() {{
            const figureData = {fig_json};
            const config = {{
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
            }};
            
            function initPlot() {{
                const plotDiv = document.getElementById('{self.plot_id}');
                const container = document.getElementById('container-{self.plot_id}');
                
                if (!plotDiv || !container) {{
                    setTimeout(initPlot, 100);
                    return;
                }}
                
                Plotly.newPlot(plotDiv, figureData.data, figureData.layout, config).then(function() {{
                    console.log('✅ Resizable plot created: {self.title}');
                    
                    if (window.ResizeObserver) {{
                        let resizeTimeout;
                        const observer = new ResizeObserver(function(entries) {{
                            clearTimeout(resizeTimeout);
                            resizeTimeout = setTimeout(function() {{
                                const rect = container.getBoundingClientRect();
                                Plotly.relayout(plotDiv, {{
                                    width: rect.width,
                                    height: rect.height
                                }});
                            }}, 50);
                        }});
                        observer.observe(container);
                    }} else {{
                        // Fallback for older browsers
                        let lastWidth = container.offsetWidth;
                        let lastHeight = container.offsetHeight;
                        
                        setInterval(function() {{
                            const w = container.offsetWidth;
                            const h = container.offsetHeight;
                            if (w !== lastWidth || h !== lastHeight) {{
                                Plotly.relayout(plotDiv, {{width: w, height: h}});
                                lastWidth = w;
                                lastHeight = h;
                            }}
                        }}, 100);
                    }}
                }}).catch(function(err) {{
                    console.error('❌ Plot creation failed:', err);
                }});
            }}
            
            if (typeof Plotly !== 'undefined') {{
                initPlot();
            }} else {{
                let checkCount = 0;
                const checkPlotly = setInterval(function() {{
                    if (typeof Plotly !== 'undefined' || checkCount > 50) {{
                        clearInterval(checkPlotly);
                        if (typeof Plotly !== 'undefined') {{
                            initPlot();
                        }} else {{
                            console.error('❌ Plotly failed to load');
                        }}
                    }}
                    checkCount++;
                }}, 100);
            }}
        }})();
        </script>
        '''
        
        display(HTML(resizable_html))


def create_resizable_plot(fig, title="Plot", width=800, height=600):
    """
    Create a resizable plot from a Plotly figure.
    
    Args:
        fig: Plotly figure object
        title: Plot title to display
        width: Initial width in pixels
        height: Initial height in pixels
    
    Returns:
        ResizablePlotWidget instance
    """
    return ResizablePlotWidget(fig, title, width, height)


def display_resizable_plot(fig, title="Plot", width=800, height=600):
    """
    Convenience function to create and immediately display a resizable plot.
    
    Args:
        fig: Plotly figure object
        title: Plot title to display
        width: Initial width in pixels
        height: Initial height in pixels
    """
    resizable_plot = create_resizable_plot(fig, title, width, height)
    resizable_plot.display()


# Integration helper for your existing PlotManager
class ResizablePlotManager:
    """Enhanced plot manager that creates resizable plots"""
    
    @staticmethod
    def display_plots_resizable(figs, names, container_widget=None):
        """
        Display multiple plots as resizable widgets.
        
        Args:
            figs: List of Plotly figures
            names: List of plot names/titles
            container_widget: Optional widget container for output
        """
        if container_widget:
            with container_widget:
                ResizablePlotManager._display_plots_internal(figs, names)
        else:
            ResizablePlotManager._display_plots_internal(figs, names)
    
    @staticmethod
    def _display_plots_internal(figs, names):
        """Internal method to display plots"""
        from IPython.display import clear_output
        clear_output(wait=True)
        
        print(f"✅ Successfully created {len(figs)} resizable plots")
        print("💡 Drag the bottom-right corner of each plot to resize")
        print()
        
        for i, (fig, name) in enumerate(zip(figs, names)):
            try:
                # Determine appropriate size based on plot type
                if "histogram" in name.lower():
                    width, height = 700, 500
                elif "jv_curve" in name.lower() or "jv curve" in name.lower():
                    width, height = 800, 600
                elif "boxplot" in name.lower():
                    width, height = 900, 600
                else:
                    width, height = 800, 600
                
                # Create and display resizable plot
                display_resizable_plot(fig, name, width, height)
                
            except Exception as e:
                print(f"❌ Error displaying plot {i+1} ({name}): {e}")
                # Fallback to regular display
                try:
                    display(widgets.HTML(f"<h4>{name} (Fallback Display)</h4>"))
                    if hasattr(fig, 'show'):
                        fig.show()
                    else:
                        display(fig)
                except Exception as e2:
                    print(f"❌ Could not display plot {name}: {e2}")


# Example usage and test function
def test_resizable_plot():
    """Test function to demonstrate resizable plots"""
    import numpy as np
    
    # Create sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x) + np.random.normal(0, 0.1, 100)
    y2 = np.cos(x) + np.random.normal(0, 0.1, 100)
    
    # Create test figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines+markers', name='Sin Wave'))
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines+markers', name='Cos Wave'))
    
    fig.update_layout(
        title='Test Resizable Plot',
        xaxis_title='X Values',
        yaxis_title='Y Values',
        template="plotly_white"
    )
    
    # Display as resizable plot
    display_resizable_plot(fig, "Test Resizable Plot - Drag corner to resize!", 800, 500)
    
    print("🎯 Test plot created! Try dragging the bottom-right corner to resize.")


if __name__ == "__main__":
    test_resizable_plot()