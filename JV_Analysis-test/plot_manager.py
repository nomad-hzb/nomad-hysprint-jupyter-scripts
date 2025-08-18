"""
Plot Management Module
Handles all plotting operations including JV curves, boxplots, and histograms.
Extracted from main.py for better organization.
"""

__author__ = "Edgar Nandayapa"
__institution__ = "Helmholtz-Zentrum Berlin"
__created__ = "August 2025"

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory for shared modules
parent_dir = os.path.dirname(os.getcwd())
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from utils import save_combined_excel_data
except ImportError:
    # Fallback if utils not available
    def save_combined_excel_data(*args, **kwargs):
        return None


def is_running_in_jupyter():
    """Check if code is running in Jupyter notebook"""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


# ENHANCED plotting_string_action function
def plotting_string_action(plot_list, data, supp, is_voila=False, color_scheme=None):
    """
    Main plotting function that processes plot codes and creates figures.
    FIXED to ensure it uses filtered data correctly
    """
    # CRITICAL: Unpack data correctly
    filtered_jv, complete_jv, filtered_curves = data
    omitted_jv, filter_pars, is_conditions, path, samples = supp

    # DEBUG: Verify we're getting the right data
    print(f"🔍 Plot function received:")
    print(f"   - Filtered JV data: {len(filtered_jv)} records")
    print(f"   - Complete JV data: {len(complete_jv)} records") 
    print(f"   - Filtered curves: {len(filtered_curves)} records")
    print(f"   - Omitted data: {len(omitted_jv)} records")

    # Create plot manager
    plot_manager = PlotManager()
    plot_manager.set_output_path(path)

    if color_scheme is None:
        color_scheme = [
            'rgba(93, 164, 214, 0.7)', 'rgba(255, 144, 14, 0.7)', 
            'rgba(44, 160, 101, 0.7)', 'rgba(255, 65, 54, 0.7)', 
            'rgba(207, 114, 255, 0.7)', 'rgba(127, 96, 0, 0.7)',
            'rgba(255, 140, 184, 0.7)', 'rgba(79, 90, 117, 0.7)'
        ]

    # Mapping dictionaries for plot codes
    varx_dict = {"a": "sample", "b": "cell", "c": "direction", "d": "ilum", "e": "batch", "g": "condition", "s": "status"}
    vary_dict = {"v": "voc", "j": "jsc", "f": "ff", "p": "pce", "u": "vmpp", "i": "jmpp", "m": "pmpp", "r": "rser", "h": "rshu"}

    fig_list = []
    fig_names = []
    
    # Convert plot selections to codes if needed
    if isinstance(plot_list[0], tuple):
        plot_codes = plot_list_from_voila(plot_list)
    else:
        plot_codes = plot_list

    for pl in plot_codes:
        print(f"Creating plot for code: {pl}")
        
        # Check if there is "condition" requirement
        if "g" in pl and not is_conditions:
            print(f"Skipping {pl} - no conditions available")
            continue
            
        # Extract variables from plot code
        var_x = next((varx_dict[key] for key in varx_dict if key in pl), None)
        var_y = next((vary_dict[key] for key in vary_dict if key in pl), None)

        try:
            # Check for combination plots FIRST (before single variable plots)
            if "sg" in pl and var_y:
                # Status and Variable combination plots
                print(f"Creating status-variable combination plots")
                figs, fig_names_combo = plot_manager.create_combination_plots(
                    filtered_jv, var_y, 'sg', [omitted_jv, filter_pars], colors=color_scheme
                )
                fig_list.extend(figs)
                fig_names.extend(fig_names_combo)
                print(f"✅ Successfully created {len(figs)} status-variable plots")
                continue
            elif "cg" in pl and var_y:
                # Direction and Variable combination plots
                print(f"Creating direction-variable combination plots")
                figs, fig_names_combo = plot_manager.create_combination_plots(
                    filtered_jv, var_y, 'cg', [omitted_jv, filter_pars], colors=color_scheme
                )
                fig_list.extend(figs)
                fig_names.extend(fig_names_combo)
                print(f"✅ Successfully created {len(figs)} direction-variable plots")
                continue
            elif "bg" in pl and var_y:
                # Cell and Variable combination plots
                print(f"Creating cell-variable combination plots")
                figs, fig_names_combo = plot_manager.create_combination_plots(
                    filtered_jv, var_y, 'bg', [omitted_jv, filter_pars], colors=color_scheme
                )
                fig_list.extend(figs)
                fig_names.extend(fig_names_combo)
                print(f"✅ Successfully created {len(figs)} cell-variable plots")
                continue
            elif "B" in pl and var_x and var_y:
                # Regular boxplot - USE FILTERED DATA
                print(f"Creating boxplot with {len(filtered_jv)} filtered records")
                fig, fig_name, _ = plot_manager.create_boxplot(
                    filtered_jv, var_x, var_y, 
                    [omitted_jv, filter_pars], 
                    "data", colors=color_scheme
                )
            elif "J" in pl and var_x and var_y:
                # Omitted data boxplot
                print(f"Creating omitted data boxplot with {len(omitted_jv)} omitted records")
                fig, fig_name, _ = plot_manager.create_boxplot(
                    omitted_jv, var_x, var_y, 
                    [filtered_jv, filter_pars], 
                    "junk"
                )
            elif "H" in pl and var_y:
                # Histogram - USE FILTERED DATA
                print(f"Creating histogram with {len(filtered_jv)} filtered records")
                fig, fig_name = plot_manager.create_histogram(filtered_jv, var_y)
            elif "Cw" in pl:
                # Best device JV curve - USE FILTERED DATA
                print(f"Creating JV curve from {len(filtered_jv)} filtered records")
                fig, fig_name = plot_manager.create_jv_best_device_plot(filtered_jv, filtered_curves)
            else:
                print(f"Plot code {pl} not fully implemented yet")
                continue

            # Only append single figures (combination plots already added above)
            if 'fig' in locals():
                fig_list.append(fig)
                fig_names.append(fig_name)
                print(f"✅ Successfully created: {fig_name}")
            
        except Exception as e:
            print(f"❌ Error creating plot {pl}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"📊 Total plots created: {len(fig_list)}")
    return fig_list, fig_names


def plot_list_from_voila(plot_list):
    """Convert plot selections from UI to plot codes"""
    jvc_dict = {'Voc': 'v', 'Jsc': 'j', 'FF': 'f', 'PCE': 'p', 'R_ser': 'r', 'R_shu': 'h', 'V_mpp': 'u', 'J_mpp': 'i', 'P_mpp': 'm'}
    box_dict = {'by Batch': 'e', 'by Variable': 'g', 'by Sample': 'a', 'by Cell': 'b', 'by Scan Direction': 'c',
                'by Status': 's', 'by Status and Variable': 'sg', 'by Direction and Variable': 'cg', 'by Cell and Variable': 'bg'}

    cur_dict = {'All cells': 'Cy', 'Only working cells': 'Cz', 'Only not working cells': 'Co', 'Best device only': 'Cw', 'Separated by cell': 'Cx', 'Separated by substrate': 'Cd'}

    new_list = []
    for plot in plot_list:
        code = ''
        plot_type, option1, option2 = plot
        
        if "omitted" in plot_type:
            code += "J"
            code += jvc_dict.get(option1, '')
            code += box_dict.get(option2, '')
        elif "Boxplot" in plot_type:
            code += "B"
            code += jvc_dict.get(option1, '')
            code += box_dict.get(option2, '')
        elif "Histogram" in plot_type:
            code += "H"
            code += jvc_dict.get(option1, '')
        elif "JV Curve" in plot_type:
            code += cur_dict.get(option1, '')
        
        if code:
            new_list.append(code)

    print(f"🔍 Generated plot codes: {new_list}")
    return new_list

class PlotManager:
    """Manages all plotting operations for JV analysis"""
    
    def __init__(self):
        self.plot_output_path = ""
    
    def set_output_path(self, path):
        """Set the output path for saving plots"""
        self.plot_output_path = path
    
    def create_jv_best_device_plot(self, jvc_data, curves_data):
        """Plot the JV curve of the best device"""
        print(f"📊 Creating JV best device plot from {len(jvc_data)} records")
        
        # Find best device
        index_num = jvc_data["PCE(%)"].idxmax()
        sample = jvc_data.loc[index_num]["sample"]
        cell = jvc_data.loc[index_num]["cell"]
        
        print(f"   Best device: {sample} (Cell {cell}) with PCE = {jvc_data.loc[index_num]['PCE(%)']:.2f}%")

        # Filter data to focus on best device
        focus = curves_data.loc[(curves_data["sample"] == sample) & (curves_data["cell"] == cell)]

        if len(focus) == 0:
            sample = jvc_data.loc[index_num]["identifier"]
            focus = curves_data.loc[(curves_data["sample"] == sample) & (curves_data["cell"] == cell)]

        plotted = focus.copy().drop(
            columns=["index", "sample", "cell", "direction", "ilum", "batch", "condition"]).set_index(["variable"]).T
        dire = focus.loc[(focus["variable"] == "Voltage (V)")]["direction"].values
        ilum = focus.loc[(focus["variable"] == "Voltage (V)")]["ilum"].values

        # Create Plotly figure
        fig = go.Figure()

        # Add x and y axis lines at 0
        fig.add_shape(type="line", x0=-0.2, y0=0, x1=1.35, y1=0, line=dict(color="gray", width=2))
        fig.add_shape(type="line", x0=0, y0=-25, x1=0, y1=3, line=dict(color="gray", width=2))

        # Plot each direction
        for c, p in enumerate(dire):
            x = plotted["Voltage (V)"].iloc[:, c]
            y = plotted["Current Density(mA/cm2)"].iloc[:, c]

            marker_symbol = 'x' if dire[c] == "Reverse" else 'circle'
            
            fig.add_trace(go.Scatter(
                x=x, 
                y=y,
                mode='lines+markers',
                marker=dict(symbol=marker_symbol),
                name=f"{dire[c]} ({ilum[c]})",
                hovertemplate='Voltage: %{x:.3f} V<br>Current Density: %{y:.3f} mA/cm²<br>%{text}',
                text=[f"Sample: {sample}, Cell: {cell}" for _ in x]
            ))

        # Get JV characteristics values and add MPP points
        df_rev = jvc_data.loc[(jvc_data["sample"] == sample) & (jvc_data["cell"] == cell) & (jvc_data["direction"] == "Reverse")]
        df_for = jvc_data.loc[(jvc_data["sample"] == sample) & (jvc_data["cell"] == cell) & (jvc_data["direction"] == "Forward")]

        # Extract values and add annotations
        char_vals = ['Voc(V)', 'Jsc(mA/cm2)', 'FF(%)', 'PCE(%)']
        char_rev = [df_rev[cv].values[0] for cv in char_vals]
        char_for = [df_for[cv].values[0] for cv in char_vals]

        # Add MPP points
        v_f = df_for['V_mpp(V)'].values[0]
        v_r = df_rev['V_mpp(V)'].values[0]
        j_f = df_for['J_mpp(mA/cm2)'].values[0]
        j_r = df_rev['J_mpp(mA/cm2)'].values[0]

        fig.add_trace(go.Scatter(x=[v_f], y=[j_f], mode='markers', marker=dict(color='red', size=10), name='Forward MPP'))
        fig.add_trace(go.Scatter(x=[v_r], y=[j_r], mode='markers', marker=dict(color='red', size=10, symbol='x'), name='Reverse MPP'))

        # Add JV information as annotations
        text_rev = f"Rev:<br>Voc: {char_rev[0]:.2f} V<br>Jsc: {char_rev[1]:.1f} mA/cm²<br>FF: {char_rev[2]:.1f}%<br>PCE: {char_rev[3]:.1f}%"
        text_for = f"For:<br>Voc: {char_for[0]:.2f} V<br>Jsc: {char_for[1]:.1f} mA/cm²<br>FF: {char_for[2]:.1f}%<br>PCE: {char_for[3]:.1f}%"

        fig.add_annotation(x=0.24, y=-5, text=text_rev, showarrow=False, font=dict(size=12), align="left")
        fig.add_annotation(x=0.55, y=-5, text=text_for, showarrow=False, font=dict(size=12), align="left")
        fig.add_annotation(x=0.20, y=1.5, text=f"Sample: {sample} ({cell}) - FILTERED DATA", showarrow=False, font=dict(size=13), align="left")

        # Update layout
        fig.update_layout(
            title="JV Curve - Best Device (From Filtered Data)",
            xaxis_title='Voltage [V]',
            yaxis_title='Current Density [mA/cm²]',
            xaxis=dict(range=[-0.2, 1.35]),
            yaxis=dict(range=[-25, 3]),
            template="plotly_white"
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

        sample_name = "JV_best_device_filtered.html"
        if not is_running_in_jupyter():
            fig.write_html(os.path.join(self.plot_output_path, sample_name))

        return fig, sample_name
    
    def create_boxplot(self, data, var_x, var_y, filtered_info, datatype="data", wb=None, colors=None):
        """Create a boxplot with statistical analysis - ENHANCED with data verification"""
        print(f"📊 Creating boxplot with {len(data)} records for {var_y} by {var_x}")
        
        names_dict = {
            "voc": 'Voc(V)', "jsc": 'Jsc(mA/cm2)', "ff": 'FF(%)', "pce": 'PCE(%)',
            "vmpp": 'V_mpp(V)', "jmpp": 'J_mpp(mA/cm2)', "pmpp": 'P_mpp(mW/cm2)',
            "rser": 'R_series(Ohmcm2)', "rshu": 'R_shunt(Ohmcm2)'
        }
        var_name_y = names_dict[var_y]
        trash, filters = filtered_info

        # Show what samples are included in this plot
        unique_samples = data['sample'].nunique()
        unique_cells = data.groupby('sample')['cell'].nunique().sum()
        print(f"   - Unique samples: {unique_samples}")
        print(f"   - Unique sample-cell combinations: {unique_cells}")

        try:
            data["sample"] = data["sample"].astype(int)
        except ValueError:
            pass

        data = data.copy()  # Don't modify original data
        data['Jsc(mA/cm2)'] = data['Jsc(mA/cm2)'].abs()

        # Calculate statistics 
        descriptor = data.groupby(var_x)[var_name_y].describe()

        # Ordering
        orderc = descriptor.sort_index()["count"].index

        # Create dictionaries to map categories to their counts
        data_counts = data.groupby(var_x)[var_name_y].count().to_dict()
        trash_counts = trash.groupby(var_x)[var_name_y].count().to_dict() if not trash.empty else {}

        # Create figure
        fig = go.Figure()
        
        # Use provided color scheme or default
        if colors is None:
            colors = [
                'rgba(93, 164, 214, 0.7)', 'rgba(255, 144, 14, 0.7)', 
                'rgba(44, 160, 101, 0.7)', 'rgba(255, 65, 54, 0.7)', 
                'rgba(207, 114, 255, 0.7)', 'rgba(127, 96, 0, 0.7)',
                'rgba(255, 140, 184, 0.7)', 'rgba(79, 90, 117, 0.7)'
            ]
        
        # Add each category's boxplot
        for i, category in enumerate(orderc):
            category_data = data[data[var_x] == category][var_name_y].dropna()
            if not category_data.empty:
                data_count = data_counts.get(category, 0)
                trash_count = trash_counts.get(category, 0)
                median = category_data.median()
                mean = category_data.mean()
                
                category_name = f"{category} (n={data_count})" if trash_count == 0 else f"{category} ({data_count}/{data_count + trash_count})"
                
                fig.add_trace(go.Box(
                    y=category_data,
                    name=category_name,
                    boxpoints='all',
                    pointpos=0,
                    jitter=0.5,
                    whiskerwidth=0.4,
                    marker=dict(size=5, opacity=0.7, color='rgba(0,0,0,0.7)'),
                    line=dict(width=1.5),
                    fillcolor=colors[i % len(colors)],
                    boxmean=True,
                    width=0.8,
                    hovertemplate=(
                        f"<b>{category}</b><br>" +
                        "Value: %{y:.3f}<br>" +
                        f"Median: {median:.3f}<br>" +
                        f"Mean: {mean:.3f}<br>" +
                        f"Count: {data_count}"
                    )
                ))
        
        # Create title with data info
        title_text = f"Boxplot of {var_y} by {var_x}" + (" (filtered out)" if datatype == "junk" else " (filtered data)")
        subtitle = f"Data from {len(data)} measurements across {data[var_x].nunique()} {var_x} categories (after filtering)"
        
        fig.update_layout(
            title=f"{title_text}<br><sup>{subtitle}</sup>",
            xaxis_title=var_x,
            yaxis_title=var_name_y,
            boxmode='group',
            boxgap=0.05,
            boxgroupgap=0.1,
            template="plotly_white",
            margin=dict(l=40, r=40, t=100, b=80),
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Rotate x-axis labels if many categories
        if len(orderc) > 4:
            fig.update_layout(xaxis=dict(tickangle=-10, tickfont=dict(size=10)))
        
        # Save to Excel if workbook provided
        if wb:
            try:
                from utils import save_combined_excel_data
                wb = save_combined_excel_data(self.plot_output_path, wb, data, filtered_info, var_x, var_name_y, var_y, descriptor)
            except ImportError:
                pass  # Skip Excel save if utils not available

        sample_name = f"boxplotj_{var_y}_by_{var_x}.html" if datatype == "junk" else f"boxplot_{var_y}_by_{var_x}.html"

        if not is_running_in_jupyter():
            fig.write_html(os.path.join(self.plot_output_path, sample_name))

        return fig, sample_name, wb
    
    def create_histogram(self, df, var_y, colors=None):
        """Create a histogram with statistics"""
        print(f"📊 Creating histogram with {len(df)} records for {var_y}")
        
        names_dict = {
            'voc': 'Voc(V)', 'jsc': 'Jsc(mA/cm2)', 'ff': 'FF(%)', 'pce': 'PCE(%)',
            'vmpp': 'V_mpp(V)', 'jmpp': 'J_mpp(mA/cm2)', 'pmpp': 'P_mpp(mW/cm2)',
            'rser': 'R_series(Ohmcm2)', 'rshu': 'R_shunt(Ohmcm2)'
        }

        pl_y = names_dict[var_y]

        # Determine number of bins
        bins = {"voc": 20, "jsc": 30}.get(var_y, 40)

        # Create histogram
        fig = go.Figure()
        
        primary_color = colors[0] if colors else 'rgba(0, 0, 255, 0.6)'
        line_color = colors[0].replace('0.6', '1.0') if colors and 'rgba' in colors[0] else 'rgba(0, 0, 255, 1)'
        
        fig.add_trace(go.Histogram(
            x=df[pl_y],
            marker=dict(
                color=primary_color,
                line=dict(color=line_color, width=1)
            ),
            hovertemplate=f'{pl_y}: %{{x:.3f}}<br>Count: %{{y}}<extra></extra>'
        ))
        
        # Add KDE if enough data points
        if len(df) > 5:
            try:
                from scipy import stats
                kde_x = np.linspace(df[pl_y].min(), df[pl_y].max(), 100)
                kde = stats.gaussian_kde(df[pl_y].dropna())
                kde_y = kde(kde_x) * len(df) * (df[pl_y].max() - df[pl_y].min()) / bins
                
                fig.add_trace(go.Scatter(
                    x=kde_x, y=kde_y, mode='lines',
                    line=dict(color='red', width=2), name='KDE', hoverinfo='skip'
                ))
            except ImportError:
                pass  # Skip KDE if scipy not available
        
        # Add statistics annotations
        mean_val = df[pl_y].mean()
        median_val = df[pl_y].median()
        std_val = df[pl_y].std()
        
        stats_text = f"Mean: {mean_val:.3f}<br>Median: {median_val:.3f}<br>Std Dev: {std_val:.3f}<br>Count: {len(df)} (filtered)"
        
        fig.add_annotation(
            x=0.95, y=0.95, xref="paper", yref="paper",
            text=stats_text, showarrow=False, font=dict(size=12), align="right",
            bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="black", borderwidth=1, borderpad=4
        )
        
        fig.update_layout(
            title=f"Histogram of {pl_y} (Filtered Data)",
            xaxis_title=pl_y,
            yaxis_title="Frequency",
            template="plotly_white",
            bargap=0.1,
            hovermode='closest'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

        sample_name = f"histogram_{var_y}_filtered.html"
        if not is_running_in_jupyter():
            fig.write_html(os.path.join(self.plot_output_path, sample_name))

        return fig, sample_name

    def create_combination_plots(self, data, var_y, combination_type, filtered_info, colors=None):
        """Create multiple plots separated by condition/direction/cell and grouped by status/condition"""
        print(f"📊 Creating combination plots: {combination_type}")
        
        names_dict = {
            "voc": 'Voc(V)', "jsc": 'Jsc(mA/cm2)', "ff": 'FF(%)', "pce": 'PCE(%)',
            "vmpp": 'V_mpp(V)', "jmpp": 'J_mpp(mA/cm2)', "pmpp": 'P_mpp(mW/cm2)',
            "rser": 'R_series(Ohmcm2)', "rshu": 'R_shunt(Ohmcm2)'
        }
        var_name_y = names_dict[var_y]
        
        # Filter out dark measurements (D1, D2, etc.) for these combination plots
        data = data[~data['status'].str.startswith('D')].copy()
        print(f"   Filtered out dark measurements, {len(data)} records remaining")
        
        # Define what we separate plots by and what we group within plots
        plot_config = {
            'sg': {
                'primary_var': 'condition', 
                'primary_label': 'Condition',
                'secondary_var': 'status',
                'secondary_label': 'Status'
            },
            'cg': {
                'primary_var': 'condition', 
                'primary_label': 'Condition',
                'secondary_var': 'direction',
                'secondary_label': 'Direction'
            },
            'bg': {
                'primary_var': 'condition', 
                'primary_label': 'Condition', 
                'secondary_var': 'cell',
                'secondary_label': 'Cell'
            }
        }
        
        if combination_type not in plot_config:
            print(f"Warning: Unknown combination type {combination_type}")
            return [], []
        
        config = plot_config[combination_type]
        primary_var = config['primary_var']
        primary_label = config['primary_label']
        secondary_var = config['secondary_var'] 
        secondary_label = config['secondary_label']
        
        # Check if required columns exist
        missing_cols = []
        for col in [primary_var, secondary_var]:
            if col not in data.columns:
                missing_cols.append(col)
                
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols} in data")
            return [], []
        
        # Get unique values for primary variable (what we separate plots by)
        primary_values = sorted(data[primary_var].unique())
        
        if not primary_values:
            print(f"Warning: No data found for primary variable {primary_var}")
            return [], []
        
        figures = []
        figure_names = []
        
        # Create separate plot for each primary variable value
        for primary_val in primary_values:
            primary_data = data[data[primary_var] == primary_val]
            
            if primary_data.empty:
                continue
                
            # Get unique secondary values within this primary group
            secondary_values = sorted(primary_data[secondary_var].unique())
            
            if not secondary_values:
                continue
                
            fig = go.Figure()
            
            # Use a pleasing color palette
            if colors is None:
                colors = [
                    'rgba(93, 164, 214, 0.7)', 'rgba(255, 144, 14, 0.7)', 
                    'rgba(44, 160, 101, 0.7)', 'rgba(255, 65, 54, 0.7)', 
                    'rgba(207, 114, 255, 0.7)', 'rgba(127, 96, 0, 0.7)',
                    'rgba(255, 140, 184, 0.7)', 'rgba(79, 90, 117, 0.7)'
                ]
            
            # Add boxplot for each secondary value within this primary value
            for i, secondary_val in enumerate(secondary_values):
                subset_data = primary_data[primary_data[secondary_var] == secondary_val][var_name_y].dropna()
                
                if not subset_data.empty:
                    count = len(subset_data)
                    median = subset_data.median()
                    mean = subset_data.mean()
                    
                    fig.add_trace(go.Box(
                        y=subset_data,
                        name=f"{secondary_val} (n={count})",
                        boxpoints='all',
                        pointpos=0,
                        jitter=0.5,
                        whiskerwidth=0.4,
                        marker=dict(size=5, opacity=0.7, color='rgba(0,0,0,0.7)'),
                        line=dict(width=1.5),
                        fillcolor=colors[i % len(colors)],
                        boxmean=True,
                        width=0.8,
                        hovertemplate=(
                            f"<b>{secondary_val}</b><br>" +
                            "Value: %{y:.3f}<br>" +
                            f"Median: {median:.3f}<br>" +
                            f"Mean: {mean:.3f}<br>" +
                            f"Count: {count}"
                        )
                    ))
            
            # Update layout
            title_text = f"{var_y} by {secondary_label} ({primary_label}: {primary_val})"
            subtitle = f"Data from {len(primary_data)} measurements (light only)"
            
            fig.update_layout(
                title=f"{title_text}<br><sup>{subtitle}</sup>",
                xaxis_title=f"{secondary_label}",
                yaxis_title=var_name_y,
                boxmode='group',
                boxgap=0.05,
                boxgroupgap=0.1,
                template="plotly_white",
                margin=dict(l=40, r=40, t=100, b=80),
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            # Rotate x-axis labels if many secondary values
            if len(secondary_values) > 4:
                fig.update_layout(xaxis=dict(tickangle=-10, tickfont=dict(size=10)))
            
            figures.append(fig)
            
            # Clean primary_val for filename (remove special characters)
            clean_primary_val = str(primary_val).replace(' ', '_').replace('/', '_').replace('&', 'and')
            figure_names.append(f"boxplot_{var_y}_by_{secondary_var}_{primary_var}_{clean_primary_val}.html")
        
        print(f"   Created {len(figures)} combination plots")
        return figures, figure_names