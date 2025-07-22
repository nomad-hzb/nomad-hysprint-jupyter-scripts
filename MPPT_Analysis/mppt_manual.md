# MPPT Analysis Tool - User Manual

## Overview
This interactive tool provides a complete workflow for analyzing Maximum Power Point Tracking (MPPT) data, from batch selection to comprehensive result export.

## 1. Batch Selection
**Filter and Load Data**
- Click "Filter MPPT Batches" to automatically identify batches containing MPPT measurements
- Use the search field to find specific batches by name
- Select one or more batches from the filtered list
- Click "Load Data" to import MPPT measurements

The system will automatically process the data by:
- Inverting power density and current density values (multiply by -1)
- Converting time units from seconds to hours
- Loading sample descriptions and metadata

## 2. Sample Selection
**Configure Sample Names and Selection**
- Review all samples found with MPPT data
- Choose naming convention using the dropdown:
  * **Sample Name**: Use lab ID (everything after "&" if present)
  * **Batch**: Use batch name (removes "_suffix" if present) 
  * **Sample Description**: Use description from NOMAD entry
  * **Custom**: Enter your own names manually
- Select/deselect samples using checkboxes
- Click "Confirm Selection" to proceed

**Note**: Sample names will be used in plots and exported files. Curves with identical names will be grouped in statistical plots.

## 3. Curve Fitting
**Model Selection and Configuration**
- Choose from available mathematical models:
  * Linear: P(t) = at + b
  * Exponential: P(t) = A·exp(-t/τ)
  * Biexponential: Two exponential components
  * Logistic + Exponential: Combined decay model
  * Stretched Exponential: Power-law decay
  * Error Function × Linear: Sigmoid-based model

- **Time Range Slider**: Restrict fitting to specific time intervals (useful for excluding burn-in periods)
- Click "Fit All Curves" to apply the selected model to all power density curves
- Review detailed fitting results and statistical summaries

**Note**: Only power density curves are fitted; voltage and current data are used for visualization only.

## 4. Plotting and Visualization
**Generate Interactive Plots**
- **Variable Selection**: Choose between Power Density, Voltage, or Current Density
- **Plot Styles**:
  * Individual: Separate plot for each curve
  * All Together: Combined view of all curves
  * By Sample: Group curves from the same sample
  * Area (Quartiles): Statistical envelope showing median and 25th-75th percentiles
  * Area (Std Dev): Statistical envelope showing mean ± standard deviation
- **Show Fitting Lines**: Overlay mathematical model fits (available for Power Density only)
- **Parameter Histograms**: Distribution analysis of fitted parameters (t80, T80, tS, etc.)

## 5. Download Results
**Export Comprehensive Analysis Package**
- **Excel File**: Multi-sheet workbook containing:
  * Raw Curve Data: Original measurements with sample IDs as column headers
  * Fitted Curve Data: Mathematical model outputs
  * Fit Results: All fitting parameters and statistics
  * Statistical Summary: Descriptive statistics for all parameters
  * Sample Information: Metadata and custom names

- **Plot Formats**:
  * HTML: Interactive plots viewable in web browsers
  * PNG: Static images for presentations/documents
  * Both: Complete set in both formats

- **Additional Options**:
  * Include/exclude raw measurement data
  * Include/exclude fitted curve data
  * Automatic README file with analysis details

The download package includes organized folders, comprehensive documentation, and all data needed to reproduce or extend the analysis.

## Tips and Best Practices
- Use "Filter MPPT Batches" to avoid manually searching through empty batches
- Experiment with different time ranges to exclude noisy initial periods
- For statistical plots (area styles), ensure multiple curves per sample exist
- HTML plots are recommended for interactive exploration; PNG for final presentations
- The Excel file uses wide format with sample IDs as column headers for easy comparison
- All time values are in hours; power/current values are sign-corrected from raw data

## Troubleshooting
- If plots don't generate: Ensure curve fitting is completed first
- If PNG export fails: HTML format is always available as fallback
- For large datasets: Consider selecting fewer samples or time ranges to improve performance
- If Excel file won't open: Check that download completed successfully (file size > 0 MB)