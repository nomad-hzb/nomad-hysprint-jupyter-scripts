import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ChemicalSolutionCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Chemical Solution Calculator")
        self.root.geometry("1000x800")

        # Load the raw data
        self.load_data()

        # Create the UI
        self.create_ui()

    def load_data(self):
        # In a real app, you would load from the Excel file
        # For this example, we'll use a simplified version of the data
        data = {
            'PbI2': {'Mol_Weight': 461.01, 'Density': 7.392},
            'PbBr2': {'Mol_Weight': 367.01, 'Density': 8.7},
            'CsI': {'Mol_Weight': 259.81, 'Density': 4.51},
            '2PACz': {'Mol_Weight': 275.24, 'Density': 1.0},
            'FAI': {'Mol_Weight': 171.97, 'Density': 2.604},
            'MABr': {'Mol_Weight': 111.97, 'Density': 2.167}
        }
        self.raw_data = data
        self.chemicals = list(data.keys())

    def create_ui(self):
        # Create a notebook (tabbed interface)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Create tabs
        tab1 = ttk.Frame(notebook)
        tab2 = ttk.Frame(notebook)
        notebook.add(tab1, text='Chemicals')
        notebook.add(tab2, text='Parameters')

        # Tab 1: Chemicals
        # Experiment number, density, x value
        frame1 = ttk.Frame(tab1)
        frame1.pack(fill='x', padx=5, pady=5)

        ttk.Label(frame1, text="Experiment #:").grid(row=0, column=0, padx=5, pady=5)
        self.experiment_number = ttk.Entry(frame1, width=10)
        self.experiment_number.insert(0, "1")
        self.experiment_number.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame1, text="Density (g/mL):").grid(row=0, column=2, padx=5, pady=5)
        self.density = ttk.Entry(frame1, width=10)
        self.density.insert(0, "0.978")
        self.density.grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(frame1, text="x value:").grid(row=0, column=4, padx=5, pady=5)
        self.x_value = ttk.Entry(frame1, width=10)
        self.x_value.insert(0, "4.0")
        self.x_value.grid(row=0, column=5, padx=5, pady=5)

        # Chemical selection
        frame2 = ttk.LabelFrame(tab1, text="Chemical Selection")
        frame2.pack(fill='x', padx=5, pady=5)

        ttk.Label(frame2, text="Chemical 1:").grid(row=0, column=0, padx=5, pady=5)
        self.chemical1 = ttk.Combobox(frame2, values=self.chemicals, width=15)
        self.chemical1.set("PbI2")
        self.chemical1.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame2, text="Chemical 2:").grid(row=0, column=2, padx=5, pady=5)
        self.chemical2 = ttk.Combobox(frame2, values=self.chemicals, width=15)
        self.chemical2.set("PbBr2")
        self.chemical2.grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(frame2, text="Chemical 3:").grid(row=0, column=4, padx=5, pady=5)
        self.chemical3 = ttk.Combobox(frame2, values=self.chemicals, width=15)
        self.chemical3.set("CsI")
        self.chemical3.grid(row=0, column=5, padx=5, pady=5)

        ttk.Label(frame2, text="Chemical 4:").grid(row=1, column=0, padx=5, pady=5)
        self.chemical4 = ttk.Combobox(frame2, values=self.chemicals, width=15)
        self.chemical4.set("2PACz")
        self.chemical4.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(frame2, text="Chemical 5:").grid(row=1, column=2, padx=5, pady=5)
        self.chemical5 = ttk.Combobox(frame2, values=self.chemicals, width=15)
        self.chemical5.set("FAI")
        self.chemical5.grid(row=1, column=3, padx=5, pady=5)

        ttk.Label(frame2, text="Chemical 6:").grid(row=1, column=4, padx=5, pady=5)
        self.chemical6 = ttk.Combobox(frame2, values=self.chemicals, width=15)
        self.chemical6.set("MABr")
        self.chemical6.grid(row=1, column=5, padx=5, pady=5)

        # Flask names
        frame3 = ttk.LabelFrame(tab1, text="Flask Names")
        frame3.pack(fill='x', padx=5, pady=5)

        ttk.Label(frame3, text="Flask 1:").grid(row=0, column=0, padx=5, pady=5)
        self.flask1 = ttk.Entry(frame3, width=15)
        self.flask1.insert(0, "PbI2_1")
        self.flask1.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame3, text="Flask 2:").grid(row=0, column=2, padx=5, pady=5)
        self.flask2 = ttk.Entry(frame3, width=15)
        self.flask2.insert(0, "PbBr2_1")
        self.flask2.grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(frame3, text="Flask 3:").grid(row=0, column=4, padx=5, pady=5)
        self.flask3 = ttk.Entry(frame3, width=15)
        self.flask3.insert(0, "CsI_1")
        self.flask3.grid(row=0, column=5, padx=5, pady=5)

        ttk.Label(frame3, text="Flask 4:").grid(row=1, column=0, padx=5, pady=5)
        self.flask4 = ttk.Entry(frame3, width=15)
        self.flask4.insert(0, "2PACz_1")
        self.flask4.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(frame3, text="Flask 5:").grid(row=1, column=2, padx=5, pady=5)
        self.flask5 = ttk.Entry(frame3, width=15)
        self.flask5.insert(0, "FAI_1")
        self.flask5.grid(row=1, column=3, padx=5, pady=5)

        ttk.Label(frame3, text="Flask 6:").grid(row=1, column=4, padx=5, pady=5)
        self.flask6 = ttk.Entry(frame3, width=15)
        self.flask6.insert(0, "MABr_1")
        self.flask6.grid(row=1, column=5, padx=5, pady=5)

        # Tab 2: Parameters
        # Concentrations
        frame4 = ttk.LabelFrame(tab2, text="Concentrations (mol/L)")
        frame4.pack(fill='x', padx=5, pady=5)

        ttk.Label(frame4, text="Conc 1:").grid(row=0, column=0, padx=5, pady=5)
        self.conc1 = ttk.Entry(frame4, width=10)
        self.conc1.insert(0, "1.5")
        self.conc1.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame4, text="Conc 2:").grid(row=0, column=2, padx=5, pady=5)
        self.conc2 = ttk.Entry(frame4, width=10)
        self.conc2.insert(0, "1.5")
        self.conc2.grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(frame4, text="Conc 3:").grid(row=0, column=4, padx=5, pady=5)
        self.conc3 = ttk.Entry(frame4, width=10)
        self.conc3.insert(0, "1.5")
        self.conc3.grid(row=0, column=5, padx=5, pady=5)

        ttk.Label(frame4, text="Conc 4:").grid(row=1, column=0, padx=5, pady=5)
        self.conc4 = ttk.Entry(frame4, width=10)
        self.conc4.insert(0, "0.0033")
        self.conc4.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(frame4, text="Conc 5:").grid(row=1, column=2, padx=5, pady=5)
        self.conc5 = ttk.Entry(frame4, width=10)
        self.conc5.insert(0, "1.24")
        self.conc5.grid(row=1, column=3, padx=5, pady=5)

        ttk.Label(frame4, text="Conc 6:").grid(row=1, column=4, padx=5, pady=5)
        self.conc6 = ttk.Entry(frame4, width=10)
        self.conc6.insert(0, "1.24")
        self.conc6.grid(row=1, column=5, padx=5, pady=5)

        # Molar ratios
        frame5 = ttk.LabelFrame(tab2, text="Molar Ratios")
        frame5.pack(fill='x', padx=5, pady=5)

        ttk.Label(frame5, text="Ratio 1:").grid(row=0, column=0, padx=5, pady=5)
        self.ratio1 = ttk.Entry(frame5, width=10)
        self.ratio1.insert(0, "83")
        self.ratio1.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame5, text="Ratio 2:").grid(row=0, column=2, padx=5, pady=5)
        self.ratio2 = ttk.Entry(frame5, width=10)
        self.ratio2.insert(0, "17")
        self.ratio2.grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(frame5, text="Ratio 3:").grid(row=0, column=4, padx=5, pady=5)
        self.ratio3 = ttk.Entry(frame5, width=10)
        self.ratio3.insert(0, "5")
        self.ratio3.grid(row=0, column=5, padx=5, pady=5)

        # Solvents
        frame6 = ttk.LabelFrame(tab2, text="Solvents")
        frame6.pack(fill='x', padx=5, pady=5)

        ttk.Label(frame6, text="Solvent 1:").grid(row=0, column=0, padx=5, pady=5)
        self.solvent1 = ttk.Entry(frame6, width=15)
        self.solvent1.insert(0, "4:1")
        self.solvent1.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame6, text="Solvent 2:").grid(row=0, column=2, padx=5, pady=5)
        self.solvent2 = ttk.Entry(frame6, width=15)
        self.solvent2.insert(0, "4:1")
        self.solvent2.grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(frame6, text="Solvent 3:").grid(row=0, column=4, padx=5, pady=5)
        self.solvent3 = ttk.Entry(frame6, width=15)
        self.solvent3.insert(0, "DMSO")
        self.solvent3.grid(row=0, column=5, padx=5, pady=5)

        ttk.Label(frame6, text="Solvent 4:").grid(row=1, column=0, padx=5, pady=5)
        self.solvent4 = ttk.Entry(frame6, width=15)
        self.solvent4.insert(0, "EtOH")
        self.solvent4.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(frame6, text="Solvent 5:").grid(row=1, column=2, padx=5, pady=5)
        self.solvent5 = ttk.Entry(frame6, width=15)
        self.solvent5.insert(0, "4:1 + PbI")
        self.solvent5.grid(row=1, column=3, padx=5, pady=5)

        ttk.Label(frame6, text="Solvent 6:").grid(row=1, column=4, padx=5, pady=5)
        self.solvent6 = ttk.Entry(frame6, width=15)
        self.solvent6.insert(0, "4:1 + PbBr")
        self.solvent6.grid(row=1, column=5, padx=5, pady=5)

        # Calculate button
        calculate_button = ttk.Button(self.root, text="Calculate", command=self.calculate)
        calculate_button.pack(pady=10)

        # Results frame
        self.results_frame = ttk.LabelFrame(self.root, text="Results")
        self.results_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Text widget for results
        self.results_text = tk.Text(self.results_frame, height=10, width=80)
        self.results_text.pack(side=tk.LEFT, fill='both', expand=True, padx=5, pady=5)

        # Scrollbar for text widget
        scrollbar = ttk.Scrollbar(self.results_frame, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill='y')
        self.results_text.config(yscrollcommand=scrollbar.set)

        # Frame for the chart
        self.chart_frame = ttk.Frame(self.root)
        self.chart_frame.pack(fill='both', expand=True, padx=10, pady=10)

    def calculate(self):
        try:
            # Get values from UI
            experiment_number = int(self.experiment_number.get())
            density = float(self.density.get())
            x_value = float(self.x_value.get())

            # Get chemical selections
            chemical1 = self.chemical1.get()
            chemical2 = self.chemical2.get()
            chemical3 = self.chemical3.get()
            chemical4 = self.chemical4.get()
            chemical5 = self.chemical5.get()
            chemical6 = self.chemical6.get()

            # Get molecular weights and densities
            mw1 = self.raw_data[chemical1]['Mol_Weight']
            mw2 = self.raw_data[chemical2]['Mol_Weight']
            mw3 = self.raw_data[chemical3]['Mol_Weight']
            mw4 = self.raw_data[chemical4]['Mol_Weight']
            mw5 = self.raw_data[chemical5]['Mol_Weight']
            mw6 = self.raw_data[chemical6]['Mol_Weight']

            d1 = self.raw_data[chemical1]['Density']
            d2 = self.raw_data[chemical2]['Density']
            d3 = self.raw_data[chemical3]['Density']
            d4 = self.raw_data[chemical4]['Density']
            d5 = self.raw_data[chemical5]['Density']
            d6 = self.raw_data[chemical6]['Density']

            # Get concentrations
            conc1 = float(self.conc1.get())
            conc2 = float(self.conc2.get())
            conc3 = float(self.conc3.get())
            conc4 = float(self.conc4.get())
            conc5 = float(self.conc5.get())
            conc6 = float(self.conc6.get())

            # Get ratios
            ratio1 = float(self.ratio1.get())
            ratio2 = float(self.ratio2.get())
            ratio3 = float(self.ratio3.get())

            # Calculate g/L
            gl1 = conc1 * mw1
            gl2 = conc2 * mw2
            gl3 = conc3 * mw3
            gl4 = conc4 * mw4
            gl5 = conc5 * mw5
            gl6 = conc6 * mw6

            # Calculate goal mL
            goal_ml1 = (ratio1 / 100) * x_value
            goal_ml2 = (ratio2 / 100) * x_value
            goal_ml3 = (ratio3 / 100) * x_value
            goal_ml4 = x_value
            goal_ml5 = goal_ml1
            goal_ml6 = goal_ml2

            # Calculate goal g
            goal_g1 = gl1 * goal_ml1 / 1000
            goal_g2 = gl2 * goal_ml2 / 1000
            goal_g3 = gl3 * goal_ml3 / 1000
            goal_g4 = gl4 * goal_ml4 / 1000
            goal_g5 = gl5 * goal_ml5 / 1000
            goal_g6 = gl6 * goal_ml6 / 1000

            # Calculate uL needed
            ul_needed1 = goal_ml1 * 1000
            ul_needed2 = goal_ml2 * 1000
            ul_needed3 = goal_ml3 * 1000

            # Calculate total volume in uL
            total_vol_ul = ul_needed1 + ul_needed2 + ul_needed3

            # Display results
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Experiment #{experiment_number}\n")
            self.results_text.insert(tk.END, f"Density: {density} g/mL\n")
            self.results_text.insert(tk.END, f"x value: {x_value}\n\n")
            self.results_text.insert(tk.END, f"Total Volume: {total_vol_ul:.1f} µL\n\n")

            self.results_text.insert(tk.END, "Results:\n")
            self.results_text.insert(tk.END, f"{chemical1} ({self.flask1.get()}):\n")
            self.results_text.insert(tk.END, f"  Mol. Weight: {mw1} g/mol\n")
            self.results_text.insert(tk.END, f"  Density: {d1} g/cm³\n")
            self.results_text.insert(tk.END, f"  Goal: {conc1} mol/L\n")
            self.results_text.insert(tk.END, f"  g/L: {gl1:.3f}\n")
            self.results_text.insert(tk.END, f"  Goal mL: {goal_ml1:.3f}\n")
            self.results_text.insert(tk.END, f"  Goal g: {goal_g1:.6f}\n\n")

            self.results_text.insert(tk.END, f"{chemical2} ({self.flask2.get()}):\n")
            self.results_text.insert(tk.END, f"  Mol. Weight: {mw2} g/mol\n")
            self.results_text.insert(tk.END, f"  Density: {d2} g/cm³\n")
            self.results_text.insert(tk.END, f"  Goal: {conc2} mol/L\n")
            self.results_text.insert(tk.END, f"  g/L: {gl2:.3f}\n")
            self.results_text.insert(tk.END, f"  Goal mL: {goal_ml2:.3f}\n")
            self.results_text.insert(tk.END, f"  Goal g: {goal_g2:.6f}\n\n")

            self.results_text.insert(tk.END, f"{chemical3} ({self.flask3.get()}):\n")
            self.results_text.insert(tk.END, f"  Mol. Weight: {mw3} g/mol\n")
            self.results_text.insert(tk.END, f"  Density: {d3} g/cm³\n")
            self.results_text.insert(tk.END, f"  Goal: {conc3} mol/L\n")
            self.results_text.insert(tk.END, f"  g/L: {gl3:.3f}\n")
            self.results_text.insert(tk.END, f"  Goal mL: {goal_ml3:.3f}\n")
            self.results_text.insert(tk.END, f"  Goal g: {goal_g3:.6f}\n\n")

            self.results_text.insert(tk.END, "Formulation Summary:\n")
            self.results_text.insert(tk.END, f"PbI2+FAI: {ul_needed1:.1f} µL\n")
            self.results_text.insert(tk.END, f"PbBr2+MABr: {ul_needed2:.1f} µL\n")
            self.results_text.insert(tk.END, f"CsI: {ul_needed3:.1f} µL\n")

            # Create pie chart
            for widget in self.chart_frame.winfo_children():
                widget.destroy()

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie([ul_needed1, ul_needed2, ul_needed3],
                   labels=[f'PbI2+FAI: {ul_needed1:.1f} µL',
                           f'PbBr2+MABr: {ul_needed2:.1f} µL',
                           f'CsI: {ul_needed3:.1f} µL'],
                   autopct='%1.1f%%')
            ax.set_title(f'Formulation Composition (Experiment #{experiment_number})')
            ax.axis('equal')

            canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)

        except Exception as e:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Error: {str(e)}\n")
            self.results_text.insert(tk.END, "Please check your inputs and try again.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChemicalSolutionCalculator(root)
    root.mainloop()
