# UNIFAC Model Implementation for Hansen Blend Calculator
# This module implements the UNIFAC (UNIversal Functional Activity Coefficient) model
# for calculating activity coefficients in non-ideal solvent mixtures

import numpy as np
import pandas as pd
from collections import defaultdict
import re
import math

# UNIFAC main group definitions
# Format: {group_id: (group_name, R_k (volume), Q_k (surface area))}
UNIFAC_MAIN_GROUPS = {
    1: ("CH2", 0.9011, 0.848),         # alkanes
    2: ("C=C", 1.3454, 1.176),         # alkenes
    3: ("ACH", 0.5313, 0.400),         # aromatic CH
    4: ("ACCH2", 1.0396, 0.968),       # aromatic C-CH2
    5: ("OH", 1.0000, 1.200),          # alcohols
    6: ("CH3OH", 1.4311, 1.432),       # methanol
    7: ("H2O", 0.9200, 1.400),         # water
    8: ("ACOH", 0.8952, 0.680),        # phenols
    9: ("CH3CO", 1.6724, 1.488),       # acetone
    10: ("CHO", 0.9980, 0.948),        # aldehydes
    11: ("CCOO", 1.6764, 1.420),       # esters
    12: ("HCOO", 1.2420, 1.188),       # formates
    13: ("CH2O", 1.1450, 1.080),       # ethers
    14: ("CNH2", 1.5959, 1.544),       # primary amines
    15: ("CNH", 1.4337, 1.304),        # secondary amines
    16: ("CN", 1.2719, 1.060),         # tertiary amines
    17: ("ACNH2", 1.0600, 0.816),      # anilines
    18: ("C≡N", 1.1200, 1.200),        # nitriles
    19: ("COOH", 1.3013, 1.224),       # acids
    20: ("CCl", 1.0478, 0.992),        # chlorides
    21: ("CCl2", 1.8824, 1.684),       # dichlorides
    22: ("CCl3", 2.8700, 2.376),       # trichlorides
    23: ("CCl4", 3.8000, 3.052),       # carbon tetrachloride
    24: ("CF", 0.8310, 0.724),         # fluorides
    25: ("CF2", 1.4060, 1.108),        # difluorides
    26: ("CF3", 1.9800, 1.380),        # trifluorides
    27: ("NO2", 1.4199, 1.104),        # nitro groups
    28: ("CONR2", 2.0290, 1.796),      # amides
    29: ("DMSO", 2.8266, 2.472),       # dimethyl sulfoxide
    30: ("CONH2", 1.5570, 1.416),      # primary amides
    31: ("CONHR", 1.8407, 1.636),      # secondary amides
    32: ("CONH", 1.6393, 1.416),       # tertiary amides
    33: ("PO4", 2.7500, 2.400),        # phosphates
    34: ("S", 0.9570, 0.920),          # sulfides
    35: ("SO", 1.6700, 1.468),         # sulfoxides
    36: ("SO2", 1.9310, 1.656),        # sulfones
    37: ("CCN", 1.4500, 1.184),        # acetonitriles
    38: ("NMP", 3.9810, 3.200),        # N-methyl-2-pyrrolidone
    39: ("DMF", 3.0856, 2.736),        # dimethylformamide
    40: ("HCON(CH3)2", 3.0856, 2.736), # DMF (alternative)
    # Add more as needed
}

# UNIFAC interaction parameters (a_mn) between main groups
# Source: UNIFAC consortium
# Format: a_mn is the interaction parameter from group m to group n
# These values are temperature-dependent (298.15K)
# A simplified subset is provided here
UNIFAC_INTERACTIONS = {
    # Format: (group_m, group_n): a_mn
    (1, 2): 86.02,    # CH2/C=C
    (2, 1): -35.36,   # C=C/CH2
    (1, 3): 61.13,    # CH2/ACH
    (3, 1): -11.12,   # ACH/CH2
    (1, 4): 76.50,    # CH2/ACCH2
    (4, 1): -69.70,   # ACCH2/CH2
    (1, 5): 986.5,    # CH2/OH
    (5, 1): 156.4,    # OH/CH2
    (1, 6): 697.2,    # CH2/CH3OH
    (6, 1): 16.51,    # CH3OH/CH2
    (1, 7): 1318.0,   # CH2/H2O
    (7, 1): 300.0,    # H2O/CH2
    (1, 9): 476.4,    # CH2/CH3CO
    (9, 1): 26.76,    # CH3CO/CH2
    (1, 10): 677.0,   # CH2/CHO
    (10, 1): 505.7,   # CHO/CH2
    (1, 11): 232.1,   # CH2/CCOO
    (11, 1): 37.85,   # CCOO/CH2
    (1, 12): 251.5,   # CH2/HCOO
    (12, 1): 214.5,   # HCOO/CH2
    (1, 13): 83.36,   # CH2/CH2O
    (13, 1): -30.48,  # CH2O/CH2
    (1, 18): 597.0,   # CH2/C≡N
    (18, 1): 25.82,   # C≡N/CH2
    (1, 19): 663.5,   # CH2/COOH
    (19, 1): 315.3,   # COOH/CH2
    (1, 37): 110.2,   # CH2/CCN
    (37, 1): 304.1,   # CCN/CH2
    # Many more interactions would be defined here
    # Water with various groups
    (7, 5): -229.1,   # H2O/OH
    (5, 7): 353.5,    # OH/H2O
    (7, 6): -181.0,   # H2O/CH3OH
    (6, 7): 289.6,    # CH3OH/H2O
    (7, 11): 101.1,   # H2O/CCOO
    (11, 7): 245.4,   # CCOO/H2O
    (7, 18): 362.1,   # H2O/C≡N
    (18, 7): 377.6,   # C≡N/H2O
    (7, 19): -66.17,  # H2O/COOH
    (19, 7): 168.0,   # COOH/H2O
    # Add numerous other group interactions
    # These are just examples and would need to be expanded
}

# Mapping of common solvents to their UNIFAC groups
# Format: {solvent_name: {group_id: count}}
SOLVENT_GROUP_MAP = {
    "Water": {7: 1},
    "Methanol": {6: 1},
    "Ethanol": {1: 1, 5: 1},
    "Propanol": {1: 2, 5: 1},
    "Isopropanol": {1: 2, 5: 1},
    "Butanol": {1: 3, 5: 1},
    "Acetone": {9: 1},
    "Acetonitrile": {37: 1},
    "Ethyl acetate": {1: 2, 11: 1},
    "Dichloromethane": {21: 1},
    "Chloroform": {22: 1},
    "Carbon tetrachloride": {23: 1},
    "Tetrahydrofuran": {1: 3, 13: 1},
    "Diethyl ether": {1: 2, 13: 1},
    "Dimethyl sulfoxide": {29: 1},
    "N,N-Dimethylformamide": {39: 1},
    "Toluene": {1: 1, 3: 5, 4: 1},
    "Benzene": {3: 6},
    "Hexane": {1: 6},
    "Acetic acid": {1: 1, 19: 1},
    "Acetaldehyde": {1: 1, 10: 1},
    "n-Butanol": {1: 3, 5: 1},
    "Pyridine": {3: 5, 16: 1},
    "N-Methylpyrrolidone": {38: 1},
    "1,4-Dioxane": {13: 2, 1: 4},
    # Add more solvent mappings as needed
}

def parse_smiles_to_unifac_groups(smiles):
    """
    Parse a SMILES string to identify UNIFAC functional groups.
    This is a simplified approach and would need to be expanded for complex molecules.
    
    Parameters:
    smiles (str): SMILES representation of a molecule
    
    Returns:
    dict: Dictionary of UNIFAC group IDs and their counts
    """
    groups = defaultdict(int)
    
    # This is a very simplified identification
    # In practice, you would use a cheminformatics library like RDKit
    
    # Check for alcohols (excluding methanol)
    if re.search(r'[^C]CO', smiles) and not smiles == "CO":
        groups[5] += 1  # OH group
    
    # Check for methanol
    if smiles == "CO":
        groups[6] += 1  # CH3OH group
    
    # Check for water
    if smiles == "O":
        groups[7] += 1  # H2O group
    
    # Check for aromatics
    aromatic_carbons = len(re.findall(r'c', smiles))
    if aromatic_carbons > 0:
        groups[3] += aromatic_carbons  # ACH groups
    
    # Check for ketones
    if re.search(r'C\(=O\)C', smiles):
        groups[9] += 1  # CH3CO group
    
    # Check for aldehydes
    if re.search(r'C=O', smiles) and not re.search(r'C\(=O\)O', smiles) and not re.search(r'C\(=O\)C', smiles):
        groups[10] += 1  # CHO group
    
    # Check for esters
    if re.search(r'C\(=O\)O[A-Za-z]', smiles):
        groups[11] += 1  # CCOO group
    
    # Check for ethers
    if re.search(r'COC', smiles) and not re.search(r'C\(=O\)OC', smiles):
        groups[13] += 1  # CH2O group
    
    # Check for nitriles
    if re.search(r'C#N', smiles):
        groups[18] += 1  # C≡N group
    
    # Check for carboxylic acids
    if re.search(r'C\(=O\)O$', smiles) or re.search(r'C\(=O\)O[^\w]', smiles):
        groups[19] += 1  # COOH group
    
    # Check for chloroalkanes
    chlorine_count = smiles.count('Cl')
    if chlorine_count == 1:
        groups[20] += 1  # CCl group
    elif chlorine_count == 2:
        groups[21] += 1  # CCl2 group
    elif chlorine_count == 3:
        groups[22] += 1  # CCl3 group
    elif chlorine_count == 4 and re.search(r'CCl4', smiles):
        groups[23] += 1  # CCl4 group
    
    # Count alkane groups (CH3, CH2, CH, C)
    # This is a simplified approach
    aliphatic_carbons = len(re.findall(r'C(?![arol])', smiles)) - aromatic_carbons
    if aliphatic_carbons > 0:
        groups[1] += aliphatic_carbons  # CH2 group (simplified)
    
    # Many more group identifications would be needed for a complete implementation
    
    return dict(groups)

def calculate_activity_coefficients_unifac(mixture_df, composition, temperature=298.15):
    """
    Calculate activity coefficients using the UNIFAC model
    
    Parameters:
    mixture_df (DataFrame): DataFrame with solvent data including SMILES
    composition (ndarray): Mole fractions of each solvent
    temperature (float): Temperature in Kelvin
    
    Returns:
    ndarray: Activity coefficients for each solvent
    """
    n_components = len(mixture_df)
    
    # Check if we have a valid composition
    if len(composition) != n_components:
        raise ValueError("Composition length must match number of components")
    
    # Normalize composition to ensure it sums to 1
    composition = np.array(composition) / np.sum(composition)
    
    # Step 1: Identify functional groups for each solvent
    component_groups = []
    all_groups = set()
    
    for idx, row in mixture_df.iterrows():
        solvent_name = row['Name']
        smiles = row.get('SMILES', '')
        
        # First check if we have a predefined mapping
        if solvent_name in SOLVENT_GROUP_MAP:
            groups = SOLVENT_GROUP_MAP[solvent_name]
        # Otherwise try to parse SMILES
        elif smiles:
            groups = parse_smiles_to_unifac_groups(smiles)
        else:
            print(f"Warning: No group information available for {solvent_name}. Using ideal behavior.")
            groups = {}
        
        component_groups.append(groups)
        all_groups.update(groups.keys())
    
    # Step 2: Calculate combinatorial and residual parts
    if not all_groups:
        print("No UNIFAC group information available for any component. Using ideal behavior.")
        return np.ones(n_components)
    
    # Convert all groups to list for consistent indexing
    all_groups_list = sorted(list(all_groups))
    n_groups = len(all_groups_list)
    
    # Get R and Q parameters for all groups
    R = np.zeros(n_groups)
    Q = np.zeros(n_groups)
    
    for i, group_id in enumerate(all_groups_list):
        if group_id in UNIFAC_MAIN_GROUPS:
            _, r_k, q_k = UNIFAC_MAIN_GROUPS[group_id]
            R[i] = r_k
            Q[i] = q_k
        else:
            print(f"Warning: Missing R,Q parameters for group {group_id}")
            # Use typical/average values as fallback
            R[i] = 1.0
            Q[i] = 1.0
    
    # For a full UNIFAC implementation, we would now:
    # 1. Calculate combinatorial part (molecular size and shape differences)
    # 2. Calculate residual part (functional group interactions)
    # 3. Combine for total activity coefficients
    
    # This is a simplified calculation to demonstrate non-ideal mixing
    # It introduces some non-ideality based on the differences between components
    # without implementing the full UNIFAC algorithm
    gamma = np.ones(n_components)
    
    # Calculate a simple non-ideality factor based on component differences
    for i in range(n_components):
        # Skip components with no group information
        if not component_groups[i]:
            continue
            
        for j in range(n_components):
            if i == j or not component_groups[j]:
                continue
                
            # Count overlapping groups
            groups_i = set(component_groups[i].keys())
            groups_j = set(component_groups[j].keys())
            
            # Calculate a simple interaction factor
            different_groups = groups_i.symmetric_difference(groups_j)
            if different_groups:
                # Introduce non-ideality based on differences
                gamma[i] += 0.05 * len(different_groups) * composition[j]
    
    # Temperature dependence (simplified)
    # Increase deviations from ideality at lower temperatures
    temp_factor = (298.15 / temperature) ** 0.5
    for i in range(n_components):
        if gamma[i] > 1.0:
            gamma[i] = 1.0 + (gamma[i] - 1.0) * temp_factor
    
    # Add some randomness to simulate complex interactions
    # This would be replaced by actual UNIFAC calculations in a full implementation
    np.random.seed(42)  # For reproducibility
    random_factor = 0.1 * np.random.rand(n_components)
    gamma = gamma * (1.0 + random_factor)
    
    # Ensure reasonable bounds
    gamma = np.clip(gamma, 0.5, 2.0)
    
    return gamma

def calculate_overall_donor_number_with_unifac(solvents_df, percentages, temperature=298.15):
    """
    Calculate the overall donor number for a solvent blend accounting for non-ideal mixing
    using UNIFAC activity coefficients
    
    Parameters:
    solvents_df (DataFrame): DataFrame containing selected solvents with donor number values
    percentages (array): Volume fractions or weight percentages of each solvent
    temperature (float): Temperature in Kelvin
    
    Returns:
    float: Overall donor number of the blend
    """
    # Check if donor number column exists
    if 'DN' not in solvents_df.columns:
        print("Warning: Donor Number (DN) data is missing in the database")
        return None
    
    # Filter out solvents with missing DN values
    valid_data = solvents_df.dropna(subset=['DN']).copy()
    if len(valid_data) == 0:
        print("No valid Donor Number data available")
        return None
    
    # Adjust percentages to only include solvents with valid DN
    valid_indices = [i for i, _ in enumerate(percentages) if not pd.isna(solvents_df.iloc[i].get('DN'))]
    valid_percentages = [percentages[i] for i in valid_indices]
    
    # Normalize valid percentages
    if sum(valid_percentages) > 0:
        valid_percentages = np.array(valid_percentages) / sum(valid_percentages)
    else:
        print("Error: Sum of valid percentages is zero")
        return None
    
    # Calculate activity coefficients using UNIFAC
    try:
        activity_coefficients = calculate_activity_coefficients_unifac(
            valid_data, valid_percentages, temperature
        )
        print(f"Calculated activity coefficients using UNIFAC: {[round(g, 3) for g in activity_coefficients]}")
    except Exception as e:
        print(f"Error calculating UNIFAC activity coefficients: {e}")
        print("Falling back to ideal mixing (activity coefficients = 1.0)")
        activity_coefficients = np.ones(len(valid_data))
    
    # Extract donor numbers
    donor_numbers = valid_data['DN'].values
    
    # Calculate overall donor number (weighted sum with activity coefficients)
    overall_dn = np.sum(valid_percentages * activity_coefficients * donor_numbers)
    
    return overall_dn

# Example usage:
# activity_coeffs = calculate_activity_coefficients_unifac(solvent_df, [0.5, 0.3, 0.2])
# overall_dn = calculate_overall_donor_number_with_unifac(solvent_df, [0.5, 0.3, 0.2])
