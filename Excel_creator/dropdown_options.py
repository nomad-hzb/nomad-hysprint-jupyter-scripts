"""
Dropdown options for Excel data validation
Each key is a field identifier, value is a list of valid options
"""

DROPDOWN_OPTIONS = {
    # Cleaning & Filtering
    'filter_material': ['Paper', 'Metal', 'Glass', 'PTFE', 'Nylon'],
    'solvent_name': ['DMF', 'DMSO', 'Chlorobenzene', 'Toluene', 'Acetone', 'IPA', 'Ethanol'],
    'gas_type': ['Nitrogen', 'Oxygen', 'Argon', 'Air', 'Forming Gas'],
    
    # Materials
    'layer_type': ['Active Layer', 'Electron Transport Layer', 'Hole Transport Layer', 
                   'Buffer Layer', 'Contact Layer', 'Additive', 'Electrode'],
    'substrate_material': ['Soda Lime Glass', 'FTO Glass', 'ITO Glass', 'Flexible Polymer'],
    'substrate_conductive': ['ITO', 'FTO', 'None'],
    
    # Common fields
    'operator': ['MaxMustermann', 'JohnDoe', 'JaneSmith'],
    'annealing_atmosphere': ['Nitrogen', 'Air', 'Vacuum', 'Oxygen', 'Argon'],
    
    # You can add more as needed
}

def get_dropdown_options(field_key):
    """
    Get dropdown options for a specific field
    Returns None if no dropdown is defined for this field
    """
    return DROPDOWN_OPTIONS.get(field_key)