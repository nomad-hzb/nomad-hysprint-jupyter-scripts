"""
Data Manager for the Design of Experiments application.
Handles variable management, data validation, and file I/O operations.
"""

import pandas as pd
import numpy as np
import json
import csv
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import io
import base64

from utils import ValidationUtils, Constants


class VariableType(Enum):
    """Enumeration of supported variable types."""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete" 
    CATEGORICAL = "categorical"


@dataclass
class Variable:
    """Data class representing a single experimental variable."""
    name: str
    type: VariableType
    description: str = ""
    
    # For continuous and discrete variables
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step_size: Optional[float] = None  # For discrete only
    
    # For categorical variables
    categories: Optional[List[str]] = None
    
    def validate(self) -> Tuple[bool, str]:
        """Validate the variable definition."""
        if not self.name or not self.name.strip():
            return False, "Variable name cannot be empty"
        
        if self.type in [VariableType.CONTINUOUS, VariableType.DISCRETE]:
            if self.min_value is None or self.max_value is None:
                return False, f"Min and max values required for {self.type.value} variables"
            
            if self.min_value >= self.max_value:
                return False, "Max value must be greater than min value"
            
            if self.type == VariableType.DISCRETE:
                if self.step_size is None or self.step_size <= 0:
                    return False, "Positive step size required for discrete variables"
        
        elif self.type == VariableType.CATEGORICAL:
            if not self.categories or len(self.categories) < 2:
                return False, "At least 2 categories required for categorical variables"
            
            if len(set(self.categories)) != len(self.categories):
                return False, "Duplicate categories not allowed"
        
        return True, ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert variable to dictionary representation."""
        return {
            'name': self.name,
            'type': self.type.value,
            'description': self.description,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'step_size': self.step_size,
            'categories': self.categories
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Variable':
        """Create Variable from dictionary representation."""
        return cls(
            name=data['name'],
            type=VariableType(data['type']),
            description=data.get('description', ''),
            min_value=data.get('min_value'),
            max_value=data.get('max_value'),
            step_size=data.get('step_size'),
            categories=data.get('categories')
        )
    
    def get_range(self) -> Union[Tuple[float, float], List[str]]:
        """Get the range/domain of the variable."""
        if self.type == VariableType.CATEGORICAL:
            return self.categories
        else:
            return (self.min_value, self.max_value)
    
    def get_discrete_values(self) -> List[Union[float, str]]:
        """Get all possible discrete values for the variable."""
        if self.type == VariableType.CATEGORICAL:
            return self.categories
        elif self.type == VariableType.DISCRETE:
            n_steps = int((self.max_value - self.min_value) / self.step_size) + 1
            return [self.min_value + i * self.step_size for i in range(n_steps)]
        else:
            # For continuous variables, return a reasonable sample
            return list(np.linspace(self.min_value, self.max_value, 10))


class DataManager:
    """Manages experimental variables and data validation."""
    
    def __init__(self):
        """Initialize the data manager."""
        self.variables: List[Variable] = []
        self.validator = ValidationUtils()
        self._variable_cache = {}
        
    def add_variable(self, variable: Variable) -> Tuple[bool, str]:
        """Add a new variable with validation."""
        # Validate variable
        is_valid, error_msg = variable.validate()
        if not is_valid:
            return False, error_msg
        
        # Check for duplicate names
        if any(v.name == variable.name for v in self.variables):
            return False, f"Variable '{variable.name}' already exists"
        
        # Add variable
        self.variables.append(variable)
        self._clear_cache()
        return True, f"Variable '{variable.name}' added successfully"
    
    def remove_variable(self, variable_name: str) -> Tuple[bool, str]:
        """Remove a variable by name."""
        original_count = len(self.variables)
        self.variables = [v for v in self.variables if v.name != variable_name]
        
        if len(self.variables) < original_count:
            self._clear_cache()
            return True, f"Variable '{variable_name}' removed successfully"
        else:
            return False, f"Variable '{variable_name}' not found"
    
    def update_variable(self, variable_name: str, updated_variable: Variable) -> Tuple[bool, str]:
        """Update an existing variable."""
        # Validate updated variable
        is_valid, error_msg = updated_variable.validate()
        if not is_valid:
            return False, error_msg
        
        # Find and update variable
        for i, var in enumerate(self.variables):
            if var.name == variable_name:
                # Check if new name conflicts with other variables
                if (updated_variable.name != variable_name and 
                    any(v.name == updated_variable.name for v in self.variables)):
                    return False, f"Variable '{updated_variable.name}' already exists"
                
                self.variables[i] = updated_variable
                self._clear_cache()
                return True, f"Variable '{variable_name}' updated successfully"
        
        return False, f"Variable '{variable_name}' not found"
    
    def get_variables(self) -> List[Variable]:
        """Get all variables."""
        return self.variables.copy()
    
    def get_variable(self, name: str) -> Optional[Variable]:
        """Get a specific variable by name."""
        for var in self.variables:
            if var.name == name:
                return var
        return None
    
    def get_variable_names(self) -> List[str]:
        """Get list of all variable names."""
        return [v.name for v in self.variables]
    
    def get_variables_by_type(self, var_type: VariableType) -> List[Variable]:
        """Get all variables of a specific type."""
        return [v for v in self.variables if v.type == var_type]
    
    def set_variables(self, variables: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """Set variables from list of dictionaries."""
        try:
            new_variables = []
            for var_dict in variables:
                var = Variable.from_dict(var_dict)
                is_valid, error_msg = var.validate()
                if not is_valid:
                    return False, f"Invalid variable '{var.name}': {error_msg}"
                new_variables.append(var)
            
            # Check for duplicate names
            names = [v.name for v in new_variables]
            if len(set(names)) != len(names):
                return False, "Duplicate variable names found"
            
            self.variables = new_variables
            self._clear_cache()
            return True, f"Successfully loaded {len(new_variables)} variables"
            
        except Exception as e:
            return False, f"Error loading variables: {str(e)}"
    
    def validate_sample_data(self, data: pd.DataFrame) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate sample data against variable definitions."""
        if data.empty:
            return False, "Sample data is empty", {}
        
        # Check if all variables are present as columns
        var_names = self.get_variable_names()
        missing_vars = set(var_names) - set(data.columns)
        extra_vars = set(data.columns) - set(var_names)
        
        validation_info = {
            'missing_variables': list(missing_vars),
            'extra_variables': list(extra_vars),
            'total_samples': len(data),
            'variable_stats': {}
        }
        
        if missing_vars:
            return False, f"Missing variables in data: {', '.join(missing_vars)}", validation_info
        
        # Validate each variable's data
        validation_errors = []
        
        for var in self.variables:
            if var.name not in data.columns:
                continue
                
            col_data = data[var.name]
            var_stats = {
                'null_count': col_data.isnull().sum(),
                'unique_count': col_data.nunique(),
                'data_type': str(col_data.dtype)
            }
            
            if var.type == VariableType.CONTINUOUS:
                # Check for numeric data and range
                if not pd.api.types.is_numeric_dtype(col_data):
                    validation_errors.append(f"Variable '{var.name}' should be numeric")
                else:
                    out_of_range = ((col_data < var.min_value) | (col_data > var.max_value)).sum()
                    if out_of_range > 0:
                        validation_errors.append(
                            f"Variable '{var.name}' has {out_of_range} values out of range "
                            f"[{var.min_value}, {var.max_value}]"
                        )
                    var_stats.update({
                        'min_value': float(col_data.min()),
                        'max_value': float(col_data.max()),
                        'mean_value': float(col_data.mean()),
                        'out_of_range_count': out_of_range
                    })
            
            elif var.type == VariableType.DISCRETE:
                # Check for numeric data, range, and step compliance
                if not pd.api.types.is_numeric_dtype(col_data):
                    validation_errors.append(f"Variable '{var.name}' should be numeric")
                else:
                    out_of_range = ((col_data < var.min_value) | (col_data > var.max_value)).sum()
                    if out_of_range > 0:
                        validation_errors.append(
                            f"Variable '{var.name}' has {out_of_range} values out of range "
                            f"[{var.min_value}, {var.max_value}]"
                        )
                    
                    # Check step compliance
                    expected_values = set(var.get_discrete_values())
                    actual_values = set(col_data.dropna().unique())
                    invalid_values = actual_values - expected_values
                    
                    if invalid_values:
                        validation_errors.append(
                            f"Variable '{var.name}' has invalid discrete values: {invalid_values}"
                        )
                    
                    var_stats.update({
                        'min_value': float(col_data.min()),
                        'max_value': float(col_data.max()),
                        'out_of_range_count': out_of_range,
                        'invalid_discrete_count': len(invalid_values)
                    })
            
            elif var.type == VariableType.CATEGORICAL:
                # Check for valid categories
                invalid_categories = set(col_data.dropna().unique()) - set(var.categories)
                if invalid_categories:
                    validation_errors.append(
                        f"Variable '{var.name}' has invalid categories: {invalid_categories}"
                    )
                
                var_stats.update({
                    'categories_present': list(col_data.value_counts().index),
                    'category_counts': col_data.value_counts().to_dict(),
                    'invalid_category_count': len(invalid_categories)
                })
            
            validation_info['variable_stats'][var.name] = var_stats
        
        if validation_errors:
            return False, '; '.join(validation_errors), validation_info
        
        return True, "Sample data validation successful", validation_info
    
    def parse_text_variables(self, text_input: str) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """Parse variables from text input format."""
        try:
            variables = []
            lines = [line.strip() for line in text_input.strip().split('\n') if line.strip()]
            
            for line_num, line in enumerate(lines, 1):
                if line.startswith('#') or not line:  # Skip comments and empty lines
                    continue
                
                parts = [part.strip() for part in line.split(',')]
                
                if len(parts) < 3:
                    return False, f"Line {line_num}: Invalid format. Expected: name,type,params...", []
                
                name = parts[0]
                var_type = parts[1].lower()
                
                if var_type == 'continuous':
                    if len(parts) != 4:
                        return False, f"Line {line_num}: Continuous variables need: name,continuous,min,max", []
                    
                    try:
                        min_val = float(parts[2])
                        max_val = float(parts[3])
                    except ValueError:
                        return False, f"Line {line_num}: Min and max values must be numeric", []
                    
                    variables.append({
                        'name': name,
                        'type': 'continuous',
                        'min_value': min_val,
                        'max_value': max_val
                    })
                
                elif var_type == 'discrete':
                    if len(parts) < 4 or len(parts) > 5:
                        return False, f"Line {line_num}: Discrete variables need: name,discrete,min,max[,step]", []
                    
                    try:
                        min_val = float(parts[2])
                        max_val = float(parts[3])
                        step = float(parts[4]) if len(parts) == 5 else 1.0
                    except ValueError:
                        return False, f"Line {line_num}: Min, max, and step values must be numeric", []
                    
                    variables.append({
                        'name': name,
                        'type': 'discrete',
                        'min_value': min_val,
                        'max_value': max_val,
                        'step_size': step
                    })
                
                elif var_type == 'categorical':
                    if len(parts) < 4:
                        return False, f"Line {line_num}: Categorical variables need at least 2 categories", []
                    
                    categories = parts[2:]
                    variables.append({
                        'name': name,
                        'type': 'categorical',
                        'categories': categories
                    })
                
                else:
                    return False, f"Line {line_num}: Unknown variable type '{var_type}'. Use: continuous, discrete, or categorical", []
            
            if not variables:
                return False, "No valid variables found in input", []
            
            return True, f"Successfully parsed {len(variables)} variables", variables
            
        except Exception as e:
            return False, f"Error parsing text input: {str(e)}", []
    
    def _clear_cache(self):
        """Clear internal caches when variables change."""
        self._variable_cache.clear()
    
    def clear_all_variables(self):
        """Remove all variables."""
        self.variables.clear()
        self._clear_cache()
    
    def has_variables(self) -> bool:
        """Check if any variables are defined."""
        return len(self.variables) > 0