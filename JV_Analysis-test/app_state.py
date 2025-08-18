"""
Application State Management Module
Centralized state management for the JV Analysis Dashboard.
Handles data storage, UI state, and state change notifications.
"""

__author__ = "Edgar Nandayapa"
__institution__ = "Helmholtz-Zentrum Berlin"
__created__ = "August 2025"

class AppState:
    def __init__(self):
        # Data storage
        self._data = {}
        self._unique_vals = []
        self._global_plot_data = {'figs': [], 'names': [], 'workbook': None}
        self._filter_vals = None
        
        # Application flags
        self._is_conditions = False
        
        # UI state
        self._current_tab = 0
        self._tabs_enabled = [True, False, False, False, False]  # Only first tab enabled initially
        
        # Callbacks for state changes
        self._state_change_callbacks = []
    
    # Data properties
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        self._data = value
        self._notify_state_change('data', value)
    
    @property
    def unique_vals(self):
        return self._unique_vals
    
    @unique_vals.setter
    def unique_vals(self, value):
        self._unique_vals = value
        self._notify_state_change('unique_vals', value)
    
    @property
    def global_plot_data(self):
        return self._global_plot_data
    
    @global_plot_data.setter
    def global_plot_data(self, value):
        self._global_plot_data = value
        self._notify_state_change('global_plot_data', value)
    
    @property
    def filter_vals(self):
        return self._filter_vals
    
    @filter_vals.setter
    def filter_vals(self, value):
        self._filter_vals = value
        self._notify_state_change('filter_vals', value)
    
    # Application flags
    @property
    def is_conditions(self):
        return self._is_conditions
    
    @is_conditions.setter
    def is_conditions(self, value):
        self._is_conditions = value
        self._notify_state_change('is_conditions', value)
    
    # UI state management
    @property
    def current_tab(self):
        return self._current_tab
    
    @current_tab.setter
    def current_tab(self, value):
        if 0 <= value < len(self._tabs_enabled):
            self._current_tab = value
            self._notify_state_change('current_tab', value)
    
    def is_tab_enabled(self, tab_index):
        """Check if a tab is enabled"""
        if 0 <= tab_index < len(self._tabs_enabled):
            return self._tabs_enabled[tab_index]
        return False
    
    def enable_tab(self, tab_index):
        """Enable a specific tab"""
        if 0 <= tab_index < len(self._tabs_enabled):
            self._tabs_enabled[tab_index] = True
            self._notify_state_change('tab_enabled', tab_index)
    
    def disable_tab(self, tab_index):
        """Disable a specific tab"""
        if 0 <= tab_index < len(self._tabs_enabled):
            self._tabs_enabled[tab_index] = False
            self._notify_state_change('tab_disabled', tab_index)
    
    # Data utility methods
    def clear_data(self):
        """Clear all loaded data"""
        self._data = {}
        self._unique_vals = []
        self._filter_vals = None
        self._notify_state_change('data_cleared', None)
    
    def has_data(self, data_type=None):
        """Check if data exists"""
        if data_type:
            return data_type in self._data and not self._data[data_type].empty
        return bool(self._data)
    
    def get_data_summary(self):
        """Get summary of loaded data"""
        summary = {}
        for key, df in self._data.items():
            if hasattr(df, 'shape'):
                summary[key] = {'rows': df.shape[0], 'columns': df.shape[1]}
            else:
                summary[key] = {'type': type(df).__name__}
        return summary
    
    def update_plot_data(self, figs, names, workbook):
        """Update plot data in one call"""
        self._global_plot_data = {'figs': figs, 'names': names, 'workbook': workbook}
        self._notify_state_change('plot_data_updated', self._global_plot_data)
    
    # State change notification system
    def add_state_change_callback(self, callback):
        """Add a callback for state changes"""
        self._state_change_callbacks.append(callback)
    
    def remove_state_change_callback(self, callback):
        """Remove a callback"""
        if callback in self._state_change_callbacks:
            self._state_change_callbacks.remove(callback)
    
    def _notify_state_change(self, change_type, value):
        """Notify all callbacks about state changes"""
        for callback in self._state_change_callbacks:
            try:
                callback(change_type, value)
            except Exception as e:
                print(f"State change callback error for {change_type}: {e}")
    
    # Reset methods
    def reset_to_initial_state(self):
        """Reset application to initial state"""
        self.clear_data()
        self._is_conditions = False
        self._current_tab = 0
        self._tabs_enabled = [True, False, False, False, False]
        self._global_plot_data = {'figs': [], 'names': [], 'workbook': None}
        self._notify_state_change('app_reset', None)