import requests
import traceback

class ErrorHandler:
    @staticmethod
    def log_error(message, error=None, output_widget=None, show_traceback=False):
        """Unified error logging with consistent formatting"""
        if error:
            full_message = f"Error: {message}: {str(error)}"
        else:
            full_message = f"Error: {message}"
        
        if output_widget:
            with output_widget:
                from IPython.display import clear_output
                clear_output(wait=True)
                print(full_message)
                if show_traceback and error:
                    traceback.print_exc()
        else:
            print(full_message)
            if show_traceback and error:
                traceback.print_exc()
    
    @staticmethod
    def log_info(message, output_widget=None):
        """Log informational messages"""
        if output_widget:
            with output_widget:
                print(message)
        else:
            print(message)
    
    @staticmethod
    def log_success(message, output_widget=None):
        """Log success messages"""
        if output_widget:
            with output_widget:
                print(message)
        else:
            print(message)
    
    @staticmethod
    def handle_auth_error(error, auth_manager):
        """Handle authentication-specific errors"""
        if isinstance(error, ValueError):
            auth_manager._update_status(f'Status: Error - {error}', 'red')
        elif isinstance(error, requests.exceptions.RequestException):
            auth_manager._handle_request_error(error)
        else:
            auth_manager._update_status(f'Status: Unexpected Error - {error}', 'red')
    
    @staticmethod
    def handle_data_loading_error(error, output_widget):
        """Handle data loading specific errors"""
        ErrorHandler.log_error("loading data", error, output_widget, show_traceback=True)
    
    @staticmethod
    def handle_plot_error(error, output_widget, plot_name=None):
        """Handle plotting specific errors"""
        if plot_name:
            ErrorHandler.log_error(f"displaying plot {plot_name}", error, output_widget)
        else:
            ErrorHandler.log_error("creating plots", error, output_widget, show_traceback=True)