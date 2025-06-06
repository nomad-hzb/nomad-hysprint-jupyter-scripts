import requests
import os
import json

# --- Constants ---
STRINGS = {
    'STATUS_NOT_AUTH': 'Status: Not Authenticated',
    'STATUS_AUTH_CHANGED': 'Status: Not Authenticated (Method changed)',
    'STATUS_AUTHENTICATING': 'Status: Authenticating...',
    'AUTH_REQUIRED': 'Authentication required. Please authenticate first in Connection Settings.',
    'SE_OASIS_URL': 'https://nomad-hzb-se.de',
    'API_ENDPOINT': '/nomad-oasis/api/v1',
    'DATA_LOADED_SUCCESS': 'Data Loaded Successfully!',
    'LOADING_DATA': 'Loading Data',
    'VARIABLES_NOT_LOADED': '⚠️ Variables not loaded',
    'VARIABLES_LOADED': 'Variables loaded',
    'ERROR_PREFIX': 'Error: ',
    'STATUS_PREFIX': 'Status: '
}

# --- Layout Constants ---
LAYOUT = {
    'TIMEOUT_STANDARD': 10,
    'BUTTON_MIN_WIDTH': '150px',
    'DROPDOWN_WIDTH': '200px',
    'DROPDOWN_WIDE': '250px',
    'DROPDOWN_EXTRA_WIDE': '300px',
    'TEXT_INPUT_WIDTH': '100px',
    'LABEL_WIDTH': '80px',
    'TOKEN_INPUT_WIDTH': '95%',
    'CONTAINER_MAX_WIDTH': '1200px',
    'OUTPUT_MIN_HEIGHT': '100px',
    'OUTPUT_SCROLL_HEIGHT': '300px',
    'OUTPUT_SCROLL_WIDTH': '400px',
    'OUTPUT_LARGE_HEIGHT': '250px',
    'OUTPUT_MEDIUM_HEIGHT': '200px',
    'PLOT_MIN_WIDTH': '400px',
    'PLOT_MIN_HEIGHT': '300px',
    'PLOT_DEFAULT_WIDTH': '820px',
    'PLOT_DEFAULT_HEIGHT': '620px',
    'PLOT_CONTAINER_WIDTH': '800px',
    'PLOT_CONTAINER_HEIGHT': '600px',
    'RESIZE_HANDLE_SIZE': '15px',
    'MARGIN_STANDARD': '10px',
    'MARGIN_SMALL': '5px',
    'BORDER_STANDARD': '1px solid #ccc',
    'BORDER_LIGHT': '1px solid #eee',
    'BORDER_DARK': '1px solid #ddd',
    'PADDING_STANDARD': '10px',
    'PADDING_LARGE': '15px',
    'DPI_HIGH': 150,
    'RESIZE_POLL_INTERVAL': 200,
    'IFRAME_RESIZE_DELAY': 100
}

# --- API Client for URL Construction ---
class APIClient:
    def __init__(self, base_url, api_endpoint):
        self.base_url = base_url
        self.api_endpoint = api_endpoint
        self.full_api_url = f"{base_url}{api_endpoint}"
    
    # Authentication endpoints
    def get_auth_token_url(self):
        """Get URL for token authentication"""
        return f"{self.full_api_url}/auth/token"
    
    def get_user_verification_url(self):
        """Get URL for user verification"""
        return f"{self.full_api_url}/users/me"
    
    # Data entry URLs
    def get_entry_data_url(self, entry_id):
        """Get URL for entry data view"""
        return f"{self.base_url}/nomad-oasis/gui/entry/id/{entry_id}/data/data"
    
    def get_entry_image_preview_url(self, entry_id, image_index=0):
        """Get URL for entry image preview (for SEM images)"""
        return f"{self.base_url}/nomad-oasis/gui/entry/id/{entry_id}/data/data/images:{image_index}/image_preview/preview"
    
    # Generic entry URL
    def get_entry_url(self, entry_id, entry_type=None):
        """Get appropriate URL based on entry type"""
        if entry_type and "SEM" in entry_type:
            return self.get_entry_image_preview_url(entry_id)
        else:
            return self.get_entry_data_url(entry_id)
    
    # API endpoint getters
    def get_api_url(self):
        """Get the full API URL"""
        return self.full_api_url
    
    def get_base_url(self):
        """Get the base URL"""
        return self.base_url

# --- Authentication Manager ---
class AuthenticationManager:
    def __init__(self, url_base, api_endpoint):
        self.url_base = url_base
        self.url = f"{url_base}{api_endpoint}"
        self.status_callback = None
        self.current_token = None
        self.current_user_info = None
        # Create API client instance for this auth manager
        self.api_client = APIClient(url_base, api_endpoint)
    
    def set_status_callback(self, callback):
        """Set callback function to update UI status"""
        self.status_callback = callback
    
    def _update_status(self, message, color=None):
        """Update status through callback if available"""
        if self.status_callback:
            self.status_callback(message, color)
    
    def authenticate_with_credentials(self, username, password):
        """Authenticate using username and password"""
        if not username or not password:
            raise ValueError("Username and Password are required.")
        
        auth_dict = dict(username=username, password=password)
        token_url = self.api_client.get_auth_token_url()
        
        try:
            response = requests.get(token_url, params=auth_dict, timeout=LAYOUT['TIMEOUT_STANDARD'])
            response.raise_for_status()
            
            token_data = response.json()
            if 'access_token' not in token_data:
                raise ValueError("Access token not found in response.")
            
            self.current_token = token_data['access_token']
            return self.current_token
            
        except requests.exceptions.RequestException as e:
            self._handle_request_error(e)
            raise
    
    def authenticate_with_token(self, token=None):
        """Authenticate using provided token or environment variable"""
        if token is None:
            token = os.environ.get('NOMAD_CLIENT_ACCESS_TOKEN')
            if not token:
                raise ValueError("Token not found in environment variable 'NOMAD_CLIENT_ACCESS_TOKEN'.")
        
        self.current_token = token
        return self.current_token
    
    def verify_token(self):
        """Verify the current token and get user info"""
        if not self.current_token:
            raise ValueError("No token available for verification.")
        
        verify_url = self.api_client.get_user_verification_url()
        headers = {'Authorization': f'Bearer {self.current_token}'}
        
        try:
            verify_response = requests.get(verify_url, headers=headers, timeout=LAYOUT['TIMEOUT_STANDARD'])
            verify_response.raise_for_status()
            
            self.current_user_info = verify_response.json()
            return self.current_user_info
            
        except requests.exceptions.RequestException as e:
            self._handle_request_error(e)
            raise
    
    def _handle_request_error(self, e):
        """Handle and format request errors consistently"""
        if e.response is not None:
            try:
                error_detail = e.response.json().get('detail', e.response.text)
                if isinstance(error_detail, list):
                    error_message = f"API Error ({e.response.status_code}): {json.dumps(error_detail)}"
                else:
                    error_message = f"API Error ({e.response.status_code}): {error_detail or e.response.text}"
            except json.JSONDecodeError:
                error_message = f"API Error ({e.response.status_code}): {e.response.text}"
        else:
            error_message = f"Network/API Error: {e}"
        
        self._update_status(f'Status: {error_message}', 'red')
    
    def get_auth_headers(self):
        """Get authorization headers for API calls"""
        if not self.current_token:
            raise ValueError("Not authenticated. Please authenticate first.")
        return {'Authorization': f'Bearer {self.current_token}'}
    
    def is_authenticated(self):
        """Check if currently authenticated"""
        return self.current_token is not None and self.current_user_info is not None
    
    def clear_authentication(self):
        """Clear current authentication state"""
        self.current_token = None
        self.current_user_info = None
    
    def get_user_display_name(self):
        """Get display name for current user"""
        if not self.current_user_info:
            return 'Unknown User'
        return self.current_user_info.get('name', self.current_user_info.get('username', 'Unknown User'))