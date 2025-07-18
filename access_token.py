import requests
import getpass
import os
import datetime
from pathlib import Path

def get_token(url, name=None):
    try:
        return os.environ['NOMAD_CLIENT_ACCESS_TOKEN']
    except KeyError:
        user = name if name is not None else input("Username")
        print("Password:")
        password = getpass.getpass()
    
        # get token from the api:
        response = requests.get(f'{url}/auth/token', params=dict(username=user,password=password))
        if response.status_code == 401:
            raise Exception(response.json()["detail"])
        return response.json()['access_token']

def log_notebook_usage(log_filename="notebook_usage.log"):
    """
    Log notebook usage to a file at the same level as this .py file.
    
    Args:
        log_filename (str): Name of the log file (default: "notebook_usage.log")
    """
    try:
        # Get the directory where this .py file is located
        script_dir = Path(__file__).parent
        log_path = script_dir / log_filename
        
        # Get current date and time
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # Get user from environment variable
        user = os.environ.get("NOMAD_CLIENT_USER", "Unknown")
        
        # Get file name from environment variable and determine app type
        session_name = os.environ.get('JPY_SESSION_NAME', 'Unknown')
        
        if session_name != 'Unknown':
            # Regular Jupyter notebook
            file = session_name.split("/", 5)[-1].split(".")[0]
            app_type = "jupyter"
        else:
            # Fallback: Check for Voilà URL first
            voila_url = os.environ.get('VOILA_REQUEST_URL', '')
            if voila_url:
                # Extract notebook name from Voilà URL
                # URLs typically look like: /voila/render/notebook.ipynb or /notebook.ipynb
                file = voila_url.split('/',12)[-1].replace('.ipynb', '')
                if not file:  # In case URL ends with /
                    file = voila_url.split('/')[-2].replace('.ipynb', '')
                app_type = "voila"
            else:
                # Additional fallback methods
                cwd = os.getcwd()
                potential_file = os.path.basename(cwd)
                
                # Look for .ipynb files in current directory
                current_dir = Path(cwd)
                ipynb_files = list(current_dir.glob("*.ipynb"))
                
                if ipynb_files:
                    # If there's only one notebook, use it
                    if len(ipynb_files) == 1:
                        file = ipynb_files[0].stem
                        app_type = "voila"
                    else:
                        # Multiple notebooks - use directory name with indicator
                        file = f"{potential_file}_voila"
                        app_type = "voila"
                else:
                    # No notebooks found, use directory name or Unknown
                    file = potential_file if potential_file else "Unknown_voila"
                    app_type = "unknown"
        
        # Create log entry
        log_entry = f"{date_str},{time_str},{user},{app_type},{file}\n"
        
        # Write to log file (create if doesn't exist, append if it does)
        with open(log_path, 'a', encoding='utf-8') as f:
            # If file is empty/new, add header
            if log_path.stat().st_size == 0:
                f.write("Date,Time,User,App,File\n")
            f.write(log_entry)
        
        #print(f"Logged usage: {date_str} {time_str} - {user} - {file}")
        
    except Exception as e:
        print(f"Error logging notebook usage: {e}")