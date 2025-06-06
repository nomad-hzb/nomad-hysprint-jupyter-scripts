import pandas as pd
import os

class DataProcessor:
    @staticmethod
    def parse_sample_id(sample_path):
        """Parse sample ID from file path consistently"""
        return sample_path.split('/')[-1].split('.')[0]
    
    @staticmethod
    def extract_subbatch(sample_path):
        """Extract subbatch from sample path"""
        return DataProcessor.parse_sample_id(sample_path).split('_')[-2]
    
    @staticmethod
    def extract_identifier_base(sample_path):
        """Extract base identifier from sample path"""
        parsed_id = DataProcessor.parse_sample_id(sample_path)
        return '_'.join(parsed_id.split('_')[3:-1])
    
    @staticmethod
    def create_full_identifier(sample_path):
        """Create full identifier from sample path"""
        return DataProcessor.parse_sample_id(sample_path)
    
    @staticmethod
    def apply_identifier_mapping(identifier, identifiers_dict):
        """Apply identifier mapping with fallback"""
        if identifiers_dict:
            base_id = "_".join(identifier.split("_")[:-1])
            variation = identifiers_dict.get(identifier, "No variation specified")
            return f'{base_id}&{variation}'
        else:
            # Fallback when no identifiers dict
            parts = identifier.split("_")
            return "_".join(parts[:-1]) if len(parts) > 1 else identifier
    
    @staticmethod
    def extract_sample_name(sample_path):
        """Extract final sample name from path"""
        return DataProcessor.parse_sample_id(sample_path).split('_', 4)[-1]
    
    @staticmethod
    def process_jv_dataframe(df_jvc, identifiers=None):
        """Process JV dataframe with consistent transformations"""
        # Apply all transformations
        df_jvc["subbatch"] = df_jvc["sample"].apply(DataProcessor.extract_subbatch)
        df_jvc["identifier"] = df_jvc["sample"].apply(DataProcessor.create_full_identifier)
        
        # Apply identifier mapping
        df_jvc["identifier"] = df_jvc["identifier"].apply(
            lambda x: DataProcessor.apply_identifier_mapping(x, identifiers)
        )
        
        # Extract final sample names
        df_jvc["sample"] = df_jvc["sample"].apply(DataProcessor.extract_sample_name)
        
        return df_jvc
    
    @staticmethod
    def process_curves_dataframe(df_curves):
        """Process curves dataframe with consistent transformations"""
        df_curves["sample"] = df_curves["sample"].apply(DataProcessor.extract_sample_name)
        return df_curves

    @staticmethod
    def create_measurement_link(entry_info):
        """Create clickable links for measurements"""
        from auth_manager import APIClient, STRINGS
        api_client = APIClient(STRINGS['SE_OASIS_URL'], STRINGS['API_ENDPOINT'])
        
        entry_type = entry_info["entry_type"]
        entry_id = entry_info["entry_id"]
        
        url = api_client.get_entry_url(entry_id, entry_type)
        display_name = entry_type.split("_")[-1]
        return f'<a href="{url}" rel="noopener noreferrer" target="_blank">{display_name}</a>'

class JVDataExtractor:
    """Utility class for extracting and processing JV data"""
    
    @staticmethod
    def extract_cell_properties(cell_name):
        """Extract illumination, cell, and direction from cell name"""
        illum = "Dark" if "dark" in cell_name.lower() else "Light"
        cell = cell_name[0]
        direction = "Forward" if "for" in cell_name.lower() else "Reverse"
        return illum, cell, direction
    
    @staticmethod
    def create_jv_row(curve_data, file_name, batch_name, illum, cell, direction):
        """Create a JV data row from curve data"""
        return [
            curve_data["open_circuit_voltage"],
            -curve_data["short_circuit_current_density"],
            100 * curve_data["fill_factor"],
            curve_data["efficiency"],
            curve_data["potential_at_maximum_power_point"],
            -curve_data["current_density_at_maximun_power_point"],
            -curve_data["potential_at_maximum_power_point"] * curve_data["current_density_at_maximun_power_point"],
            curve_data["series_resistance"],
            curve_data["shunt_resistance"],
            file_name,
            batch_name,
            "w",
            cell,
            direction,
            illum
        ]
    
    @staticmethod
    def create_curve_rows(curve_data, file_name, batch_name, illum, cell, direction):
        """Create voltage and current density rows for curves data"""
        base_id = "_".join([file_name, batch_name, "w"])
        
        row_v = [
            "_".join(["Voltage (V)", cell, direction, illum]),
            file_name, batch_name, "w", "Voltage (V)",
            cell, direction, illum
        ]
        row_v.extend(curve_data["voltage"])
        
        row_j = [
            "_".join(["Current Density(mA/cm2)", cell, direction, illum]),
            file_name, batch_name, "w", "Current Density(mA/cm2)",
            cell, direction, illum
        ]
        row_j.extend(curve_data["current_density"])
        
        return row_v, row_j


def get_jv_data_for_analysis(sample_ids, url, token, load_status_output):
    """Extract JV data for analysis using utility classes - WITH STATUS EXTRACTION"""
    # Import everything we need at the top
    import os
    import pandas as pd

    def extract_status_from_metadata(data, metadata):
        """
        Extract status from API metadata containing filename
        """
        import re
        
        # Look for filename in metadata
        filename_candidates = [
            metadata.get('mainfile', ''),
            metadata.get('upload_name', ''),
            metadata.get('entry_name', ''),
            metadata.get('filename', ''),
        ]
        
        for candidate in filename_candidates:
            if candidate:
                # Extract status from filename like "HZB_JJ_1_B_C-8.JJ_1_B_8_L1_jv.jv.txt"
                status_match = re.search(r'_([LD]\d+)(?:_3min)?_', candidate)
                if not status_match:
                    status_match = re.search(r'([LD]\d+)', candidate)
                
                if status_match:
                    return status_match.group(1)
        
        return 'N/A'
    
    # Import here to avoid issues
    try:
        from api_calls import get_all_JV
    except ImportError:
        import sys
        sys.path.append(os.path.dirname(os.getcwd()))
        from api_calls import get_all_JV
    
    columns_jvc = ['Voc(V)','Jsc(mA/cm2)','FF(%)','PCE(%)','V_mpp(V)','J_mpp(mA/cm2)',
                  'P_mpp(mW/cm2)','R_series(Ohmcm2)','R_shunt(Ohmcm2)','sample','batch',
                  'condition','cell','direction','ilum','status']  # ADD STATUS COLUMN
    columns_cur = ['index','sample','batch','condition','variable','cell','direction','ilum','status']  # ADD STATUS COLUMN
    rows_jvc = []
    rows_cur = []

    try:
        # Simple progress update
        print(f"Loading JV data for {len(sample_ids)} samples...")
        
        all_jvs = get_all_JV(url, token, sample_ids)
        
        for sid in sample_ids:
            jv_res = all_jvs.get(sid, [])
            print(f"Processing sample {sid}: {len(jv_res)} measurements")
            
            for jv_data, jv_md in jv_res:
                # EXTRACT STATUS FROM METADATA HERE
                status = extract_status_from_metadata(jv_data, jv_md)
                print(f"  Extracted status: {status} from metadata")
                
                for curve_data in jv_data["jv_curve"]:
                    # Extract common properties
                    file_name = os.path.join("../", jv_md["upload_id"], jv_data.get("data_file"))
                    batch_name = file_name.split("/")[1]
                    illum, cell, direction = JVDataExtractor.extract_cell_properties(curve_data["cell_name"])
                    
                    # Create JV row - ADD STATUS TO THE ROW
                    jv_row = JVDataExtractor.create_jv_row(
                        curve_data, file_name, batch_name, illum, cell, direction
                    )
                    jv_row.append(status)  # ADD STATUS AS LAST COLUMN
                    rows_jvc.append(jv_row)
                    
                    # Create curve rows - ADD STATUS TO ROWS
                    row_v, row_j = JVDataExtractor.create_curve_rows(
                        curve_data, file_name, batch_name, illum, cell, direction
                    )
                    
                    # ADD STATUS TO BOTH VOLTAGE AND CURRENT ROWS
                    row_v.append(status)
                    row_j.append(status)
                    
                    # Update columns for curves if needed
                    for i in range(len(row_v[9:])):  # Changed from 8 to 9 to account for status
                        if i not in columns_cur:
                            columns_cur.append(i)
                    
                    rows_cur.append(row_v)
                    rows_cur.append(row_j)
        
        print(f"Created {len(rows_jvc)} JV records and {len(rows_cur)} curve records")
        
    except Exception as e:
        print(f"Error during data processing: {e}")
        import traceback
        traceback.print_exc()
        # Return empty dataframes on error
        return pd.DataFrame(columns=columns_jvc), pd.DataFrame(columns=columns_cur)
    
    # Create DataFrames
    df_jvc = pd.DataFrame(rows_jvc, columns=columns_jvc)
    df_cur = pd.DataFrame(rows_cur, columns=columns_cur)
    
    print(f"Status distribution in JVC: {df_jvc['status'].value_counts().to_dict()}")
    
    return df_jvc, df_cur