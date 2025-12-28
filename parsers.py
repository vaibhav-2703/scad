"""
Survey data parser for South Total Station and DGPS equipment.
Supports CSV, DAT, and CRD file formats.
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path


class SurveyDataParser:
    """Parse survey data from various South instrument formats."""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.dat', '.crd', '.txt']
    
    def parse_file(self, file_path: str) -> pd.DataFrame:
        """
        Auto-detect format and parse survey data file.
        
        Returns:
            DataFrame with columns: Point_ID, Easting, Northing, Elevation, Description
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension == '.csv':
            return self._parse_csv(file_path)
        elif extension == '.dat':
            return self._parse_dat(file_path)
        elif extension == '.crd':
            return self._parse_crd(file_path)
        else:
            # Try CSV format as fallback
            return self._parse_csv(file_path)
    
    def _parse_csv(self, file_path: Path) -> pd.DataFrame:
        """Parse CSV format survey data."""
        try:
            # First, try reading with header to detect WKT format and column names
            df = pd.read_csv(file_path, encoding='utf-8', skipinitialspace=True)
            
            # Check if this is a WKT POLYGON format (has 'geometry' column)
            if 'geometry' in df.columns.str.lower() or 'geometry' in df.columns:
                return self._parse_wkt_polygon(df)
            
            # Normalize column names (remove spaces, make case-insensitive)
            df.columns = df.columns.str.strip().str.replace(' ', '_')
            original_cols = df.columns.tolist()
            
            # Map column names intelligently
            col_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ['point_id', 'pointid', 'point', 'id']:
                    col_mapping[col] = 'Point_ID'
                elif col_lower in ['easting', 'east', 'x']:
                    col_mapping[col] = 'Easting'
                elif col_lower in ['northing', 'north', 'y']:
                    col_mapping[col] = 'Northing'
                elif col_lower in ['elevation', 'elev', 'z']:
                    col_mapping[col] = 'Elevation'
                elif col_lower in ['description', 'desc', 'remarks', 'note']:
                    col_mapping[col] = 'Description'
            
            # Rename columns that we've mapped
            df = df.rename(columns=col_mapping)
            
            # If we don't have required columns yet, try to infer from remaining columns
            # Prioritize Easting/Northing over Latitude/Longitude if both exist
            if 'Easting' not in df.columns:
                # Look for numeric columns that could be easting
                for col in df.columns:
                    if col not in ['Point_ID', 'Easting', 'Northing', 'Elevation', 'Description']:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            # Check if column name suggests easting
                            if 'lat' not in col.lower() and 'long' not in col.lower():
                                df = df.rename(columns={col: 'Easting'})
                                break
            
            if 'Northing' not in df.columns:
                # Similar for northing
                for col in df.columns:
                    if col not in ['Point_ID', 'Easting', 'Northing', 'Elevation', 'Description']:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            if 'lat' not in col.lower() and 'long' not in col.lower():
                                df = df.rename(columns={col: 'Northing'})
                                break
            
            # If still missing, try positional mapping (Point_ID, Easting, Northing should be first 3)
            if 'Point_ID' not in df.columns:
                if len(df.columns) > 0:
                    df = df.rename(columns={df.columns[0]: 'Point_ID'})
            
            if 'Easting' not in df.columns:
                # Find first numeric column after Point_ID
                for i, col in enumerate(df.columns):
                    if col != 'Point_ID' and pd.api.types.is_numeric_dtype(df[col]):
                        df = df.rename(columns={col: 'Easting'})
                        break
            
            if 'Northing' not in df.columns:
                # Find next numeric column after Point_ID and Easting
                for col in df.columns:
                    if col not in ['Point_ID', 'Easting'] and pd.api.types.is_numeric_dtype(df[col]):
                        df = df.rename(columns={col: 'Northing'})
                        break
            
            # Add missing optional columns
            if 'Elevation' not in df.columns:
                # Try to find elevation column
                elev_found = False
                for col in df.columns:
                    if col not in ['Point_ID', 'Easting', 'Northing', 'Description']:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            df = df.rename(columns={col: 'Elevation'})
                            elev_found = True
                            break
                if not elev_found:
                    df['Elevation'] = 0.0
            else:
                # Convert elevation if it exists
                df['Elevation'] = pd.to_numeric(df['Elevation'], errors='coerce').fillna(0.0)
            
            if 'Description' not in df.columns:
                # Try to find description column (look for text columns)
                desc_found = False
                for col in df.columns:
                    if col not in ['Point_ID', 'Easting', 'Northing', 'Elevation']:
                        col_lower = col.lower()
                        if col_lower in ['code', 'description', 'desc', 'remarks', 'note']:
                            df = df.rename(columns={col: 'Description'})
                            desc_found = True
                            break
                if not desc_found:
                    # Use last text column if available
                    for col in reversed(df.columns):
                        if col not in ['Point_ID', 'Easting', 'Northing', 'Elevation']:
                            if not pd.api.types.is_numeric_dtype(df[col]):
                                df = df.rename(columns={col: 'Description'})
                                desc_found = True
                                break
                if not desc_found:
                    df['Description'] = ''
            
            # Clean and standardize
            df = df.dropna(how='all')  # Remove empty rows
            
            # Remove rows without point ID
            if 'Point_ID' in df.columns:
                df = df[df['Point_ID'].notna()]
            else:
                raise ValueError("Could not identify Point_ID column")
            
            # Clean point IDs
            df['Point_ID'] = df['Point_ID'].astype(str).str.strip()
            
            # Convert coordinates to float
            df['Easting'] = pd.to_numeric(df['Easting'], errors='coerce')
            df['Northing'] = pd.to_numeric(df['Northing'], errors='coerce')
            
            # Remove rows with invalid coordinates
            df = df[df['Easting'].notna() & df['Northing'].notna()]
            
            if len(df) == 0:
                raise ValueError("No valid survey points found in file")
            
            # Clean descriptions
            if 'Description' in df.columns:
                df['Description'] = df['Description'].astype(str).fillna('').str.strip()
            
            # Return only the standard columns we need
            result_df = df[['Point_ID', 'Easting', 'Northing', 'Elevation', 'Description']].copy()
            
            return result_df.reset_index(drop=True)
            
        except Exception as e:
            raise ValueError(f"Error parsing CSV file: {str(e)}")
    
    def _parse_wkt_polygon(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse WKT POLYGON format from CSV.
        Converts lat/lon to UTM coordinates.
        Preserves polygon order for proper boundary connection.
        Adds Polygon_Name column to track which polygon each point belongs to.
        """
        from utils import CoordinateTransformer
        
        data = []
        point_counter = 1
        
        for idx, row in df.iterrows():
            geometry = str(row.get('geometry', ''))
            name = str(row.get('Name', f'polygon_{idx}'))
            area = row.get('Area', '')
            
            # Extract POLYGON coordinates
            if not geometry.upper().startswith('POLYGON'):
                continue
            
            # Parse POLYGON((lon lat, lon lat, ...))
            # Extract coordinates using regex
            # Match POLYGON((...)) pattern
            match = re.search(r'POLYGON\(\((.*?)\)\)', geometry, re.IGNORECASE)
            if not match:
                continue
            
            coords_str = match.group(1)
            
            # Split by comma to get coordinate pairs
            coord_pairs = [cp.strip() for cp in coords_str.split(',') if cp.strip()]
            
            # Transform coordinates from WGS84 to UTM
            # Auto-detect UTM zone based on longitude (India: zones 43-46)
            # Use first coordinate to determine zone
            if coord_pairs:
                first_coord = coord_pairs[0].split()
                if len(first_coord) >= 1:
                    try:
                        lon = float(first_coord[0])
                        # Determine UTM zone (India: 43-46)
                        utm_zone = int((lon + 180) / 6) + 1
                        if utm_zone < 43:
                            utm_zone = 43
                        elif utm_zone > 46:
                            utm_zone = 46
                        target_crs = f'EPSG:326{utm_zone}'
                    except:
                        target_crs = 'EPSG:32644'  # Default to zone 44
            else:
                target_crs = 'EPSG:32644'
            
            transformer = CoordinateTransformer(source_crs='EPSG:4326', target_crs=target_crs)
            
            polygon_points = []
            vertex_num = 1
            for coord_pair in coord_pairs:
                coord_pair = coord_pair.strip()
                parts = coord_pair.split()
                if len(parts) >= 2:
                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        
                        # Transform to UTM
                        easting, northing = transformer.transform_coordinates(lon, lat)
                        
                        # Create point ID (preserve order with vertex number)
                        point_id = f"{name}_{vertex_num:03d}"
                        vertex_num += 1
                        
                        polygon_points.append({
                            'Point_ID': point_id,
                            'Easting': easting,
                            'Northing': northing,
                            'Elevation': 0.0,
                            'Description': f"{name} - V{vertex_num-1}",
                            'Polygon_Name': name,  # Track which polygon this point belongs to
                            'Polygon_Area': area    # Store original area from CSV
                        })
                    except (ValueError, IndexError):
                        continue
            
            # Add all points from this polygon (order preserved)
            data.extend(polygon_points)
            point_counter += len(polygon_points)
        
        if not data:
            raise ValueError("No valid POLYGON data found in WKT format")
        
        result_df = pd.DataFrame(data)
        return result_df.reset_index(drop=True)
    
    def _parse_dat(self, file_path: Path) -> pd.DataFrame:
        """Parse DAT format (South Total Station format).
        
        Supports two formats:
        1. Point_ID, Easting, Northing, Elevation, Description (space or comma separated)
        2. Point_ID, Description, Easting, Northing, Elevation (comma-separated)
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            if not lines:
                raise ValueError("DAT file is empty")
            
            # Detect format by checking first non-empty line
            format_detected = None
            sample_line = None
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    sample_line = line
                    break
            
            if not sample_line:
                raise ValueError("No valid data lines found in DAT file")
            
            # Check if comma-separated or space-separated
            is_comma_separated = ',' in sample_line
            
            if is_comma_separated:
                parts = [p.strip() for p in sample_line.split(',') if p.strip()]
                if len(parts) >= 5:
                    # Smart format detection:
                    # Format 1: Point_ID, Easting, Northing, Elevation, Description
                    # Format 2: Point_ID, Description, Easting, Northing, Elevation
                    # 
                    # Strategy: Check if columns 1&2 OR columns 2&3 are both large coordinates
                    # Typical UTM coordinates are > 100000 for Easting and > 1000000 for Northing
                    try:
                        col1 = float(parts[1])
                        col2 = float(parts[2])
                        col3 = float(parts[3]) if len(parts) > 3 else 0
                        
                        # If columns 2&3 are both large (Easting & Northing), format is description_second
                        if col2 > 10000 and col3 > 10000:
                            format_detected = 'description_second'
                        # If columns 1&2 are both large (Easting & Northing), format is standard
                        elif col1 > 10000 and col2 > 10000:
                            format_detected = 'standard'
                        # If column 2 is large but column 1 is small, likely description_second
                        elif col2 > 10000 and col1 < 10000:
                            format_detected = 'description_second'
                        else:
                            # Fallback: check if column 1 can be a coordinate
                            format_detected = 'standard' if col1 > 10000 else 'description_second'
                    except (ValueError, IndexError):
                        # If column 2 is not numeric, it's description_second format
                        try:
                            float(parts[1])
                            format_detected = 'standard'
                        except ValueError:
                            format_detected = 'description_second'
                elif len(parts) >= 3:
                    format_detected = 'standard'
            else:
                # Space-separated, use standard format
                format_detected = 'standard'
            
            data = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if is_comma_separated:
                    parts = [p.strip() for p in line.split(',') if p.strip()]
                else:
                    # Space-separated
                    parts = re.split(r'\s+', line)
                    parts = [p.strip() for p in parts if p.strip()]
                
                if len(parts) < 3:
                    continue
                
                try:
                    if format_detected == 'description_second' and len(parts) >= 5:
                        # Format: Point_ID, Description, Easting, Northing, Elevation
                        point_id = parts[0]
                        description = parts[1]
                        easting = float(parts[2])
                        northing = float(parts[3])
                        elevation = float(parts[4]) if len(parts) > 4 else 0.0
                    else:
                        # Standard format: Point_ID, Easting, Northing, Elevation, Description
                        point_id = parts[0]
                        easting = float(parts[1])
                        northing = float(parts[2])
                        elevation = float(parts[3]) if len(parts) > 3 else 0.0
                        description = ' '.join(parts[4:]) if len(parts) > 4 else ''
                    
                    data.append({
                        'Point_ID': point_id,
                        'Easting': easting,
                        'Northing': northing,
                        'Elevation': elevation,
                        'Description': description
                    })
                except (ValueError, IndexError) as e:
                    # Skip invalid lines
                    continue
            
            if not data:
                raise ValueError("No valid data found in DAT file")
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            raise ValueError(f"Error parsing DAT file: {str(e)}")
    
    def _parse_crd(self, file_path: Path) -> pd.DataFrame:
        """Parse CRD format (coordinate file)."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            data = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # CRD format: Point_ID,Easting,Northing,Elevation,Description
                parts = line.split(',')
                if len(parts) >= 3:
                    point_id = parts[0].strip()
                    try:
                        easting = float(parts[1].strip())
                        northing = float(parts[2].strip())
                        elevation = float(parts[3].strip()) if len(parts) > 3 else 0.0
                        description = parts[4].strip() if len(parts) > 4 else ''
                        
                        data.append({
                            'Point_ID': point_id,
                            'Easting': easting,
                            'Northing': northing,
                            'Elevation': elevation,
                            'Description': description
                        })
                    except ValueError:
                        continue
            
            if not data:
                raise ValueError("No valid data found in CRD file")
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            raise ValueError(f"Error parsing CRD file: {str(e)}")
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate survey data for common issues.
        
        Returns:
            (is_valid, list_of_warnings)
        """
        warnings = []
        
        if df.empty:
            return False, ["No data found"]
        
        # Check for missing coordinates
        missing_coords = df[df['Easting'].isna() | df['Northing'].isna()]
        if not missing_coords.empty:
            warnings.append(f"Found {len(missing_coords)} points with missing coordinates")
        
        # Check for duplicate point IDs
        duplicates = df[df['Point_ID'].duplicated()]
        if not duplicates.empty:
            warnings.append(f"Found {len(duplicates)} duplicate point IDs")
        
        # Check for outliers (points far from centroid)
        if len(df) > 1:
            centroid_easting = df['Easting'].median()
            centroid_northing = df['Northing'].median()
            distances = np.sqrt(
                (df['Easting'] - centroid_easting)**2 + 
                (df['Northing'] - centroid_northing)**2
            )
            median_distance = distances.median()
            outliers = df[distances > 5 * median_distance]
            if not outliers.empty:
                warnings.append(f"Found {len(outliers)} potential outlier points")
        
        # Check coordinate ranges (typical India UTM ranges)
        if df['Easting'].min() < 100000 or df['Easting'].max() > 1000000:
            warnings.append("Easting coordinates outside typical UTM range (100000-1000000)")
        
        if df['Northing'].min() < 0 or df['Northing'].max() > 10000000:
            warnings.append("Northing coordinates outside typical UTM range")
        
        return True, warnings

