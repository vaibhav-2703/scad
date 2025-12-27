"""
Utility functions for coordinate transformations, calculations, and geometry.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy.spatial import ConvexHull
import pyproj


class CoordinateTransformer:
    """Handle coordinate system transformations."""
    
    def __init__(self, source_crs: str = 'EPSG:4326', target_crs: str = 'EPSG:32644'):
        """
        Initialize transformer.
        
        Args:
            source_crs: Source CRS (default: WGS84)
            target_crs: Target CRS (default: UTM Zone 44N for India)
        """
        self.source_crs = source_crs
        self.target_crs = target_crs
        self.transformer = pyproj.Transformer.from_crs(
            source_crs, target_crs, always_xy=True
        )
    
    def transform_coordinates(self, lon: float, lat: float) -> Tuple[float, float]:
        """Transform coordinates from source to target CRS."""
        easting, northing = self.transformer.transform(lon, lat)
        return easting, northing
    
    def transform_dataframe(self, df: pd.DataFrame, 
                          lon_col: str = 'Longitude', 
                          lat_col: str = 'Latitude') -> pd.DataFrame:
        """Transform coordinates in DataFrame."""
        df = df.copy()
        df['Easting'], df['Northing'] = zip(*df.apply(
            lambda row: self.transform_coordinates(row[lon_col], row[lat_col]),
            axis=1
        ))
        return df


class GeometryCalculator:
    """Calculate geometric properties of survey data."""
    
    @staticmethod
    def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    @staticmethod
    def calculate_bearing(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """
        Calculate bearing from p1 to p2 in degrees.
        Returns bearing in degrees (0-360, where 0 is North).
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Calculate angle in radians
        angle = np.arctan2(dx, dy)
        
        # Convert to degrees and normalize to 0-360
        bearing = np.degrees(angle)
        if bearing < 0:
            bearing += 360
        
        return bearing
    
    @staticmethod
    def calculate_area(points: List[Tuple[float, float]]) -> float:
        """
        Calculate area of polygon using shoelace formula.
        Points should be in order (boundary points).
        """
        if len(points) < 3:
            return 0.0
        
        # Ensure closed polygon
        if points[0] != points[-1]:
            points = points + [points[0]]
        
        n = len(points)
        area = 0.0
        
        for i in range(n - 1):
            area += points[i][0] * points[i + 1][1]
            area -= points[i + 1][0] * points[i][1]
        
        return abs(area) / 2.0
    
    @staticmethod
    def calculate_perimeter(points: List[Tuple[float, float]]) -> float:
        """Calculate perimeter of polygon."""
        if len(points) < 2:
            return 0.0
        
        perimeter = 0.0
        for i in range(len(points)):
            next_i = (i + 1) % len(points)
            perimeter += GeometryCalculator.calculate_distance(
                points[i], points[next_i]
            )
        
        return perimeter
    
    @staticmethod
    def calculate_closure_error(points: List[Tuple[float, float]]) -> Tuple[float, float, float]:
        """
        Calculate closure error for closed traverse.
        
        Returns:
            (linear_error, angular_error, precision_ratio)
        """
        if len(points) < 3:
            return 0.0, 0.0, 0.0
        
        # Ensure closed polygon
        if points[0] != points[-1]:
            points = points + [points[0]]
        
        # Calculate linear closure error
        dx = points[-1][0] - points[0][0]
        dy = points[-1][1] - points[0][1]
        linear_error = np.sqrt(dx**2 + dy**2)
        
        # Calculate perimeter
        perimeter = GeometryCalculator.calculate_perimeter(points[:-1])
        
        # Calculate precision ratio (1:precision_ratio)
        if perimeter > 0:
            precision_ratio = perimeter / linear_error if linear_error > 0 else float('inf')
        else:
            precision_ratio = 0.0
        
        # Calculate angular error (sum of interior angles should be (n-2)*180)
        # Simplified: check if traverse closes
        angular_error = 0.0  # More complex calculation needed for full angular closure
        
        return linear_error, angular_error, precision_ratio
    
    @staticmethod
    def find_boundary_points(points: np.ndarray) -> List[int]:
        """
        Find boundary points using convex hull algorithm.
        Returns indices of boundary points in order.
        """
        if len(points) < 3:
            return list(range(len(points)))
        
        try:
            hull = ConvexHull(points)
            return hull.vertices.tolist()
        except:
            # Fallback: return all points if convex hull fails
            return list(range(len(points)))
    
    @staticmethod
    def order_boundary_points(points: np.ndarray, boundary_indices: List[int]) -> List[int]:
        """
        Order boundary points in clockwise or counter-clockwise order.
        Returns ordered indices.
        """
        if len(boundary_indices) < 3:
            return boundary_indices
        
        # Get boundary points
        boundary_points = points[boundary_indices]
        
        # Calculate centroid
        centroid = np.mean(boundary_points, axis=0)
        
        # Calculate angles from centroid
        angles = []
        for point in boundary_points:
            dx = point[0] - centroid[0]
            dy = point[1] - centroid[1]
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        
        # Sort by angle
        sorted_indices = sorted(range(len(boundary_indices)), 
                               key=lambda i: angles[i])
        
        # Return original indices in sorted order
        return [boundary_indices[i] for i in sorted_indices]
    
    @staticmethod
    def get_boundaries_by_prefix(df: pd.DataFrame, gap_threshold: float = 50.0, 
                                  min_area_ratio: float = 0.01,
                                  split_on_gaps: bool = False) -> dict:
        """
        Group survey points by their prefix (e.g., a001-a060 = 'a', b001-b102 = 'b').
        
        For DGPS/Total Station data, points are surveyed in sequential order.
        Points within each prefix group should be connected in file order.
        
        By default (split_on_gaps=False), each prefix forms ONE continuous polyline,
        preserving the surveyor's path including traverses between areas.
        
        If split_on_gaps=True, large coordinate gaps split groups into separate polygons,
        and area-based filtering removes small reference point clusters.
        
        Args:
            df: DataFrame with Point_ID, Easting, Northing columns
            gap_threshold: Distance threshold (meters) to split (only if split_on_gaps=True)
            min_area_ratio: Minimum area as fraction of largest group (only if split_on_gaps=True)
            split_on_gaps: If True, split groups on large gaps. If False, keep as continuous path.
            
        Returns:
            dict of {group_name: [{'id': Point_ID, 'coords': (Easting, Northing)}, ...]}
            Points are in file order for sequential connection.
        """
        import re
        
        # Group by prefix - each prefix becomes ONE continuous polyline
        prefix_groups = {}
        for _, row in df.iterrows():
            point_id = str(row['Point_ID'])
            # Extract letter prefix (a, b, c, etc.)
            match = re.match(r'^([a-zA-Z]+)', point_id)
            prefix = match.group(1).upper() if match else 'DEFAULT'
            
            if prefix not in prefix_groups:
                prefix_groups[prefix] = []
            prefix_groups[prefix].append({
                'id': point_id,
                'coords': (float(row['Easting']), float(row['Northing']))
            })
        
        # If not splitting on gaps, return prefix groups directly (minimum 3 points)
        if not split_on_gaps:
            return {k: v for k, v in prefix_groups.items() if len(v) >= 3}
        
        # Otherwise, split each prefix group on large coordinate gaps
        all_sub_groups = []
        for prefix, points in prefix_groups.items():
            if len(points) < 3:
                all_sub_groups.append({'prefix': prefix, 'points': points})
                continue
            
            # Detect gaps and split
            sub_groups = []
            current_group = [points[0]]
            
            for i in range(1, len(points)):
                prev_coords = points[i-1]['coords']
                curr_coords = points[i]['coords']
                
                distance = np.sqrt(
                    (curr_coords[0] - prev_coords[0])**2 + 
                    (curr_coords[1] - prev_coords[1])**2
                )
                
                if distance > gap_threshold:
                    if len(current_group) >= 3:
                        sub_groups.append(current_group)
                    current_group = [points[i]]
                else:
                    current_group.append(points[i])
            
            if len(current_group) >= 3:
                sub_groups.append(current_group)
            
            if len(sub_groups) == 1:
                all_sub_groups.append({'prefix': prefix, 'points': sub_groups[0]})
            else:
                for idx, sub_group in enumerate(sub_groups):
                    group_name = f"{prefix}_{idx+1}"
                    all_sub_groups.append({'prefix': group_name, 'points': sub_group})
        
        # Calculate areas and filter by relative size
        group_areas = []
        for group_info in all_sub_groups:
            points = group_info['points']
            if len(points) >= 3:
                coords = [p['coords'] for p in points]
                area = GeometryCalculator.calculate_area(coords)
                group_areas.append((group_info, area))
            else:
                group_areas.append((group_info, 0.0))
        
        max_area = max([a for _, a in group_areas]) if group_areas else 0.0
        
        final_groups = {}
        for group_info, area in group_areas:
            if len(group_info['points']) < 3:
                continue
            if max_area > 0 and area >= min_area_ratio * max_area:
                final_groups[group_info['prefix']] = group_info['points']
            elif max_area == 0:
                final_groups[group_info['prefix']] = group_info['points']
        
        return final_groups
    
    @staticmethod
    def format_bearing(bearing: float) -> str:
        """Format bearing as degrees, minutes, seconds."""
        degrees = int(bearing)
        minutes_float = (bearing - degrees) * 60
        minutes = int(minutes_float)
        seconds = (minutes_float - minutes) * 60
        
        return f"{degrees}°{minutes}'{seconds:.2f}\""
    
    @staticmethod
    def format_distance(distance: float, unit: str = 'm') -> str:
        """Format distance with appropriate precision."""
        if distance < 1:
            return f"{distance * 100:.2f} cm"
        elif distance < 1000:
            return f"{distance:.3f} m"
        else:
            return f"{distance / 1000:.3f} km"
    
    @staticmethod
    def format_area(area: float) -> str:
        """Format area with appropriate units."""
        if area < 10000:  # Less than 1 hectare
            return f"{area:.2f} m² ({area / 4046.86:.4f} acres)"
        else:
            hectares = area / 10000
            return f"{hectares:.4f} hectares ({area:.2f} m²)"


def detect_coordinate_system(df: pd.DataFrame) -> str:
    """
    Auto-detect coordinate system based on coordinate ranges.
    Returns CRS code.
    """
    easting_min = df['Easting'].min()
    easting_max = df['Easting'].max()
    northing_min = df['Northing'].min()
    northing_max = df['Northing'].max()
    
    # UTM zones for India
    if 200000 <= easting_max <= 900000:
        if 0 <= northing_max <= 10000000:
            # Determine UTM zone based on easting
            if 200000 <= easting_max <= 500000:
                return 'EPSG:32644'  # UTM Zone 44N
            elif 500000 <= easting_max <= 800000:
                return 'EPSG:32645'  # UTM Zone 45N
            else:
                return 'EPSG:32646'  # UTM Zone 46N
    
    # WGS84 (lat/lon)
    if -180 <= easting_min <= 180 and -90 <= northing_min <= 90:
        return 'EPSG:4326'
    
    # Default to UTM 44N
    return 'EPSG:32644'

