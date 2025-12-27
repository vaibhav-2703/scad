"""
Contour Map Generator
Generates professional contour maps from survey elevation data.
"""

import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter
import ezdxf
from ezdxf import units
from ezdxf.enums import TextEntityAlignment
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import math


class ContourGenerator:
    """
    Generate contour maps from scattered survey points.
    Produces DXF files with proper contour layers and labels.
    """
    
    # Sheet sizes in mm
    SHEET_SIZES = {
        'A1': (841, 594),
        'A2': (594, 420),
        'A3': (420, 297),
    }
    
    # Contour styling
    CONTOUR_COLORS = {
        'major': 1,      # Red for major contours
        'minor': 8,      # Gray for minor contours
        'index': 3,      # Green for index contours (every 5th major)
    }
    
    def __init__(self, 
                 major_interval: float = 1.0,
                 minor_interval: float = 0.5,
                 index_interval: float = 5.0,
                 grid_resolution: int = 100,
                 smoothing: float = 1.0,
                 sheet_size: str = 'A1'):
        """
        Initialize contour generator.
        
        Args:
            major_interval: Interval for major contour lines (meters)
            minor_interval: Interval for minor contour lines (meters)
            index_interval: Interval for index/labeled contours (meters)
            grid_resolution: Number of grid points for interpolation
            smoothing: Gaussian smoothing factor (0 = none)
            sheet_size: Output sheet size
        """
        self.major_interval = major_interval
        self.minor_interval = minor_interval
        self.index_interval = index_interval
        self.grid_resolution = grid_resolution
        self.smoothing = smoothing
        self.sheet_size = sheet_size
        
        self.sheet_width, self.sheet_height = self.SHEET_SIZES.get(sheet_size, (841, 594))
        self.margin = 20
        
        self.doc = None
        self.msp = None
        
    def create_document(self):
        """Create DXF document with contour layers."""
        self.doc = ezdxf.new('R2010')
        self.doc.units = units.MM
        self.msp = self.doc.modelspace()
        
        # Create layers
        self.doc.layers.add('BORDER', color=7)
        self.doc.layers.add('CONTOUR_MAJOR', color=self.CONTOUR_COLORS['major'])
        self.doc.layers.add('CONTOUR_MINOR', color=self.CONTOUR_COLORS['minor'])
        self.doc.layers.add('CONTOUR_INDEX', color=self.CONTOUR_COLORS['index'])
        self.doc.layers.add('CONTOUR_LABELS', color=self.CONTOUR_COLORS['index'])
        self.doc.layers.add('SURVEY_POINTS', color=4)  # Cyan
        self.doc.layers.add('TITLE_BLOCK', color=7)
        self.doc.layers.add('NORTH_ARROW', color=7)
        
        return self
    
    def interpolate_grid(self, points: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate scattered points to a regular grid.
        
        Args:
            points: List of dicts with 'x', 'y', 'z' keys
            
        Returns:
            Tuple of (X grid, Y grid, Z grid)
        """
        # Extract coordinates
        x = np.array([p['x'] for p in points])
        y = np.array([p['y'] for p in points])
        z = np.array([p['z'] for p in points])
        
        # Create grid
        xi = np.linspace(x.min(), x.max(), self.grid_resolution)
        yi = np.linspace(y.min(), y.max(), self.grid_resolution)
        X, Y = np.meshgrid(xi, yi)
        
        # Interpolate using cubic method, fall back to linear if needed
        try:
            Z = interpolate.griddata((x, y), z, (X, Y), method='cubic')
        except:
            Z = interpolate.griddata((x, y), z, (X, Y), method='linear')
        
        # Fill NaN values with nearest neighbor
        mask = np.isnan(Z)
        if mask.any():
            Z_nearest = interpolate.griddata((x, y), z, (X, Y), method='nearest')
            Z[mask] = Z_nearest[mask]
        
        # Apply smoothing
        if self.smoothing > 0:
            Z = gaussian_filter(Z, sigma=self.smoothing)
        
        return X, Y, Z
    
    def extract_contours(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Dict:
        """
        Extract contour lines at specified intervals.
        
        Returns:
            Dict with 'major', 'minor', 'index' contour collections
        """
        import matplotlib.pyplot as plt
        
        z_min, z_max = np.nanmin(Z), np.nanmax(Z)
        
        # Generate contour levels
        def get_levels(interval, z_min, z_max):
            start = np.floor(z_min / interval) * interval
            end = np.ceil(z_max / interval) * interval
            return np.arange(start, end + interval, interval)
        
        minor_levels = get_levels(self.minor_interval, z_min, z_max)
        major_levels = get_levels(self.major_interval, z_min, z_max)
        index_levels = get_levels(self.index_interval, z_min, z_max)
        
        # Extract contours using matplotlib (we won't display, just extract geometry)
        fig, ax = plt.subplots()
        
        contours = {
            'minor': [],
            'major': [],
            'index': []
        }
        
        # Minor contours - use allsegs for matplotlib 3.8+ compatibility
        cs_minor = ax.contour(X, Y, Z, levels=minor_levels)
        for i, level in enumerate(cs_minor.levels):
            if i < len(cs_minor.allsegs):
                for seg in cs_minor.allsegs[i]:
                    if len(seg) > 1:
                        contours['minor'].append({
                            'level': float(level),
                            'vertices': seg.tolist()
                        })
        
        # Major contours
        cs_major = ax.contour(X, Y, Z, levels=major_levels)
        for i, level in enumerate(cs_major.levels):
            if i < len(cs_major.allsegs):
                for seg in cs_major.allsegs[i]:
                    if len(seg) > 1:
                        contours['major'].append({
                            'level': float(level),
                            'vertices': seg.tolist()
                        })
        
        # Index contours
        for i, level in enumerate(cs_major.levels):
            if level in index_levels and i < len(cs_major.allsegs):
                for seg in cs_major.allsegs[i]:
                    if len(seg) > 1:
                        contours['index'].append({
                            'level': float(level),
                            'vertices': seg.tolist()
                        })
        
        plt.close(fig)
        
        return contours
    
    def _transform_point(self, x: float, y: float, extents: Tuple) -> Tuple[float, float]:
        """Transform real-world coordinates to sheet coordinates."""
        min_x, min_y, max_x, max_y = extents
        
        data_width = max_x - min_x
        data_height = max_y - min_y
        
        if data_width == 0 or data_height == 0:
            return (self.sheet_width / 2, self.sheet_height / 2)
        
        # Available drawing area
        avail_width = self.sheet_width - (2 * self.margin) - 50
        avail_height = self.sheet_height - (2 * self.margin) - 50
        
        # Calculate scale
        scale_x = avail_width / data_width
        scale_y = avail_height / data_height
        scale = min(scale_x, scale_y) * 0.9
        
        # Center offset
        scaled_width = data_width * scale
        scaled_height = data_height * scale
        offset_x = self.margin + 25 + (avail_width - scaled_width) / 2
        offset_y = self.margin + 25 + (avail_height - scaled_height) / 2
        
        tx = (x - min_x) * scale + offset_x
        ty = (y - min_y) * scale + offset_y
        
        return (tx, ty)
    
    def add_contours_to_dxf(self, contours: Dict, extents: Tuple):
        """Add contour lines to DXF."""
        if self.msp is None:
            raise ValueError("Document not created")
        
        # Track which levels are index contours
        index_levels = set(c['level'] for c in contours['index'])
        major_levels = set(c['level'] for c in contours['major'])
        
        # Add minor contours (skip if also major)
        for contour in contours['minor']:
            if contour['level'] in major_levels:
                continue
            vertices = [self._transform_point(v[0], v[1], extents) for v in contour['vertices']]
            if len(vertices) > 1:
                self.msp.add_lwpolyline(
                    vertices,
                    dxfattribs={'layer': 'CONTOUR_MINOR', 'lineweight': 13}
                )
        
        # Add major contours (skip if also index)
        for contour in contours['major']:
            if contour['level'] in index_levels:
                continue
            vertices = [self._transform_point(v[0], v[1], extents) for v in contour['vertices']]
            if len(vertices) > 1:
                self.msp.add_lwpolyline(
                    vertices,
                    dxfattribs={'layer': 'CONTOUR_MAJOR', 'lineweight': 25}
                )
        
        # Add index contours (thickest, with labels)
        for contour in contours['index']:
            vertices = [self._transform_point(v[0], v[1], extents) for v in contour['vertices']]
            if len(vertices) > 1:
                self.msp.add_lwpolyline(
                    vertices,
                    dxfattribs={'layer': 'CONTOUR_INDEX', 'lineweight': 50}
                )
                
                # Add elevation label at midpoint
                mid_idx = len(vertices) // 2
                mid_point = vertices[mid_idx]
                
                # Calculate rotation angle
                if mid_idx < len(vertices) - 1:
                    dx = vertices[mid_idx + 1][0] - vertices[mid_idx][0]
                    dy = vertices[mid_idx + 1][1] - vertices[mid_idx][1]
                    angle = math.degrees(math.atan2(dy, dx))
                else:
                    angle = 0
                
                self.msp.add_text(
                    f"{contour['level']:.1f}",
                    height=3.0,
                    rotation=angle,
                    dxfattribs={'layer': 'CONTOUR_LABELS'}
                ).set_placement(mid_point, align=TextEntityAlignment.MIDDLE_CENTER)
    
    def add_survey_points(self, points: List[Dict], extents: Tuple, show_labels: bool = False):
        """Add original survey points to DXF."""
        if self.msp is None:
            return
        
        for p in points:
            tx, ty = self._transform_point(p['x'], p['y'], extents)
            
            # Add point marker (small cross)
            size = 1.5
            self.msp.add_line((tx - size, ty), (tx + size, ty), dxfattribs={'layer': 'SURVEY_POINTS'})
            self.msp.add_line((tx, ty - size), (tx, ty + size), dxfattribs={'layer': 'SURVEY_POINTS'})
            
            if show_labels and 'id' in p:
                self.msp.add_text(
                    f"{p['id']}: {p['z']:.2f}",
                    height=2.0,
                    dxfattribs={'layer': 'SURVEY_POINTS'}
                ).set_placement((tx + 2, ty + 2), align=TextEntityAlignment.LEFT)
    
    def add_border(self):
        """Add drawing border."""
        if self.msp is None:
            return
        
        border_points = [
            (self.margin, self.margin),
            (self.sheet_width - self.margin, self.margin),
            (self.sheet_width - self.margin, self.sheet_height - self.margin),
            (self.margin, self.sheet_height - self.margin),
            (self.margin, self.margin)
        ]
        self.msp.add_lwpolyline(border_points, dxfattribs={'layer': 'BORDER', 'lineweight': 50})
    
    def add_legend(self, z_min: float, z_max: float):
        """Add contour legend."""
        if self.msp is None:
            return
        
        # Legend position (bottom-left)
        lx = self.margin + 10
        ly = self.margin + 10
        
        # Legend box
        box_width = 80
        box_height = 50
        self.msp.add_lwpolyline([
            (lx, ly), (lx + box_width, ly), 
            (lx + box_width, ly + box_height), (lx, ly + box_height), (lx, ly)
        ], dxfattribs={'layer': 'TITLE_BLOCK'})
        
        # Title
        self.msp.add_text(
            "CONTOUR LEGEND",
            height=3.5,
            dxfattribs={'layer': 'TITLE_BLOCK'}
        ).set_placement((lx + box_width/2, ly + box_height - 8), align=TextEntityAlignment.MIDDLE_CENTER)
        
        # Legend items
        y_pos = ly + box_height - 18
        
        # Index contour
        self.msp.add_line((lx + 5, y_pos), (lx + 20, y_pos), 
                         dxfattribs={'layer': 'CONTOUR_INDEX', 'lineweight': 50})
        self.msp.add_text(f"Index ({self.index_interval:.1f}m)", height=2.5,
                         dxfattribs={'layer': 'TITLE_BLOCK'}).set_placement((lx + 25, y_pos), align=TextEntityAlignment.LEFT)
        
        # Major contour
        y_pos -= 10
        self.msp.add_line((lx + 5, y_pos), (lx + 20, y_pos),
                         dxfattribs={'layer': 'CONTOUR_MAJOR', 'lineweight': 25})
        self.msp.add_text(f"Major ({self.major_interval:.1f}m)", height=2.5,
                         dxfattribs={'layer': 'TITLE_BLOCK'}).set_placement((lx + 25, y_pos), align=TextEntityAlignment.LEFT)
        
        # Minor contour
        y_pos -= 10
        self.msp.add_line((lx + 5, y_pos), (lx + 20, y_pos),
                         dxfattribs={'layer': 'CONTOUR_MINOR', 'lineweight': 13})
        self.msp.add_text(f"Minor ({self.minor_interval:.1f}m)", height=2.5,
                         dxfattribs={'layer': 'TITLE_BLOCK'}).set_placement((lx + 25, y_pos), align=TextEntityAlignment.LEFT)
        
        # Elevation range
        y_pos -= 10
        self.msp.add_text(f"Range: {z_min:.1f}m - {z_max:.1f}m", height=2.0,
                         dxfattribs={'layer': 'TITLE_BLOCK'}).set_placement((lx + 5, y_pos), align=TextEntityAlignment.LEFT)
    
    def add_north_arrow(self):
        """Add north arrow."""
        if self.msp is None:
            return
        
        cx = self.sheet_width - self.margin - 25
        cy = self.sheet_height - self.margin - 35
        size = 12
        
        arrow_points = [
            (cx, cy - size),
            (cx + size * 0.25, cy + size * 0.4),
            (cx, cy + size * 0.2),
            (cx - size * 0.25, cy + size * 0.4),
            (cx, cy - size)
        ]
        self.msp.add_lwpolyline(arrow_points, close=True, dxfattribs={'layer': 'NORTH_ARROW'})
        
        self.msp.add_text('N', height=6, dxfattribs={'layer': 'NORTH_ARROW'}).set_placement(
            (cx, cy + size + 4), align=TextEntityAlignment.MIDDLE_CENTER
        )
    
    def generate(self, points: List[Dict], show_points: bool = True, show_point_labels: bool = False) -> str:
        """
        Generate complete contour map.
        
        Args:
            points: List of dicts with 'x', 'y', 'z' (and optionally 'id')
            show_points: Whether to show original survey points
            show_point_labels: Whether to label survey points
            
        Returns:
            Statistics dict
        """
        if len(points) < 4:
            raise ValueError("Need at least 4 points to generate contours")
        
        # Calculate extents
        x_vals = [p['x'] for p in points]
        y_vals = [p['y'] for p in points]
        z_vals = [p['z'] for p in points]
        
        extents = (min(x_vals), min(y_vals), max(x_vals), max(y_vals))
        z_min, z_max = min(z_vals), max(z_vals)
        
        # Create document
        self.create_document()
        
        # Interpolate and extract contours
        X, Y, Z = self.interpolate_grid(points)
        contours = self.extract_contours(X, Y, Z)
        
        # Add elements to DXF
        self.add_border()
        self.add_contours_to_dxf(contours, extents)
        
        if show_points:
            self.add_survey_points(points, extents, show_point_labels)
        
        self.add_legend(z_min, z_max)
        self.add_north_arrow()
        
        # Return stats
        return {
            'num_points': len(points),
            'elevation_range': (z_min, z_max),
            'num_minor': len(contours['minor']),
            'num_major': len(contours['major']),
            'num_index': len(contours['index']),
            'extents': extents
        }
    
    def save(self, filepath: str) -> str:
        """Save DXF file."""
        if not self.doc:
            raise ValueError("No document to save")
        
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.saveas(str(output_path))
        return str(output_path.absolute())

