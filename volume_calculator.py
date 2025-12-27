"""
Volume Calculator for Cut/Fill Analysis
Calculates earthwork volumes between existing and proposed surfaces.
"""

import numpy as np
from scipy import interpolate
import ezdxf
from ezdxf import units
from ezdxf.enums import TextEntityAlignment
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import math


class VolumeCalculator:
    """
    Calculate cut/fill volumes between two surfaces.
    Supports grid-based and TIN-based volume calculations.
    """
    
    # Sheet sizes in mm
    SHEET_SIZES = {
        'A1': (841, 594),
        'A2': (594, 420),
        'A3': (420, 297),
    }
    
    def __init__(self, 
                 grid_resolution: int = 50,
                 sheet_size: str = 'A1'):
        """
        Initialize volume calculator.
        
        Args:
            grid_resolution: Number of grid cells for volume calculation
            sheet_size: Output sheet size
        """
        self.grid_resolution = grid_resolution
        self.sheet_size = sheet_size
        self.sheet_width, self.sheet_height = self.SHEET_SIZES.get(sheet_size, (841, 594))
        self.margin = 20
        
        self.doc = None
        self.msp = None
        
        # Results storage
        self.cut_volume = 0.0
        self.fill_volume = 0.0
        self.net_volume = 0.0
        self.cut_area = 0.0
        self.fill_area = 0.0
        self.grid_data = None
    
    def create_document(self):
        """Create DXF document with volume layers."""
        self.doc = ezdxf.new('R2010')
        self.doc.units = units.MM
        self.msp = self.doc.modelspace()
        
        # Create layers
        self.doc.layers.add('BORDER', color=7)
        self.doc.layers.add('CUT_ZONE', color=1)      # Red for cut
        self.doc.layers.add('FILL_ZONE', color=3)     # Green for fill
        self.doc.layers.add('NEUTRAL_ZONE', color=8)  # Gray for no change
        self.doc.layers.add('GRID_LINES', color=8)    # Gray grid
        self.doc.layers.add('LABELS', color=7)
        self.doc.layers.add('LEGEND', color=7)
        self.doc.layers.add('TITLE_BLOCK', color=7)
        
        return self
    
    def interpolate_surface(self, points: List[Dict], X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Interpolate scattered points to a regular grid.
        
        Args:
            points: List of dicts with 'x', 'y', 'z' keys
            X, Y: Meshgrid arrays
            
        Returns:
            Z grid array
        """
        x = np.array([p['x'] for p in points])
        y = np.array([p['y'] for p in points])
        z = np.array([p['z'] for p in points])
        
        # Interpolate using linear method (more robust)
        try:
            Z = interpolate.griddata((x, y), z, (X, Y), method='cubic')
        except:
            Z = interpolate.griddata((x, y), z, (X, Y), method='linear')
        
        # Fill NaN values with nearest neighbor
        mask = np.isnan(Z)
        if mask.any():
            Z_nearest = interpolate.griddata((x, y), z, (X, Y), method='nearest')
            Z[mask] = Z_nearest[mask]
        
        return Z
    
    def calculate_volumes(self, 
                         existing_points: List[Dict],
                         proposed_level: float = None,
                         proposed_points: List[Dict] = None) -> Dict:
        """
        Calculate cut and fill volumes.
        
        Args:
            existing_points: List of existing ground points {'x', 'y', 'z'}
            proposed_level: Single proposed level (for flat grading)
            proposed_points: List of proposed surface points (for complex grading)
            
        Returns:
            Dict with volume statistics
        """
        if len(existing_points) < 4:
            raise ValueError("Need at least 4 points for volume calculation")
        
        # Get extents
        x_vals = [p['x'] for p in existing_points]
        y_vals = [p['y'] for p in existing_points]
        
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        
        # Create grid
        xi = np.linspace(x_min, x_max, self.grid_resolution)
        yi = np.linspace(y_min, y_max, self.grid_resolution)
        X, Y = np.meshgrid(xi, yi)
        
        # Cell size
        cell_width = (x_max - x_min) / (self.grid_resolution - 1)
        cell_height = (y_max - y_min) / (self.grid_resolution - 1)
        cell_area = cell_width * cell_height
        
        # Interpolate existing surface
        Z_existing = self.interpolate_surface(existing_points, X, Y)
        
        # Get proposed surface
        if proposed_level is not None:
            Z_proposed = np.full_like(Z_existing, proposed_level)
        elif proposed_points is not None:
            Z_proposed = self.interpolate_surface(proposed_points, X, Y)
        else:
            # Default: use mean level
            proposed_level = np.nanmean(Z_existing)
            Z_proposed = np.full_like(Z_existing, proposed_level)
        
        # Calculate difference (positive = cut, negative = fill)
        Z_diff = Z_existing - Z_proposed
        
        # Calculate volumes using prismoidal method
        cut_volume = 0.0
        fill_volume = 0.0
        cut_cells = 0
        fill_cells = 0
        
        for i in range(Z_diff.shape[0]):
            for j in range(Z_diff.shape[1]):
                diff = Z_diff[i, j]
                if not np.isnan(diff):
                    volume = abs(diff) * cell_area
                    if diff > 0:
                        cut_volume += volume
                        cut_cells += 1
                    elif diff < 0:
                        fill_volume += volume
                        fill_cells += 1
        
        # Store results
        self.cut_volume = cut_volume
        self.fill_volume = fill_volume
        self.net_volume = cut_volume - fill_volume
        self.cut_area = cut_cells * cell_area
        self.fill_area = fill_cells * cell_area
        
        # Store grid data for visualization
        self.grid_data = {
            'X': X,
            'Y': Y,
            'Z_existing': Z_existing,
            'Z_proposed': Z_proposed,
            'Z_diff': Z_diff,
            'extents': (x_min, y_min, x_max, y_max),
            'cell_width': cell_width,
            'cell_height': cell_height
        }
        
        return {
            'cut_volume': cut_volume,
            'fill_volume': fill_volume,
            'net_volume': self.net_volume,
            'cut_area': self.cut_area,
            'fill_area': self.fill_area,
            'total_area': (x_max - x_min) * (y_max - y_min),
            'existing_elevation_range': (np.nanmin(Z_existing), np.nanmax(Z_existing)),
            'proposed_level': proposed_level if proposed_level else np.nanmean(Z_proposed)
        }
    
    def _transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """Transform real-world coordinates to sheet coordinates."""
        if not self.grid_data:
            return (x, y)
        
        extents = self.grid_data['extents']
        min_x, min_y, max_x, max_y = extents
        
        data_width = max_x - min_x
        data_height = max_y - min_y
        
        if data_width == 0 or data_height == 0:
            return (self.sheet_width / 2, self.sheet_height / 2)
        
        # Available drawing area (leave space for legend)
        avail_width = self.sheet_width - (2 * self.margin) - 150
        avail_height = self.sheet_height - (2 * self.margin) - 80
        
        # Calculate scale
        scale_x = avail_width / data_width
        scale_y = avail_height / data_height
        scale = min(scale_x, scale_y) * 0.85
        
        # Offset
        scaled_width = data_width * scale
        scaled_height = data_height * scale
        offset_x = self.margin + 30 + (avail_width - scaled_width) / 2
        offset_y = self.margin + 60 + (avail_height - scaled_height) / 2
        
        tx = (x - min_x) * scale + offset_x
        ty = (y - min_y) * scale + offset_y
        
        return (tx, ty)
    
    def add_cut_fill_zones(self):
        """Add cut/fill zone visualization to DXF."""
        if self.msp is None or self.grid_data is None:
            return
        
        X = self.grid_data['X']
        Y = self.grid_data['Y']
        Z_diff = self.grid_data['Z_diff']
        cell_w = self.grid_data['cell_width']
        cell_h = self.grid_data['cell_height']
        
        # Draw each cell as a rectangle with appropriate color
        for i in range(Z_diff.shape[0] - 1):
            for j in range(Z_diff.shape[1] - 1):
                diff = Z_diff[i, j]
                if np.isnan(diff):
                    continue
                
                # Get cell corners
                x = X[i, j]
                y = Y[i, j]
                
                corners = [
                    self._transform_point(x, y),
                    self._transform_point(x + cell_w, y),
                    self._transform_point(x + cell_w, y + cell_h),
                    self._transform_point(x, y + cell_h),
                    self._transform_point(x, y)
                ]
                
                # Determine layer based on cut/fill
                if diff > 0.1:
                    layer = 'CUT_ZONE'
                elif diff < -0.1:
                    layer = 'FILL_ZONE'
                else:
                    layer = 'NEUTRAL_ZONE'
                
                self.msp.add_lwpolyline(corners, dxfattribs={'layer': layer})
    
    def add_grid_lines(self):
        """Add grid lines to DXF."""
        if self.msp is None or self.grid_data is None:
            return
        
        X = self.grid_data['X']
        Y = self.grid_data['Y']
        
        # Horizontal lines
        for i in range(0, X.shape[0], max(1, X.shape[0] // 10)):
            points = [self._transform_point(X[i, j], Y[i, j]) for j in range(X.shape[1])]
            self.msp.add_lwpolyline(points, dxfattribs={'layer': 'GRID_LINES', 'lineweight': 5})
        
        # Vertical lines
        for j in range(0, X.shape[1], max(1, X.shape[1] // 10)):
            points = [self._transform_point(X[i, j], Y[i, j]) for i in range(X.shape[0])]
            self.msp.add_lwpolyline(points, dxfattribs={'layer': 'GRID_LINES', 'lineweight': 5})
    
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
    
    def add_volume_summary(self):
        """Add volume summary table."""
        if self.msp is None:
            return
        
        # Summary position (right side)
        sx = self.sheet_width - self.margin - 140
        sy = self.sheet_height - self.margin - 20
        
        # Title
        self.msp.add_text(
            "VOLUME SUMMARY",
            height=5,
            dxfattribs={'layer': 'TITLE_BLOCK'}
        ).set_placement((sx + 65, sy), align=TextEntityAlignment.MIDDLE_CENTER)
        
        # Box
        box_height = 100
        box_width = 130
        sy -= 10
        self.msp.add_lwpolyline([
            (sx, sy), (sx + box_width, sy),
            (sx + box_width, sy - box_height), (sx, sy - box_height), (sx, sy)
        ], dxfattribs={'layer': 'TITLE_BLOCK'})
        
        # Content
        y_pos = sy - 15
        line_height = 12
        
        items = [
            ("CUT VOLUME:", f"{self.cut_volume:,.2f} m³"),
            ("FILL VOLUME:", f"{self.fill_volume:,.2f} m³"),
            ("NET VOLUME:", f"{self.net_volume:,.2f} m³"),
            ("CUT AREA:", f"{self.cut_area:,.2f} m²"),
            ("FILL AREA:", f"{self.fill_area:,.2f} m²"),
        ]
        
        for label, value in items:
            self.msp.add_text(
                label,
                height=3,
                dxfattribs={'layer': 'LABELS'}
            ).set_placement((sx + 5, y_pos), align=TextEntityAlignment.LEFT)
            
            self.msp.add_text(
                value,
                height=3,
                dxfattribs={'layer': 'LABELS'}
            ).set_placement((sx + box_width - 5, y_pos), align=TextEntityAlignment.RIGHT)
            
            y_pos -= line_height
        
        # Net volume interpretation
        y_pos -= 10
        if self.net_volume > 0:
            interpretation = "EXCESS CUT (export)"
        elif self.net_volume < 0:
            interpretation = "EXCESS FILL (import)"
        else:
            interpretation = "BALANCED"
        
        self.msp.add_text(
            interpretation,
            height=3.5,
            dxfattribs={'layer': 'LABELS'}
        ).set_placement((sx + 65, y_pos), align=TextEntityAlignment.MIDDLE_CENTER)
    
    def add_legend(self):
        """Add color legend."""
        if self.msp is None:
            return
        
        lx = self.sheet_width - self.margin - 140
        ly = self.margin + 10
        
        # Legend box
        box_width = 130
        box_height = 45
        self.msp.add_lwpolyline([
            (lx, ly), (lx + box_width, ly),
            (lx + box_width, ly + box_height), (lx, ly + box_height), (lx, ly)
        ], dxfattribs={'layer': 'LEGEND'})
        
        # Title
        self.msp.add_text(
            "LEGEND",
            height=4,
            dxfattribs={'layer': 'LEGEND'}
        ).set_placement((lx + box_width/2, ly + box_height - 8), align=TextEntityAlignment.MIDDLE_CENTER)
        
        # Cut zone
        y_pos = ly + box_height - 20
        self.msp.add_lwpolyline([
            (lx + 5, y_pos), (lx + 20, y_pos), (lx + 20, y_pos + 6), (lx + 5, y_pos + 6), (lx + 5, y_pos)
        ], dxfattribs={'layer': 'CUT_ZONE'})
        self.msp.add_text("CUT (Remove)", height=3, dxfattribs={'layer': 'LEGEND'}).set_placement(
            (lx + 25, y_pos + 3), align=TextEntityAlignment.LEFT)
        
        # Fill zone
        y_pos -= 12
        self.msp.add_lwpolyline([
            (lx + 5, y_pos), (lx + 20, y_pos), (lx + 20, y_pos + 6), (lx + 5, y_pos + 6), (lx + 5, y_pos)
        ], dxfattribs={'layer': 'FILL_ZONE'})
        self.msp.add_text("FILL (Add)", height=3, dxfattribs={'layer': 'LEGEND'}).set_placement(
            (lx + 25, y_pos + 3), align=TextEntityAlignment.LEFT)
    
    def generate(self, 
                existing_points: List[Dict],
                proposed_level: float = None,
                proposed_points: List[Dict] = None,
                show_grid: bool = True) -> Dict:
        """
        Generate complete volume analysis drawing.
        
        Args:
            existing_points: List of existing ground points
            proposed_level: Proposed flat level (optional)
            proposed_points: Proposed surface points (optional)
            show_grid: Whether to show grid lines
            
        Returns:
            Volume statistics dict
        """
        # Calculate volumes
        stats = self.calculate_volumes(existing_points, proposed_level, proposed_points)
        
        # Create document
        self.create_document()
        
        # Add elements
        self.add_border()
        self.add_cut_fill_zones()
        
        if show_grid:
            self.add_grid_lines()
        
        self.add_volume_summary()
        self.add_legend()
        
        return stats
    
    def save(self, filepath: str) -> str:
        """Save DXF file."""
        if not self.doc:
            raise ValueError("No document to save")
        
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.saveas(str(output_path))
        return str(output_path.absolute())


