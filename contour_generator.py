"""
Contour Map Generator
Generates professional contour maps from survey elevation data.
Uses TIN-based interpolation for clean, accurate contours like professional CAD software.
"""

import numpy as np
from scipy import interpolate
from scipy.spatial import Delaunay, ConvexHull
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
    Uses TIN (Triangulated Irregular Network) approach for professional results.
    """
    
    # Sheet sizes in mm
    SHEET_SIZES = {
        'A1': (841, 594),
        'A2': (594, 420),
        'A3': (420, 297),
    }
    
    def __init__(self, 
                 major_interval: float = 1.0,
                 minor_interval: float = 0.5,
                 index_interval: float = 5.0,
                 grid_resolution: int = 150,
                 smoothing: float = 2.0,
                 sheet_size: str = 'A1',
                 grid_spacing: float = 10.0):
        """
        Initialize contour generator.
        
        Args:
            major_interval: Interval for major contour lines (meters)
            minor_interval: Interval for minor contour lines (meters)
            index_interval: Interval for index/labeled contours (meters)
            grid_resolution: Number of grid points for interpolation
            smoothing: Gaussian smoothing factor (higher = smoother)
            sheet_size: Output sheet size
            grid_spacing: Grid line spacing in meters (for overlay)
        """
        self.major_interval = major_interval
        self.minor_interval = minor_interval
        self.index_interval = index_interval
        self.grid_resolution = grid_resolution
        self.smoothing = smoothing
        self.sheet_size = sheet_size
        self.grid_spacing = grid_spacing
        
        self.sheet_width, self.sheet_height = self.SHEET_SIZES.get(sheet_size, (841, 594))
        self.margin = 20
        
        self.doc = None
        self.msp = None
        self.boundary_polygon = None
        self.extents = None
        self.triangulation = None
        
    def create_document(self):
        """Create DXF document with contour layers."""
        self.doc = ezdxf.new('R2010')
        self.doc.units = units.MM
        self.msp = self.doc.modelspace()
        
        # Create layers with appropriate colors
        self.doc.layers.add('BORDER', color=7)
        self.doc.layers.add('BOUNDARY', color=5)       # Blue - survey boundary
        self.doc.layers.add('GRID', color=8)            # Gray - grid lines
        self.doc.layers.add('CONTOUR_MAJOR', color=1)   # Red - major contours
        self.doc.layers.add('CONTOUR_MINOR', color=8)   # Gray - minor contours  
        self.doc.layers.add('CONTOUR_INDEX', color=3)   # Green - index contours
        self.doc.layers.add('CONTOUR_LABELS', color=1)  # Red - elevation labels
        self.doc.layers.add('SURVEY_POINTS', color=3)   # Green - survey points
        self.doc.layers.add('POINT_LABELS', color=1)    # Red - point labels
        self.doc.layers.add('TITLE_BLOCK', color=7)
        self.doc.layers.add('NORTH_ARROW', color=7)
        
        return self
    
    def compute_boundary(self, points: List[Dict]) -> List[Tuple[float, float]]:
        """Compute the convex hull boundary of survey points."""
        coords = np.array([[p['x'], p['y']] for p in points])
        
        if len(coords) < 3:
            return []
        
        try:
            hull = ConvexHull(coords)
            boundary = [tuple(coords[i]) for i in hull.vertices]
            return boundary
        except Exception:
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    
    def point_in_triangle(self, px: float, py: float, 
                          t1: Tuple, t2: Tuple, t3: Tuple) -> bool:
        """Check if point is inside triangle using barycentric coordinates."""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = sign((px, py), t1, t2)
        d2 = sign((px, py), t2, t3)
        d3 = sign((px, py), t3, t1)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)
    
    def interpolate_tin(self, points: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate using TIN (Triangulated Irregular Network).
        This produces cleaner contours similar to professional CAD software.
        """
        x = np.array([p['x'] for p in points])
        y = np.array([p['y'] for p in points])
        z = np.array([p['z'] for p in points])
        
        # Store for later use
        self.points_x = x
        self.points_y = y
        self.points_z = z
        
        # Create Delaunay triangulation
        coords = np.column_stack((x, y))
        self.triangulation = Delaunay(coords)
        
        # Create interpolation grid
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        # Add small buffer
        buffer = 0.02 * max(x_max - x_min, y_max - y_min)
        
        xi = np.linspace(x_min - buffer, x_max + buffer, self.grid_resolution)
        yi = np.linspace(y_min - buffer, y_max + buffer, self.grid_resolution)
        X, Y = np.meshgrid(xi, yi)
        
        # Use LinearNDInterpolator for TIN-based interpolation
        interp = interpolate.LinearNDInterpolator(coords, z)
        Z = interp(X, Y)
        
        # Apply Gaussian smoothing for cleaner contours
        if self.smoothing > 0:
            # Create mask for valid data
            valid_mask = ~np.isnan(Z)
            Z_filled = np.copy(Z)
            
            # Fill NaN with nearest values temporarily for smoothing
            if np.any(np.isnan(Z)):
                nearest_interp = interpolate.NearestNDInterpolator(coords, z)
                Z_filled = nearest_interp(X, Y)
            
            # Apply smoothing
            Z_smooth = gaussian_filter(Z_filled, sigma=self.smoothing)
            
            # Restore NaN outside triangulation
            Z_smooth[~valid_mask] = np.nan
            Z = Z_smooth
        
        return X, Y, Z
    
    def extract_contours(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> Dict:
        """Extract contour lines at specified intervals."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # Get elevation range from valid data
        z_valid = Z[~np.isnan(Z)]
        if len(z_valid) == 0:
            return {'major': [], 'minor': [], 'index': []}
        
        z_min, z_max = np.min(z_valid), np.max(z_valid)
        
        # Generate contour levels
        def get_levels(interval, z_min, z_max):
            start = np.floor(z_min / interval) * interval
            end = np.ceil(z_max / interval) * interval
            levels = np.arange(start, end + interval/2, interval)
            return levels[(levels >= z_min - interval) & (levels <= z_max + interval)]
        
        minor_levels = get_levels(self.minor_interval, z_min, z_max)
        major_levels = get_levels(self.major_interval, z_min, z_max)
        index_levels = get_levels(self.index_interval, z_min, z_max)
        
        # Create masked array
        Z_masked = np.ma.masked_invalid(Z)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        contours = {'minor': [], 'major': [], 'index': []}
        
        # Extract minor contours
        if len(minor_levels) > 0:
            try:
                cs = ax.contour(X, Y, Z_masked, levels=minor_levels)
                for i, level in enumerate(cs.levels):
                    if i < len(cs.allsegs):
                        for seg in cs.allsegs[i]:
                            if len(seg) >= 3:
                                # Simplify contour
                                simplified = self._douglas_peucker(seg.tolist(), 0.3)
                                if len(simplified) >= 2:
                                    contours['minor'].append({
                                        'level': float(level),
                                        'vertices': simplified
                                    })
            except Exception:
                pass
        
        # Extract major contours
        if len(major_levels) > 0:
            try:
                cs = ax.contour(X, Y, Z_masked, levels=major_levels)
                for i, level in enumerate(cs.levels):
                    if i < len(cs.allsegs):
                        for seg in cs.allsegs[i]:
                            if len(seg) >= 3:
                                simplified = self._douglas_peucker(seg.tolist(), 0.2)
                                if len(simplified) >= 2:
                                    contours['major'].append({
                                        'level': float(level),
                                        'vertices': simplified
                                    })
                                    
                                    # Check if index contour
                                    if any(abs(level - idx) < 0.01 for idx in index_levels):
                                        contours['index'].append({
                                            'level': float(level),
                                            'vertices': simplified
                                        })
            except Exception:
                pass
        
        plt.close(fig)
        
        return contours
    
    def _douglas_peucker(self, points: List, epsilon: float) -> List:
        """Douglas-Peucker line simplification algorithm."""
        if len(points) < 3:
            return points
        
        # Find point with maximum distance from line between first and last
        dmax = 0
        index = 0
        end = len(points) - 1
        
        for i in range(1, end):
            d = self._perpendicular_distance(points[i], points[0], points[end])
            if d > dmax:
                index = i
                dmax = d
        
        # If max distance > epsilon, recursively simplify
        if dmax > epsilon:
            rec1 = self._douglas_peucker(points[:index+1], epsilon)
            rec2 = self._douglas_peucker(points[index:], epsilon)
            return rec1[:-1] + rec2
        else:
            return [points[0], points[end]]
    
    def _perpendicular_distance(self, point, line_start, line_end):
        """Calculate perpendicular distance from point to line."""
        if line_start == line_end:
            return math.sqrt((point[0] - line_start[0])**2 + (point[1] - line_start[1])**2)
        
        n = abs((line_end[1] - line_start[1]) * point[0] - 
                (line_end[0] - line_start[0]) * point[1] + 
                line_end[0] * line_start[1] - 
                line_end[1] * line_start[0])
        d = math.sqrt((line_end[1] - line_start[1])**2 + (line_end[0] - line_start[0])**2)
        
        return n / d if d > 0 else 0
    
    def _transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """Transform real-world coordinates to sheet coordinates."""
        if not self.extents:
            return (x, y)
        
        min_x, min_y, max_x, max_y = self.extents
        
        data_width = max_x - min_x
        data_height = max_y - min_y
        
        if data_width == 0 or data_height == 0:
            return (self.sheet_width / 2, self.sheet_height / 2)
        
        # Available drawing area
        avail_width = self.sheet_width - (2 * self.margin) - 120
        avail_height = self.sheet_height - (2 * self.margin) - 80
        
        # Calculate scale
        scale_x = avail_width / data_width
        scale_y = avail_height / data_height
        scale = min(scale_x, scale_y) * 0.85
        
        # Store scale for grid
        self.scale = scale
        
        # Center offset
        scaled_width = data_width * scale
        scaled_height = data_height * scale
        offset_x = self.margin + 30 + (avail_width - scaled_width) / 2
        offset_y = self.margin + 60 + (avail_height - scaled_height) / 2
        
        self.offset_x = offset_x
        self.offset_y = offset_y
        
        tx = (x - min_x) * scale + offset_x
        ty = (y - min_y) * scale + offset_y
        
        return (tx, ty)
    
    def add_grid_overlay(self):
        """Add regular grid overlay like professional CAD software."""
        if self.msp is None or not self.extents:
            return
        
        min_x, min_y, max_x, max_y = self.extents
        
        # Round to grid spacing
        grid_min_x = math.floor(min_x / self.grid_spacing) * self.grid_spacing
        grid_max_x = math.ceil(max_x / self.grid_spacing) * self.grid_spacing
        grid_min_y = math.floor(min_y / self.grid_spacing) * self.grid_spacing
        grid_max_y = math.ceil(max_y / self.grid_spacing) * self.grid_spacing
        
        # Vertical grid lines
        x = grid_min_x
        while x <= grid_max_x:
            start = self._transform_point(x, min_y)
            end = self._transform_point(x, max_y)
            self.msp.add_line(start, end, dxfattribs={'layer': 'GRID', 'lineweight': 5})
            x += self.grid_spacing
        
        # Horizontal grid lines
        y = grid_min_y
        while y <= grid_max_y:
            start = self._transform_point(min_x, y)
            end = self._transform_point(max_x, y)
            self.msp.add_line(start, end, dxfattribs={'layer': 'GRID', 'lineweight': 5})
            y += self.grid_spacing
    
    def add_boundary(self):
        """Add survey boundary polygon (blue)."""
        if self.msp is None or not self.boundary_polygon:
            return
        
        vertices = [self._transform_point(x, y) for x, y in self.boundary_polygon]
        vertices.append(vertices[0])
        
        self.msp.add_lwpolyline(
            vertices,
            dxfattribs={'layer': 'BOUNDARY', 'lineweight': 50}
        )
    
    def add_contours_to_dxf(self, contours: Dict):
        """Add contour lines to DXF."""
        if self.msp is None:
            return
        
        index_levels = set(c['level'] for c in contours['index'])
        major_levels = set(c['level'] for c in contours['major'])
        
        # Minor contours (skip if also major)
        for contour in contours['minor']:
            if contour['level'] in major_levels:
                continue
            vertices = [self._transform_point(v[0], v[1]) for v in contour['vertices']]
            if len(vertices) >= 2:
                self.msp.add_lwpolyline(
                    vertices,
                    dxfattribs={'layer': 'CONTOUR_MINOR', 'lineweight': 9}
                )
        
        # Major contours (skip if also index)
        for contour in contours['major']:
            if contour['level'] in index_levels:
                continue
            vertices = [self._transform_point(v[0], v[1]) for v in contour['vertices']]
            if len(vertices) >= 2:
                self.msp.add_lwpolyline(
                    vertices,
                    dxfattribs={'layer': 'CONTOUR_MAJOR', 'lineweight': 18}
                )
        
        # Index contours (thickest, with labels)
        for contour in contours['index']:
            vertices = [self._transform_point(v[0], v[1]) for v in contour['vertices']]
            if len(vertices) >= 2:
                self.msp.add_lwpolyline(
                    vertices,
                    dxfattribs={'layer': 'CONTOUR_INDEX', 'lineweight': 35}
                )
                
                # Add label at midpoint for longer contours
                if len(vertices) >= 4:
                    mid_idx = len(vertices) // 2
                    mid_point = vertices[mid_idx]
                    
                    # Calculate angle
                    if mid_idx < len(vertices) - 1:
                        dx = vertices[mid_idx + 1][0] - vertices[mid_idx][0]
                        dy = vertices[mid_idx + 1][1] - vertices[mid_idx][1]
                        angle = math.degrees(math.atan2(dy, dx))
                        if angle > 90 or angle < -90:
                            angle += 180
                    else:
                        angle = 0
                    
                    self.msp.add_text(
                        f"{contour['level']:.2f}",
                        height=2.5,
                        rotation=angle,
                        dxfattribs={'layer': 'CONTOUR_LABELS'}
                    ).set_placement(mid_point, align=TextEntityAlignment.MIDDLE_CENTER)
    
    def add_survey_points(self, points: List[Dict], show_labels: bool = False):
        """Add survey points as GREEN FILLED CIRCLES (like reference)."""
        if self.msp is None:
            return
        
        # Calculate appropriate circle radius based on scale
        radius = 2.0  # Visible green circles
        
        for i, p in enumerate(points):
            tx, ty = self._transform_point(p['x'], p['y'])
            
            # Add filled circle (solid hatch would be ideal, but circle is visible)
            self.msp.add_circle(
                center=(tx, ty),
                radius=radius,
                dxfattribs={'layer': 'SURVEY_POINTS'}
            )
            
            # Add small cross inside for visibility
            cross_size = radius * 0.7
            self.msp.add_line(
                (tx - cross_size, ty), (tx + cross_size, ty),
                dxfattribs={'layer': 'SURVEY_POINTS'}
            )
            self.msp.add_line(
                (tx, ty - cross_size), (tx, ty + cross_size),
                dxfattribs={'layer': 'SURVEY_POINTS'}
            )
            
            # Add elevation label (sparse - every 10th point or if explicitly requested)
            if show_labels and i % 10 == 0:
                label = f"{p.get('id', '')}\n{p['z']:.2f}" if 'id' in p else f"{p['z']:.2f}"
                self.msp.add_text(
                    f"{p['z']:.2f}",
                    height=2.0,
                    dxfattribs={'layer': 'POINT_LABELS'}
                ).set_placement((tx + 3, ty + 2), align=TextEntityAlignment.LEFT)
    
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
        
        lx = self.sheet_width - self.margin - 100
        ly = self.margin + 10
        
        box_width = 90
        box_height = 70
        self.msp.add_lwpolyline([
            (lx, ly), (lx + box_width, ly), 
            (lx + box_width, ly + box_height), (lx, ly + box_height), (lx, ly)
        ], dxfattribs={'layer': 'TITLE_BLOCK'})
        
        self.msp.add_text(
            "LEGEND",
            height=3.5,
            dxfattribs={'layer': 'TITLE_BLOCK'}
        ).set_placement((lx + box_width/2, ly + box_height - 7), align=TextEntityAlignment.MIDDLE_CENTER)
        
        y_pos = ly + box_height - 18
        
        # Grid info
        self.msp.add_text(
            f"GRID: {self.grid_spacing:.0f}M x {self.grid_spacing:.0f}M",
            height=2.2,
            dxfattribs={'layer': 'TITLE_BLOCK'}
        ).set_placement((lx + 5, y_pos), align=TextEntityAlignment.LEFT)
        
        y_pos -= 10
        self.msp.add_text(
            f"LVL INTERVAL: {self.minor_interval*1000:.0f}MM",
            height=2.2,
            dxfattribs={'layer': 'TITLE_BLOCK'}
        ).set_placement((lx + 5, y_pos), align=TextEntityAlignment.LEFT)
        
        # Survey point
        y_pos -= 12
        self.msp.add_circle((lx + 10, y_pos + 3), 2.0, dxfattribs={'layer': 'SURVEY_POINTS'})
        self.msp.add_text("Survey Point", height=2.2,
                         dxfattribs={'layer': 'TITLE_BLOCK'}).set_placement((lx + 18, y_pos + 3), align=TextEntityAlignment.LEFT)
        
        # Elevation range
        y_pos -= 12
        self.msp.add_text(
            f"Elev: {z_min:.2f} - {z_max:.2f}m",
            height=2.0,
            dxfattribs={'layer': 'TITLE_BLOCK'}
        ).set_placement((lx + 5, y_pos), align=TextEntityAlignment.LEFT)
    
    def add_north_arrow(self):
        """Add north arrow."""
        if self.msp is None:
            return
        
        cx = self.sheet_width - self.margin - 30
        cy = self.sheet_height - self.margin - 40
        size = 15
        
        arrow_points = [
            (cx, cy - size),
            (cx + size * 0.3, cy + size * 0.5),
            (cx, cy + size * 0.25),
            (cx - size * 0.3, cy + size * 0.5),
            (cx, cy - size)
        ]
        self.msp.add_lwpolyline(arrow_points, close=True, dxfattribs={'layer': 'NORTH_ARROW'})
        
        self.msp.add_text('N', height=8, dxfattribs={'layer': 'NORTH_ARROW'}).set_placement(
            (cx, cy + size + 6), align=TextEntityAlignment.MIDDLE_CENTER
        )
    
    def add_coordinate_grid_with_labels(self):
        """Add coordinate grid with labeled axes (like professional CAD)."""
        if self.msp is None or not self.extents:
            return
        
        min_x, min_y, max_x, max_y = self.extents
        
        # Round to grid spacing for clean numbers
        grid_min_x = math.floor(min_x / self.grid_spacing) * self.grid_spacing
        grid_max_x = math.ceil(max_x / self.grid_spacing) * self.grid_spacing
        grid_min_y = math.floor(min_y / self.grid_spacing) * self.grid_spacing
        grid_max_y = math.ceil(max_y / self.grid_spacing) * self.grid_spacing
        
        # Vertical grid lines with X-coordinate labels at top
        x = grid_min_x
        while x <= grid_max_x:
            start = self._transform_point(x, min_y)
            end = self._transform_point(x, max_y)
            self.msp.add_line(start, end, dxfattribs={'layer': 'GRID', 'lineweight': 5})
            
            # Add coordinate label at top
            label_pos = self._transform_point(x, max_y)
            self.msp.add_text(
                f"{x:.0f}",
                height=2.0,
                rotation=90,
                dxfattribs={'layer': 'GRID'}
            ).set_placement((label_pos[0], label_pos[1] + 3), align=TextEntityAlignment.BOTTOM_CENTER)
            
            x += self.grid_spacing
        
        # Horizontal grid lines with Y-coordinate labels on left
        y = grid_min_y
        while y <= grid_max_y:
            start = self._transform_point(min_x, y)
            end = self._transform_point(max_x, y)
            self.msp.add_line(start, end, dxfattribs={'layer': 'GRID', 'lineweight': 5})
            
            # Add coordinate label on left
            label_pos = self._transform_point(min_x, y)
            self.msp.add_text(
                f"{y:.0f}",
                height=2.0,
                dxfattribs={'layer': 'GRID'}
            ).set_placement((label_pos[0] - 3, label_pos[1]), align=TextEntityAlignment.RIGHT)
            
            y += self.grid_spacing
    
    def add_all_elevation_labels(self, points: List[Dict]):
        """Add RED elevation labels at EVERY survey point (like reference)."""
        if self.msp is None:
            return
        
        for p in points:
            tx, ty = self._transform_point(p['x'], p['y'])
            
            # Add elevation label (RED) offset from point
            self.msp.add_text(
                f"{p['z']:.2f}",
                height=1.8,
                dxfattribs={'layer': 'POINT_LABELS'}
            ).set_placement((tx + 3, ty - 1), align=TextEntityAlignment.LEFT)
    
    def add_spot_elevation_legend(self, z_min: float, z_max: float, num_points: int):
        """Add legend for spot elevation grid style."""
        if self.msp is None:
            return
        
        lx = self.sheet_width - self.margin - 110
        ly = self.margin + 10
        
        box_width = 100
        box_height = 55
        self.msp.add_lwpolyline([
            (lx, ly), (lx + box_width, ly), 
            (lx + box_width, ly + box_height), (lx, ly + box_height), (lx, ly)
        ], dxfattribs={'layer': 'TITLE_BLOCK'})
        
        y_pos = ly + box_height - 10
        
        # Grid info
        self.msp.add_text(
            f"GRID: {self.grid_spacing:.0f}M x {self.grid_spacing:.0f}M",
            height=2.5,
            dxfattribs={'layer': 'TITLE_BLOCK'}
        ).set_placement((lx + 5, y_pos), align=TextEntityAlignment.LEFT)
        
        y_pos -= 10
        self.msp.add_text(
            f"LVL INTERVAL: {self.minor_interval*1000:.0f}MM",
            height=2.5,
            dxfattribs={'layer': 'TITLE_BLOCK'}
        ).set_placement((lx + 5, y_pos), align=TextEntityAlignment.LEFT)
        
        # Survey point symbol
        y_pos -= 12
        self.msp.add_circle((lx + 8, y_pos + 2), 2.5, dxfattribs={'layer': 'SURVEY_POINTS'})
        self.msp.add_text("Survey Point", height=2.2,
                         dxfattribs={'layer': 'TITLE_BLOCK'}).set_placement((lx + 15, y_pos + 2), align=TextEntityAlignment.LEFT)
        
        # Stats
        y_pos -= 12
        self.msp.add_text(
            f"Points: {num_points} | Elev: {z_min:.1f}-{z_max:.1f}m",
            height=1.8,
            dxfattribs={'layer': 'TITLE_BLOCK'}
        ).set_placement((lx + 5, y_pos), align=TextEntityAlignment.LEFT)
    
    def generate_spot_elevation_grid(self, points: List[Dict]) -> Dict:
        """
        Generate SPOT ELEVATION GRID style map (like reference).
        Shows: coordinate grid, survey points as green circles, elevation labels at each point.
        NO contour lines.
        """
        if len(points) < 3:
            raise ValueError("Need at least 3 points")
        
        # Calculate extents
        x_vals = [p['x'] for p in points]
        y_vals = [p['y'] for p in points]
        z_vals = [p['z'] for p in points]
        
        self.extents = (min(x_vals), min(y_vals), max(x_vals), max(y_vals))
        z_min, z_max = min(z_vals), max(z_vals)
        
        # Compute boundary
        self.boundary_polygon = self.compute_boundary(points)
        
        # Create document
        self.create_document()
        
        # Add elements (order matters for layering)
        self.add_border()
        self.add_coordinate_grid_with_labels()  # Grid with coordinate labels
        self.add_boundary()  # Blue boundary polygon
        
        # Add survey points as green circles with crosses
        for p in points:
            tx, ty = self._transform_point(p['x'], p['y'])
            
            # Green filled circle
            radius = 2.5
            self.msp.add_circle(
                center=(tx, ty),
                radius=radius,
                dxfattribs={'layer': 'SURVEY_POINTS'}
            )
            
            # Cross inside circle
            cross_size = radius * 0.6
            self.msp.add_line(
                (tx - cross_size, ty), (tx + cross_size, ty),
                dxfattribs={'layer': 'SURVEY_POINTS'}
            )
            self.msp.add_line(
                (tx, ty - cross_size), (tx, ty + cross_size),
                dxfattribs={'layer': 'SURVEY_POINTS'}
            )
        
        # Add elevation labels at EVERY point (RED)
        self.add_all_elevation_labels(points)
        
        self.add_spot_elevation_legend(z_min, z_max, len(points))
        self.add_north_arrow()
        
        return {
            'num_points': len(points),
            'elevation_range': (z_min, z_max),
            'num_minor': 0,
            'num_major': 0,
            'num_index': 0,
            'extents': self.extents,
            'style': 'spot_elevation'
        }
    
    def generate(self, points: List[Dict], show_points: bool = True, 
                 show_point_labels: bool = False, show_grid: bool = True,
                 style: str = 'contour') -> Dict:
        """
        Generate contour or spot elevation map.
        
        Args:
            points: List of dicts with 'x', 'y', 'z' (and optionally 'id')
            show_points: Whether to show survey points
            show_point_labels: Whether to label survey points
            show_grid: Whether to show grid overlay
            style: 'contour' for traditional contour lines, 'spot_elevation' for grid+points+labels
        """
        # If spot elevation style requested, use that generator
        if style == 'spot_elevation':
            return self.generate_spot_elevation_grid(points)
        
        if len(points) < 4:
            raise ValueError("Need at least 4 points to generate contours")
        
        # Calculate extents
        x_vals = [p['x'] for p in points]
        y_vals = [p['y'] for p in points]
        z_vals = [p['z'] for p in points]
        
        self.extents = (min(x_vals), min(y_vals), max(x_vals), max(y_vals))
        z_min, z_max = min(z_vals), max(z_vals)
        
        # Compute boundary
        self.boundary_polygon = self.compute_boundary(points)
        
        # Create document
        self.create_document()
        
        # TIN-based interpolation
        X, Y, Z = self.interpolate_tin(points)
        contours = self.extract_contours(X, Y, Z)
        
        # Add elements (order matters for layering)
        self.add_border()
        
        if show_grid:
            self.add_coordinate_grid_with_labels()
        
        self.add_boundary()
        self.add_contours_to_dxf(contours)
        
        if show_points:
            self.add_survey_points(points, show_labels=show_point_labels)
        
        self.add_legend(z_min, z_max)
        self.add_north_arrow()
        
        return {
            'num_points': len(points),
            'elevation_range': (z_min, z_max),
            'num_minor': len(contours['minor']),
            'num_major': len(contours['major']),
            'num_index': len(contours['index']),
            'extents': self.extents,
            'style': 'contour'
        }
    
    def save(self, filepath: str) -> str:
        """Save DXF file."""
        if not self.doc:
            raise ValueError("No document to save")
        
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.saveas(str(output_path))
        return str(output_path.absolute())
