"""
Land Subdivision Planner
Automatically subdivides land parcels into plots based on rules.
"""

import numpy as np
import math
import ezdxf
from ezdxf import units
from ezdxf.enums import TextEntityAlignment
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy.spatial import ConvexHull


class SubdivisionPlanner:
    """
    Automatically subdivide a land parcel into plots.
    Supports various subdivision schemes and road layouts.
    """
    
    # Sheet sizes in mm
    SHEET_SIZES = {
        'A1': (841, 594),
        'A2': (594, 420),
        'A3': (420, 297),
    }
    
    def __init__(self, 
                 sheet_size: str = 'A1',
                 road_width: float = 6.0,
                 min_plot_area: float = 100.0,
                 min_frontage: float = 6.0):
        """
        Initialize subdivision planner.
        
        Args:
            sheet_size: Output sheet size
            road_width: Road width in meters
            min_plot_area: Minimum plot area in square meters
            min_frontage: Minimum plot frontage in meters
        """
        self.sheet_size = sheet_size
        self.road_width = road_width
        self.min_plot_area = min_plot_area
        self.min_frontage = min_frontage
        
        self.sheet_width, self.sheet_height = self.SHEET_SIZES.get(sheet_size, (841, 594))
        self.margin = 20
        
        self.doc = None
        self.msp = None
        
        self.plots = []
        self.roads = []
        self.boundary = []
        self.extents = None
    
    def create_document(self):
        """Create DXF document with subdivision layers."""
        self.doc = ezdxf.new('R2010')
        self.doc.units = units.MM
        self.msp = self.doc.modelspace()
        
        # Create layers
        self.doc.layers.add('BORDER', color=7)
        self.doc.layers.add('BOUNDARY', color=2)       # Yellow - outer boundary
        self.doc.layers.add('PLOTS', color=3)          # Green - plot boundaries
        self.doc.layers.add('ROADS', color=8)          # Gray - roads
        self.doc.layers.add('PLOT_NUMBERS', color=4)   # Cyan - plot numbers
        self.doc.layers.add('DIMENSIONS', color=5)     # Blue - dimensions
        self.doc.layers.add('LABELS', color=7)
        self.doc.layers.add('TITLE_BLOCK', color=7)
        self.doc.layers.add('NORTH_ARROW', color=7)
        
        return self
    
    def calculate_polygon_area(self, coords: List[Tuple[float, float]]) -> float:
        """Calculate polygon area using Shoelace formula."""
        n = len(coords)
        if n < 3:
            return 0.0
        
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += coords[i][0] * coords[j][1]
            area -= coords[j][0] * coords[i][1]
        
        return abs(area) / 2.0
    
    def get_bounding_box(self, coords: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        """Get bounding box of polygon."""
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        return (min(xs), min(ys), max(xs), max(ys))
    
    def subdivide_rectangular(self, 
                              boundary: List[Tuple[float, float]],
                              num_plots: int = None,
                              plot_width: float = None,
                              plot_depth: float = None,
                              road_pattern: str = 'single') -> List[Dict]:
        """
        Subdivide into rectangular plots.
        
        Args:
            boundary: List of boundary coordinates
            num_plots: Target number of plots (if specified)
            plot_width: Target plot width (if specified)
            plot_depth: Target plot depth (if specified)
            road_pattern: 'single', 'double', or 'grid'
            
        Returns:
            List of plot dictionaries
        """
        self.boundary = boundary
        self.extents = self.get_bounding_box(boundary)
        min_x, min_y, max_x, max_y = self.extents
        
        total_width = max_x - min_x
        total_depth = max_y - min_y
        
        # Calculate available area after roads
        if road_pattern == 'single':
            # Single road in the middle
            available_width = total_width
            available_depth = (total_depth - self.road_width) / 2
        elif road_pattern == 'double':
            # Road on both ends
            available_width = total_width - 2 * self.road_width
            available_depth = total_depth
        else:  # grid
            available_width = total_width - self.road_width
            available_depth = total_depth - self.road_width
        
        # Determine plot dimensions
        if plot_width and plot_depth:
            pw, pd = plot_width, plot_depth
        elif num_plots:
            # Estimate dimensions based on number of plots
            aspect_ratio = available_width / available_depth
            plots_per_row = int(math.sqrt(num_plots * aspect_ratio))
            plots_per_col = int(math.ceil(num_plots / plots_per_row))
            pw = available_width / plots_per_row
            pd = available_depth / plots_per_col
        else:
            # Default: aim for ~200 sqm plots
            target_area = 200
            pw = max(self.min_frontage, 10)
            pd = target_area / pw
        
        # Ensure minimum dimensions
        pw = max(pw, self.min_frontage)
        pd = max(pd, self.min_frontage)
        
        # Generate plots
        plots = []
        plot_num = 1
        
        if road_pattern == 'single':
            # Create central road
            road_y = min_y + total_depth / 2 - self.road_width / 2
            self.roads.append({
                'type': 'horizontal',
                'coords': [
                    (min_x, road_y),
                    (max_x, road_y),
                    (max_x, road_y + self.road_width),
                    (min_x, road_y + self.road_width)
                ]
            })
            
            # Plots on south side of road
            y_start = min_y
            y_end = road_y
            num_rows = int((y_end - y_start) / pd)
            num_cols = int(total_width / pw)
            
            for row in range(num_rows):
                for col in range(num_cols):
                    x1 = min_x + col * pw
                    y1 = y_start + row * pd
                    x2 = x1 + pw
                    y2 = y1 + pd
                    
                    if y2 <= y_end:
                        plots.append({
                            'number': plot_num,
                            'coords': [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                            'area': pw * pd,
                            'frontage': pw
                        })
                        plot_num += 1
            
            # Plots on north side of road
            y_start = road_y + self.road_width
            y_end = max_y
            num_rows = int((y_end - y_start) / pd)
            
            for row in range(num_rows):
                for col in range(num_cols):
                    x1 = min_x + col * pw
                    y1 = y_start + row * pd
                    x2 = x1 + pw
                    y2 = y1 + pd
                    
                    if y2 <= y_end:
                        plots.append({
                            'number': plot_num,
                            'coords': [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                            'area': pw * pd,
                            'frontage': pw
                        })
                        plot_num += 1
        
        else:
            # Simple grid layout without central road
            num_cols = int(available_width / pw)
            num_rows = int(available_depth / pd)
            
            for row in range(num_rows):
                for col in range(num_cols):
                    x1 = min_x + col * pw
                    y1 = min_y + row * pd
                    x2 = x1 + pw
                    y2 = y1 + pd
                    
                    plots.append({
                        'number': plot_num,
                        'coords': [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                        'area': pw * pd,
                        'frontage': pw
                    })
                    plot_num += 1
        
        self.plots = plots
        return plots
    
    def subdivide_by_area(self,
                          boundary: List[Tuple[float, float]],
                          target_areas: List[float]) -> List[Dict]:
        """
        Subdivide into plots of specific areas.
        
        Args:
            boundary: List of boundary coordinates
            target_areas: List of target areas for each plot
            
        Returns:
            List of plot dictionaries
        """
        self.boundary = boundary
        self.extents = self.get_bounding_box(boundary)
        min_x, min_y, max_x, max_y = self.extents
        
        total_width = max_x - min_x
        total_height = max_y - min_y
        total_area = self.calculate_polygon_area(boundary)
        
        # Simple strip subdivision
        plots = []
        current_x = min_x
        
        for i, target_area in enumerate(target_areas):
            if current_x >= max_x:
                break
            
            # Calculate strip width for this area
            strip_width = target_area / total_height
            
            # Ensure minimum frontage
            strip_width = max(strip_width, self.min_frontage)
            
            x1 = current_x
            x2 = min(current_x + strip_width, max_x)
            
            actual_area = (x2 - x1) * total_height
            
            plots.append({
                'number': i + 1,
                'coords': [(x1, min_y), (x2, min_y), (x2, max_y), (x1, max_y)],
                'area': actual_area,
                'frontage': x2 - x1
            })
            
            current_x = x2
        
        self.plots = plots
        return plots
    
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
        avail_width = self.sheet_width - (2 * self.margin) - 180
        avail_height = self.sheet_height - (2 * self.margin) - 60
        
        # Calculate scale
        scale_x = avail_width / data_width
        scale_y = avail_height / data_height
        scale = min(scale_x, scale_y) * 0.85
        
        # Offset
        scaled_width = data_width * scale
        scaled_height = data_height * scale
        offset_x = self.margin + 30 + (avail_width - scaled_width) / 2
        offset_y = self.margin + 50 + (avail_height - scaled_height) / 2
        
        tx = (x - min_x) * scale + offset_x
        ty = (y - min_y) * scale + offset_y
        
        return (tx, ty)
    
    def add_boundary(self):
        """Add outer boundary to DXF."""
        if self.msp is None or not self.boundary:
            return
        
        coords = [self._transform_point(x, y) for x, y in self.boundary]
        coords.append(coords[0])  # Close the polygon
        
        self.msp.add_lwpolyline(
            coords,
            dxfattribs={'layer': 'BOUNDARY', 'lineweight': 50}
        )
    
    def add_plots(self):
        """Add plot boundaries to DXF."""
        if self.msp is None or not self.plots:
            return
        
        for plot in self.plots:
            coords = [self._transform_point(x, y) for x, y in plot['coords']]
            coords.append(coords[0])
            
            self.msp.add_lwpolyline(
                coords,
                dxfattribs={'layer': 'PLOTS', 'lineweight': 25}
            )
            
            # Add plot number at centroid
            centroid_x = sum(c[0] for c in coords[:-1]) / len(coords[:-1])
            centroid_y = sum(c[1] for c in coords[:-1]) / len(coords[:-1])
            
            self.msp.add_text(
                f"Plot {plot['number']}",
                height=4,
                dxfattribs={'layer': 'PLOT_NUMBERS'}
            ).set_placement((centroid_x, centroid_y + 3), align=TextEntityAlignment.MIDDLE_CENTER)
            
            self.msp.add_text(
                f"{plot['area']:.1f} m²",
                height=3,
                dxfattribs={'layer': 'PLOT_NUMBERS'}
            ).set_placement((centroid_x, centroid_y - 3), align=TextEntityAlignment.MIDDLE_CENTER)
    
    def add_roads(self):
        """Add roads to DXF."""
        if self.msp is None or not self.roads:
            return
        
        for road in self.roads:
            coords = [self._transform_point(x, y) for x, y in road['coords']]
            coords.append(coords[0])
            
            # Road outline
            self.msp.add_lwpolyline(
                coords,
                dxfattribs={'layer': 'ROADS', 'lineweight': 35}
            )
            
            # Road label
            centroid_x = sum(c[0] for c in coords[:-1]) / 4
            centroid_y = sum(c[1] for c in coords[:-1]) / 4
            
            self.msp.add_text(
                f"ROAD ({self.road_width}m)",
                height=3,
                dxfattribs={'layer': 'ROADS'}
            ).set_placement((centroid_x, centroid_y), align=TextEntityAlignment.MIDDLE_CENTER)
    
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
    
    def add_summary_table(self):
        """Add plot summary table."""
        if self.msp is None or not self.plots:
            return
        
        # Table position (right side)
        tx = self.sheet_width - self.margin - 170
        ty = self.sheet_height - self.margin - 20
        
        # Title
        self.msp.add_text(
            "PLOT SCHEDULE",
            height=5,
            dxfattribs={'layer': 'TITLE_BLOCK'}
        ).set_placement((tx + 75, ty), align=TextEntityAlignment.MIDDLE_CENTER)
        
        # Table header
        ty -= 15
        col_widths = [40, 50, 50]
        row_height = 8
        
        headers = ['Plot No.', 'Area (m²)', 'Frontage (m)']
        x_pos = tx
        for header, width in zip(headers, col_widths):
            self.msp.add_text(
                header,
                height=3,
                dxfattribs={'layer': 'TITLE_BLOCK'}
            ).set_placement((x_pos + width/2, ty), align=TextEntityAlignment.MIDDLE_CENTER)
            x_pos += width
        
        # Header line
        ty -= 5
        self.msp.add_line((tx, ty), (tx + sum(col_widths), ty), dxfattribs={'layer': 'TITLE_BLOCK'})
        
        # Data rows
        total_area = 0
        for plot in self.plots[:15]:  # Limit to 15 rows to fit
            ty -= row_height
            x_pos = tx
            
            self.msp.add_text(
                str(plot['number']),
                height=2.5,
                dxfattribs={'layer': 'LABELS'}
            ).set_placement((x_pos + col_widths[0]/2, ty), align=TextEntityAlignment.MIDDLE_CENTER)
            x_pos += col_widths[0]
            
            self.msp.add_text(
                f"{plot['area']:.1f}",
                height=2.5,
                dxfattribs={'layer': 'LABELS'}
            ).set_placement((x_pos + col_widths[1]/2, ty), align=TextEntityAlignment.MIDDLE_CENTER)
            x_pos += col_widths[1]
            
            self.msp.add_text(
                f"{plot['frontage']:.1f}",
                height=2.5,
                dxfattribs={'layer': 'LABELS'}
            ).set_placement((x_pos + col_widths[2]/2, ty), align=TextEntityAlignment.MIDDLE_CENTER)
            
            total_area += plot['area']
        
        # Total
        ty -= row_height + 5
        self.msp.add_line((tx, ty + 5), (tx + sum(col_widths), ty + 5), dxfattribs={'layer': 'TITLE_BLOCK'})
        
        self.msp.add_text(
            "TOTAL",
            height=3,
            dxfattribs={'layer': 'TITLE_BLOCK'}
        ).set_placement((tx + col_widths[0]/2, ty), align=TextEntityAlignment.MIDDLE_CENTER)
        
        self.msp.add_text(
            f"{total_area:.1f}",
            height=3,
            dxfattribs={'layer': 'TITLE_BLOCK'}
        ).set_placement((tx + col_widths[0] + col_widths[1]/2, ty), align=TextEntityAlignment.MIDDLE_CENTER)
        
        self.msp.add_text(
            f"{len(self.plots)} plots",
            height=3,
            dxfattribs={'layer': 'TITLE_BLOCK'}
        ).set_placement((tx + col_widths[0] + col_widths[1] + col_widths[2]/2, ty), align=TextEntityAlignment.MIDDLE_CENTER)
    
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
    
    def generate(self,
                boundary: List[Tuple[float, float]],
                num_plots: int = None,
                plot_width: float = None,
                plot_depth: float = None,
                road_pattern: str = 'single') -> Dict:
        """
        Generate complete subdivision plan.
        
        Args:
            boundary: Outer boundary coordinates
            num_plots: Target number of plots
            plot_width: Target plot width
            plot_depth: Target plot depth
            road_pattern: Road layout pattern
            
        Returns:
            Statistics dict
        """
        # Subdivide
        plots = self.subdivide_rectangular(
            boundary, num_plots, plot_width, plot_depth, road_pattern
        )
        
        # Create document
        self.create_document()
        
        # Add elements
        self.add_border()
        self.add_boundary()
        self.add_roads()
        self.add_plots()
        self.add_summary_table()
        self.add_north_arrow()
        
        # Calculate stats
        total_area = sum(p['area'] for p in plots)
        boundary_area = self.calculate_polygon_area(boundary)
        
        return {
            'num_plots': len(plots),
            'total_plot_area': total_area,
            'boundary_area': boundary_area,
            'avg_plot_area': total_area / len(plots) if plots else 0,
            'road_area': boundary_area - total_area,
            'plots': plots
        }
    
    def save(self, filepath: str) -> str:
        """Save DXF file."""
        if not self.doc:
            raise ValueError("No document to save")
        
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.saveas(str(output_path))
        return str(output_path.absolute())


