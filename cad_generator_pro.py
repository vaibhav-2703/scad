"""
Professional CAD Generator for Survey Data
Generates municipal-quality DXF files with proper layers, dimensions, and annotations.
"""

import ezdxf
from ezdxf import units
from ezdxf.enums import TextEntityAlignment
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import io


class ProfessionalCADGenerator:
    """
    Professional-grade DXF generator for survey/land parcel drawings.
    Produces AutoCAD-compatible files with proper layers, dimensions, and annotations.
    """
    
    # Sheet sizes in mm
    SHEET_SIZES = {
        'A1': (841, 594),
        'A2': (594, 420),
        'A3': (420, 297),
    }
    
    # Standard layers
    LAYERS = {
        'BORDER': {'color': 7},           # White
        'TITLE_BLOCK': {'color': 7},      # White
        'PLOT_BOUNDARY': {'color': 3},    # Green
        'DIMENSIONS': {'color': 4},       # Cyan
        'LABELS': {'color': 2},           # Yellow
        'NORTH_ARROW': {'color': 7},      # White
        'SUMMARY_TABLE': {'color': 7},    # White
        'TEXT_CLEAN': {'color': 3},       # Green
    }
    
    def __init__(self, sheet_size: str = 'A1', scale: str = '1:500'):
        """Initialize the CAD generator."""
        self.sheet_size = sheet_size
        self.scale = scale
        self.scale_factor = int(scale.split(':')[1])
        
        # Get sheet dimensions
        self.sheet_width, self.sheet_height = self.SHEET_SIZES.get(sheet_size, (841, 594))
        
        # Drawing area (with margins)
        self.margin = 20  # mm
        self.drawing_width = self.sheet_width - (2 * self.margin)
        self.drawing_height = self.sheet_height - (2 * self.margin)
        
        # Document
        self.doc = None
        self.msp = None
        
        # Transform parameters (set during add_plot_boundary)
        self.transform_scale = 1.0
        self.transform_offset_x = 0.0
        self.transform_offset_y = 0.0
    
    def create_document(self):
        """Create a new DXF document with proper setup."""
        self.doc = ezdxf.new('R2010')
        self.doc.units = units.MM
        self.msp = self.doc.modelspace()
        
        # Create layers
        for layer_name, props in self.LAYERS.items():
            self.doc.layers.add(layer_name, color=props['color'])
        
        return self
    
    def add_drawing_border(self):
        """Add a professional drawing border."""
        if self.msp is None:
            raise ValueError("Document not created. Call create_document() first.")
        
        # Outer border
        border_points = [
            (self.margin, self.margin),
            (self.sheet_width - self.margin, self.margin),
            (self.sheet_width - self.margin, self.sheet_height - self.margin),
            (self.margin, self.sheet_height - self.margin),
            (self.margin, self.margin)
        ]
        self.msp.add_lwpolyline(border_points, dxfattribs={'layer': 'BORDER', 'lineweight': 50})
        
        # Inner border (5mm inside)
        inner_margin = self.margin + 5
        inner_points = [
            (inner_margin, inner_margin),
            (self.sheet_width - inner_margin, inner_margin),
            (self.sheet_width - inner_margin, self.sheet_height - inner_margin),
            (inner_margin, self.sheet_height - inner_margin),
            (inner_margin, inner_margin)
        ]
        self.msp.add_lwpolyline(inner_points, dxfattribs={'layer': 'BORDER', 'lineweight': 25})
        
        return self
    
    def add_title_block(self, project_name: str = "", location: str = "", drawn_by: str = ""):
        """Add a simple title block (just a box, no text fields)."""
        if self.msp is None:
            raise ValueError("Document not created. Call create_document() first.")
        
        # Title block dimensions
        tb_width = 180
        tb_height = 40
        tb_x = self.sheet_width - self.margin - tb_width - 5
        tb_y = self.margin + 5
        
        # Draw title block border only
        tb_points = [
            (tb_x, tb_y),
            (tb_x + tb_width, tb_y),
            (tb_x + tb_width, tb_y + tb_height),
            (tb_x, tb_y + tb_height),
            (tb_x, tb_y)
        ]
        self.msp.add_lwpolyline(tb_points, dxfattribs={'layer': 'TITLE_BLOCK', 'lineweight': 35})
        
        return self
    
    def add_north_arrow(self):
        """Add a north arrow symbol."""
        if self.msp is None:
            raise ValueError("Document not created. Call create_document() first.")
        
        # Position in top-right corner
        cx = self.sheet_width - self.margin - 30
        cy = self.sheet_height - self.margin - 40
        size = 15
        
        # Arrow shape
        arrow_points = [
            (cx, cy - size),
            (cx + size * 0.3, cy + size * 0.5),
            (cx, cy + size * 0.3),
            (cx - size * 0.3, cy + size * 0.5),
            (cx, cy - size)
        ]
        self.msp.add_lwpolyline(arrow_points, close=True, dxfattribs={'layer': 'NORTH_ARROW'})
        
        # Fill the right half
        fill_points = [
            (cx, cy - size),
            (cx + size * 0.3, cy + size * 0.5),
            (cx, cy + size * 0.3),
            (cx, cy - size)
        ]
        self.msp.add_hatch(dxfattribs={'layer': 'NORTH_ARROW'}).paths.add_polyline_path(fill_points)
        
        # 'N' label
        self.msp.add_text(
            'N',
            height=8,
            dxfattribs={'layer': 'NORTH_ARROW'}
        ).set_placement((cx, cy + size + 5), align=TextEntityAlignment.MIDDLE_CENTER)
        
        return self
    
    def _transform_point(self, x: float, y: float, extents: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Transform real-world coordinates to sheet coordinates."""
        min_x, min_y, max_x, max_y = extents
        
        # Calculate data dimensions
        data_width = max_x - min_x
        data_height = max_y - min_y
        
        if data_width == 0 or data_height == 0:
            return (self.sheet_width / 2, self.sheet_height / 2)
        
        # Available drawing area (leave space for title block and table)
        avail_width = self.drawing_width - 200  # Reserve space for table
        avail_height = self.drawing_height - 80  # Reserve space for title block
        
        # Calculate scale to fit
        scale_x = avail_width / data_width
        scale_y = avail_height / data_height
        scale = min(scale_x, scale_y) * 0.85  # 85% to add padding
        
        # Store transform params
        self.transform_scale = scale
        
        # Center offset
        scaled_width = data_width * scale
        scaled_height = data_height * scale
        offset_x = self.margin + 50 + (avail_width - scaled_width) / 2
        offset_y = self.margin + 60 + (avail_height - scaled_height) / 2
        
        self.transform_offset_x = offset_x - min_x * scale
        self.transform_offset_y = offset_y - min_y * scale
        
        # Transform
        tx = (x - min_x) * scale + offset_x
        ty = (y - min_y) * scale + offset_y
        
        return (tx, ty)
    
    def add_plot_boundary(self, coords: List[Tuple[float, float]], extents: Tuple[float, float, float, float]):
        """Add a plot boundary polygon."""
        if self.msp is None:
            raise ValueError("Document not created. Call create_document() first.")
        
        if len(coords) < 3:
            return self
        
        # Transform coordinates
        transformed = [self._transform_point(x, y, extents) for x, y in coords]
        
        # Draw polyline
        self.msp.add_lwpolyline(
            transformed,
            close=True,
            dxfattribs={'layer': 'PLOT_BOUNDARY', 'lineweight': 35}
        )
        
        return self
    
    def add_dimensions(self, coords: List[Tuple[float, float]], extents: Tuple[float, float, float, float], point_ids: List[str] = None):
        """Add dimension annotations for each segment."""
        if self.msp is None or len(coords) < 2:
            return self
        
        text_height = 0.0018 * (extents[2] - extents[0])  # Reduced height
        text_height = max(2.0, min(4.0, text_height * self.transform_scale))
        
        for i in range(len(coords)):
            p1 = coords[i]
            p2 = coords[(i + 1) % len(coords)]
            
            # Calculate distance
            dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # Transform midpoint
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            t_mid = self._transform_point(mid_x, mid_y, extents)
            
            # Calculate angle for text rotation
            angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
            if angle > 90 or angle < -90:
                angle += 180
            
            # Offset text slightly from line
            offset = text_height * 1.2
            perp_angle = math.radians(angle + 90)
            text_x = t_mid[0] + math.cos(perp_angle) * offset
            text_y = t_mid[1] + math.sin(perp_angle) * offset
            
            # Add dimension text
            dim_text = f"{dist:.2f}m"
            self.msp.add_text(
                dim_text,
                height=text_height,
                rotation=angle,
                dxfattribs={'layer': 'DIMENSIONS'}
            ).set_placement((text_x, text_y), align=TextEntityAlignment.MIDDLE_CENTER)
            
            # Add point ID if provided
            if point_ids and i < len(point_ids):
                t_p1 = self._transform_point(p1[0], p1[1], extents)
                self.msp.add_text(
                    point_ids[i],
                    height=text_height * 0.8,
                    dxfattribs={'layer': 'TEXT_CLEAN'}
                ).set_placement((t_p1[0], t_p1[1] + text_height * 1.5), align=TextEntityAlignment.MIDDLE_CENTER)
        
        return self
    
    def add_clean_label(self, name: str, area: float, centroid: Tuple[float, float], 
                        coords: List[Tuple[float, float]], extents: Tuple[float, float, float, float]):
        """Add a clean area label at the polygon centroid."""
        if self.msp is None:
            return self
        
        t_centroid = self._transform_point(centroid[0], centroid[1], extents)
        text_height = 3.0  # Fixed 3mm height
        
        # Convert area
        area_hectares = area / 10000
        
        # Line 1: Name
        self.msp.add_text(
            name,
            height=text_height,
            dxfattribs={'layer': 'LABELS'}
        ).set_placement((t_centroid[0], t_centroid[1] + text_height * 2), align=TextEntityAlignment.MIDDLE_CENTER)
        
        # Line 2: Area in sq.m
        self.msp.add_text(
            f"{area:,.2f} sq.m",
            height=text_height,
            dxfattribs={'layer': 'LABELS'}
        ).set_placement((t_centroid[0], t_centroid[1]), align=TextEntityAlignment.MIDDLE_CENTER)
        
        # Line 3: Area in hectares
        self.msp.add_text(
            f"({area_hectares:.4f} Ha)",
            height=text_height * 0.8,
            dxfattribs={'layer': 'LABELS'}
        ).set_placement((t_centroid[0], t_centroid[1] - text_height * 1.5), align=TextEntityAlignment.MIDDLE_CENTER)
        
        return self
    
    def add_area_summary_table(self, polygon_data: List[Dict], extents: Tuple[float, float, float, float]):
        """Add an area summary table."""
        if self.msp is None or not polygon_data:
            return self
        
        # Table position (bottom-right)
        table_x = self.sheet_width - self.margin - 200
        table_y = self.margin + 60
        
        # Table dimensions
        col_widths = [60, 70, 70]  # Wider columns
        row_height = 8
        header_height = 10
        text_height_header = 3.5
        text_height_body = 3.0
        padding = 3
        
        total_width = sum(col_widths)
        total_rows = len(polygon_data) + 2  # Header + data + total
        total_height = header_height + (total_rows - 1) * row_height
        
        # Draw outer border
        border = [
            (table_x, table_y),
            (table_x + total_width, table_y),
            (table_x + total_width, table_y + total_height),
            (table_x, table_y + total_height),
            (table_x, table_y)
        ]
        self.msp.add_lwpolyline(border, dxfattribs={'layer': 'SUMMARY_TABLE', 'lineweight': 35})
        
        # Draw header separator
        header_y = table_y + total_height - header_height
        self.msp.add_line(
            (table_x, header_y),
            (table_x + total_width, header_y),
            dxfattribs={'layer': 'SUMMARY_TABLE'}
        )
        
        # Draw column separators
        x_pos = table_x
        for i, w in enumerate(col_widths[:-1]):
            x_pos += w
            self.msp.add_line(
                (x_pos, table_y),
                (x_pos, table_y + total_height),
                dxfattribs={'layer': 'SUMMARY_TABLE'}
            )
        
        # Draw row separators
        y_pos = header_y
        for i in range(len(polygon_data) + 1):
            y_pos -= row_height
            self.msp.add_line(
                (table_x, y_pos),
                (table_x + total_width, y_pos),
                dxfattribs={'layer': 'SUMMARY_TABLE'}
            )
        
        # Header text
        headers = ['Parcel', 'Area (sq.m)', 'Area (Ha)']
        x_pos = table_x
        for i, (header, width) in enumerate(zip(headers, col_widths)):
            self.msp.add_text(
                header,
                height=text_height_header,
                dxfattribs={'layer': 'SUMMARY_TABLE'}
            ).set_placement((x_pos + padding, header_y + header_height / 2), align=TextEntityAlignment.MIDDLE_LEFT)
            x_pos += width
        
        # Data rows
        y_pos = header_y - row_height / 2
        total_area = 0
        for data in polygon_data:
            if not data.get('valid', True):
                continue
                
            area = data.get('area_m2', 0)
            total_area += area
            
            x_pos = table_x
            
            # Name
            self.msp.add_text(
                str(data.get('name', 'Unknown')),
                height=text_height_body,
                dxfattribs={'layer': 'SUMMARY_TABLE'}
            ).set_placement((x_pos + padding, y_pos), align=TextEntityAlignment.MIDDLE_LEFT)
            x_pos += col_widths[0]
            
            # Area sq.m
            self.msp.add_text(
                f"{area:,.2f}",
                height=text_height_body,
                dxfattribs={'layer': 'SUMMARY_TABLE'}
            ).set_placement((x_pos + padding, y_pos), align=TextEntityAlignment.MIDDLE_LEFT)
            x_pos += col_widths[1]
            
            # Area Ha
            self.msp.add_text(
                f"{area / 10000:.4f}",
                height=text_height_body,
                dxfattribs={'layer': 'SUMMARY_TABLE'}
            ).set_placement((x_pos + padding, y_pos), align=TextEntityAlignment.MIDDLE_LEFT)
            
            y_pos -= row_height
        
        # Total row
        x_pos = table_x
        self.msp.add_text(
            "TOTAL",
            height=text_height_header,
            dxfattribs={'layer': 'SUMMARY_TABLE'}
        ).set_placement((x_pos + padding, y_pos), align=TextEntityAlignment.MIDDLE_LEFT)
        x_pos += col_widths[0]
        
        self.msp.add_text(
            f"{total_area:,.2f}",
            height=text_height_header,
            dxfattribs={'layer': 'SUMMARY_TABLE'}
        ).set_placement((x_pos + padding, y_pos), align=TextEntityAlignment.MIDDLE_LEFT)
        x_pos += col_widths[1]
        
        self.msp.add_text(
            f"{total_area / 10000:.4f}",
            height=text_height_header,
            dxfattribs={'layer': 'SUMMARY_TABLE'}
        ).set_placement((x_pos + padding, y_pos), align=TextEntityAlignment.MIDDLE_LEFT)
        
        return self
    
    def save(self, filepath: str) -> str:
        """Save DXF file to specified path."""
        if not self.doc:
            raise ValueError("No document to save. Create document first.")
        
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.doc.saveas(str(output_path))
        return str(output_path.absolute())
