"""
Survey to CAD - Professional Edition
Complete survey automation suite:
- Plot Boundaries
- Contour Maps
- Cut/Fill Volumes
- Land Subdivision
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from datetime import datetime
import zipfile
import io

from parsers import SurveyDataParser
from utils import GeometryCalculator, detect_coordinate_system, CoordinateTransformer
from cad_generator_pro import ProfessionalCADGenerator
from contour_generator import ContourGenerator
from volume_calculator import VolumeCalculator
from subdivision_planner import SubdivisionPlanner

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Survey to CAD Professional",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. SESSION STATE MANAGEMENT
# ==========================================
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'batch_id' not in st.session_state:
    st.session_state.batch_id = None


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def reset_state():
    st.session_state.processed_results = []
    st.session_state.batch_id = None

def load_data(file_content, file_name):
    """Load and parse survey data file."""
    tmp_path = None
    try:
        suffix = '.' + file_name.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        parser = SurveyDataParser()
        df = parser.parse_file(tmp_path)
        
        crs = detect_coordinate_system(df)
        if crs == 'EPSG:4326':
            transformer = CoordinateTransformer('EPSG:4326', 'EPSG:32644')
            df = transformer.transform_dataframe(df, lon_col='Easting', lat_col='Northing')
            
        return df, None
    except Exception as e:
        return None, str(e)
    finally:
        # Clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass

def analyze_groups(df):
    if 'Point_ID' not in df.columns:
        return {}
        
    groups = {}
    for _, row in df.iterrows():
        pid = str(row['Point_ID'])
        prefix = ''.join(filter(str.isalpha, pid)).upper() or 'DEFAULT'
        
        if prefix not in groups:
            groups[prefix] = []
        
        groups[prefix].append({
            'id': pid,
            'x': float(row['Easting']),
            'y': float(row['Northing']),
            'z': float(row.get('Elevation', 0))
        })
        
    return groups


# ==========================================
# PROCESSING FUNCTIONS
# ==========================================

def process_boundary(uploaded_file, project_name, location, drawn_by, scale, sheet_size, show_dims, show_labels):
    """Generate plot boundary drawing."""
    try:
        df, err = load_data(uploaded_file.getvalue(), uploaded_file.name)
        if err:
            return {'success': False, 'file': uploaded_file.name, 'error': err}
            
        groups = analyze_groups(df)
        if not groups:
            return {'success': False, 'file': uploaded_file.name, 'error': "No valid groups found"}

        cad = ProfessionalCADGenerator(sheet_size=sheet_size, scale=scale)
        cad.create_document()
        cad.add_drawing_border()
        cad.add_title_block(project_name, location, drawn_by)
        cad.add_north_arrow()
        
        all_x = df['Easting'].values
        all_y = df['Northing'].values
        extents = (min(all_x), min(all_y), max(all_x), max(all_y))
        
        stats = []
        for name, points in groups.items():
            if len(points) < 3:
                continue
                
            coords = [(p['x'], p['y']) for p in points]
            if coords[0] != coords[-1]:
                coords.append(coords[0])
                
            area = GeometryCalculator.calculate_area(coords[:-1])
            cad.add_plot_boundary(coords, extents)
            
            if show_dims:
                pids = [p['id'] for p in points]
                cad.add_dimensions(coords[:-1], extents, pids)
                
            if show_labels:
                xs = [c[0] for c in coords[:-1]]
                ys = [c[1] for c in coords[:-1]]
                centroid = (sum(xs)/len(xs), sum(ys)/len(ys))
                cad.add_clean_label(name, area, centroid, coords, extents)
                
            stats.append({'Group': name, 'Points': len(points), 'Area (m²)': area})
            
        if stats:
            summary_data = [{'name': s['Group'], 'area_m2': s['Area (m²)'], 'valid': True} for s in stats]
            cad.add_area_summary_table(summary_data, extents)
            
        # Use temp directory for cloud compatibility
        output_dir = Path(tempfile.gettempdir()) / "survey_to_cad"
        output_dir.mkdir(exist_ok=True, parents=True)
        filename = f"{Path(uploaded_file.name).stem}_BOUNDARY.dxf"
        output_path = output_dir / filename
        cad.save(str(output_path))
        
        return {
            'success': True, 'file': uploaded_file.name, 'path': str(output_path),
            'filename': filename, 'polygons': len(stats),
            'area': sum(s['Area (m²)'] for s in stats), 'type': 'boundary'
        }
    except Exception as e:
        return {'success': False, 'file': uploaded_file.name, 'error': str(e)}


def process_contour(uploaded_file, major_interval, minor_interval, index_interval, 
                    smoothing, sheet_size, show_points, show_point_labels, show_grid=True, 
                    grid_spacing=10.0, style='contour'):
    """Generate contour map or spot elevation grid."""
    try:
        df, err = load_data(uploaded_file.getvalue(), uploaded_file.name)
        if err:
            return {'success': False, 'file': uploaded_file.name, 'error': err}
        
        if 'Elevation' not in df.columns or df['Elevation'].isna().all() or (df['Elevation'] == 0).all():
            return {'success': False, 'file': uploaded_file.name, 'error': "No elevation data found"}
        
        points = [{'id': str(row['Point_ID']), 'x': float(row['Easting']), 
                   'y': float(row['Northing']), 'z': float(row['Elevation'])} 
                  for _, row in df.iterrows()]
        
        min_points = 3 if style == 'spot_elevation' else 4
        if len(points) < min_points:
            return {'success': False, 'file': uploaded_file.name, 'error': f"Need at least {min_points} points"}
        
        generator = ContourGenerator(
            major_interval=major_interval, minor_interval=minor_interval,
            index_interval=index_interval, smoothing=smoothing, sheet_size=sheet_size,
            grid_spacing=grid_spacing
        )
        stats = generator.generate(points, show_points=show_points, show_point_labels=show_point_labels, 
                                   show_grid=show_grid, style=style)
        
        # Use temp directory for cloud compatibility
        output_dir = Path(tempfile.gettempdir()) / "survey_to_cad"
        output_dir.mkdir(exist_ok=True, parents=True)
        filename = f"{Path(uploaded_file.name).stem}_CONTOUR.dxf"
        output_path = output_dir / filename
        generator.save(str(output_path))
        
        return {
            'success': True, 'file': uploaded_file.name, 'path': str(output_path),
            'filename': filename, 'num_points': stats['num_points'],
            'elevation_range': stats['elevation_range'],
            'num_contours': stats['num_major'] + stats['num_minor'], 'type': 'contour'
        }
    except Exception as e:
        return {'success': False, 'file': uploaded_file.name, 'error': str(e)}


def process_volume(uploaded_file, proposed_level, sheet_size, show_grid):
    """Generate cut/fill volume analysis."""
    try:
        df, err = load_data(uploaded_file.getvalue(), uploaded_file.name)
        if err:
            return {'success': False, 'file': uploaded_file.name, 'error': err}
        
        if 'Elevation' not in df.columns or df['Elevation'].isna().all():
            return {'success': False, 'file': uploaded_file.name, 'error': "No elevation data found"}
        
        points = [{'x': float(row['Easting']), 'y': float(row['Northing']), 
                   'z': float(row['Elevation'])} for _, row in df.iterrows()]
        
        if len(points) < 4:
            return {'success': False, 'file': uploaded_file.name, 'error': "Need at least 4 points"}
        
        calculator = VolumeCalculator(sheet_size=sheet_size)
        stats = calculator.generate(points, proposed_level=proposed_level, show_grid=show_grid)
        
        # Use temp directory for cloud compatibility
        output_dir = Path(tempfile.gettempdir()) / "survey_to_cad"
        output_dir.mkdir(exist_ok=True, parents=True)
        filename = f"{Path(uploaded_file.name).stem}_VOLUME.dxf"
        output_path = output_dir / filename
        calculator.save(str(output_path))
        
        return {
            'success': True, 'file': uploaded_file.name, 'path': str(output_path),
            'filename': filename, 'cut_volume': stats['cut_volume'],
            'fill_volume': stats['fill_volume'], 'net_volume': stats['net_volume'],
            'type': 'volume'
        }
    except Exception as e:
        return {'success': False, 'file': uploaded_file.name, 'error': str(e)}


def process_subdivision(uploaded_file, num_plots, plot_width, plot_depth, road_width, road_pattern, sheet_size):
    """Generate subdivision plan."""
    try:
        df, err = load_data(uploaded_file.getvalue(), uploaded_file.name)
        if err:
            return {'success': False, 'file': uploaded_file.name, 'error': err}
        
        # Get boundary from first group
        groups = analyze_groups(df)
        if not groups:
            return {'success': False, 'file': uploaded_file.name, 'error': "No valid boundary found"}
        
        first_group = list(groups.values())[0]
        boundary = [(p['x'], p['y']) for p in first_group]
        
        if len(boundary) < 3:
            return {'success': False, 'file': uploaded_file.name, 'error': "Need at least 3 boundary points"}
        
        planner = SubdivisionPlanner(sheet_size=sheet_size, road_width=road_width)
        stats = planner.generate(
            boundary, num_plots=num_plots if num_plots > 0 else None,
            plot_width=plot_width if plot_width > 0 else None,
            plot_depth=plot_depth if plot_depth > 0 else None,
            road_pattern=road_pattern
        )
        
        # Use temp directory for cloud compatibility
        output_dir = Path(tempfile.gettempdir()) / "survey_to_cad"
        output_dir.mkdir(exist_ok=True, parents=True)
        filename = f"{Path(uploaded_file.name).stem}_SUBDIVISION.dxf"
        output_path = output_dir / filename
        planner.save(str(output_path))
        
        return {
            'success': True, 'file': uploaded_file.name, 'path': str(output_path),
            'filename': filename, 'num_plots': stats['num_plots'],
            'total_area': stats['total_plot_area'], 'avg_area': stats['avg_plot_area'],
            'type': 'subdivision'
        }
    except Exception as e:
        return {'success': False, 'file': uploaded_file.name, 'error': str(e)}


# ==========================================
# 4. SIDEBAR - CONFIGURATION
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Processing Mode")
    mode = st.radio(
        "Select Mode",
        ["🗺️ Plot Boundaries", "📈 Contour Map", "⛏️ Cut/Fill Volume", "🏗️ Subdivision"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Mode-specific settings
    if mode == "🗺️ Plot Boundaries":
        st.subheader("Project Details")
        project_name = st.text_input("Project Name", value="Survey Project")
        location = st.text_input("Location", value="Site Location")
        drawn_by = st.text_input("Drawn By", value="Engineer")
        st.divider()
        st.subheader("Output Settings")
        scale = st.selectbox("Scale", ["1:100", "1:200", "1:500", "1:1000"], index=2)
        sheet_size = st.selectbox("Sheet Size", ["A1", "A2", "A3"], index=0)
        st.divider()
        st.subheader("Layers")
        show_dims = st.toggle("Show Dimensions", value=True)
        show_labels = st.toggle("Show Area Labels", value=True)
        
    elif mode == "📈 Contour Map":
        st.subheader("Map Style")
        map_style = st.radio(
            "Output Type",
            ["📊 Spot Elevation Grid", "🗺️ Contour Lines"],
            help="Spot Elevation = Grid + Points + Labels (like reference). Contour Lines = Traditional topographic contours."
        )
        
        st.divider()
        st.subheader("Grid Settings")
        grid_spacing = st.number_input("Grid Spacing (m)", min_value=5.0, max_value=50.0, value=10.0, step=5.0, 
                                       help="Grid overlay spacing in meters")
        sheet_size = st.selectbox("Sheet Size", ["A1", "A2", "A3"], index=0, key="contour_sheet")
        
        # Only show contour-specific settings for contour mode
        if map_style == "🗺️ Contour Lines":
            st.divider()
            st.subheader("Contour Intervals")
            major_interval = st.number_input("Major (m)", min_value=0.5, max_value=10.0, value=1.0, step=0.5)
            minor_interval = st.number_input("Minor (m)", min_value=0.25, max_value=5.0, value=0.5, step=0.25)
            index_interval = st.number_input("Index (m)", min_value=1.0, max_value=20.0, value=5.0, step=1.0)
            smoothing = st.slider("Smoothing", 1.0, 5.0, 2.5, 0.5, help="Higher = smoother contours")
            st.divider()
            st.subheader("Display Options")
            show_grid = st.toggle("Show Grid Overlay", value=True, key="contour_grid")
            show_points = st.toggle("Show Survey Points", value=True)
            show_point_labels = st.toggle("Show Elevation Labels", value=False)
        else:
            # Defaults for spot elevation mode (not used but need to be defined)
            major_interval = 1.0
            minor_interval = 0.5
            index_interval = 5.0
            smoothing = 2.5
            show_grid = True
            show_points = True
            show_point_labels = True
        
    elif mode == "⛏️ Cut/Fill Volume":
        st.subheader("Proposed Level")
        proposed_level = st.number_input("Proposed Level (m)", value=0.0, step=0.1,
                                         help="Leave 0 to use average existing level")
        st.divider()
        st.subheader("Output")
        sheet_size = st.selectbox("Sheet Size", ["A1", "A2", "A3"], index=0, key="volume_sheet")
        show_grid = st.toggle("Show Grid Lines", value=True)
        
    else:  # Subdivision
        st.subheader("Plot Configuration")
        num_plots = st.number_input("Number of Plots", min_value=0, value=0, 
                                    help="0 = auto-calculate based on dimensions")
        plot_width = st.number_input("Plot Width (m)", min_value=0.0, value=10.0, step=1.0)
        plot_depth = st.number_input("Plot Depth (m)", min_value=0.0, value=20.0, step=1.0)
        st.divider()
        st.subheader("Road Layout")
        road_width = st.number_input("Road Width (m)", min_value=3.0, max_value=20.0, value=6.0, step=0.5)
        road_pattern = st.selectbox("Road Pattern", ["single", "double", "none"])
        st.divider()
        sheet_size = st.selectbox("Sheet Size", ["A1", "A2", "A3"], index=0, key="subdiv_sheet")
    
    st.divider()
    if st.button("🔄 Reset", use_container_width=True):
        reset_state()
        st.rerun()


# ==========================================
# 5. MAIN CONTENT
# ==========================================

st.title("📐 Survey to CAD")

mode_descriptions = {
    "🗺️ Plot Boundaries": "Convert survey points to professional boundary drawings",
    "📈 Contour Map": "Generate contour maps from elevation data",
    "⛏️ Cut/Fill Volume": "Calculate earthwork volumes for grading",
    "🏗️ Subdivision": "Auto-subdivide land into plots with roads"
}
st.caption(mode_descriptions.get(mode, ""))

# --- UPLOAD & PROCESS ---
if not st.session_state.processed_results:
    uploaded_files = st.file_uploader(
        "Upload Survey Files", 
        type=['dat', 'csv', 'txt'], 
        accept_multiple_files=True,
        help="Supports .DAT, .CSV, .TXT from DGPS/Total Station"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) ready")
        
        if st.button("⚡ Generate", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                
                if mode == "🗺️ Plot Boundaries":
                    result = process_boundary(file, project_name, location, drawn_by, 
                                              scale, sheet_size, show_dims, show_labels)
                elif mode == "📈 Contour Map":
                    # Determine style based on selection
                    style = 'spot_elevation' if map_style == "📊 Spot Elevation Grid" else 'contour'
                    result = process_contour(file, major_interval, minor_interval, index_interval,
                                             smoothing, sheet_size, show_points, show_point_labels, 
                                             show_grid, grid_spacing, style)
                elif mode == "⛏️ Cut/Fill Volume":
                    level = proposed_level if proposed_level != 0 else None
                    result = process_volume(file, level, sheet_size, show_grid)
                else:
                    result = process_subdivision(file, num_plots, plot_width, plot_depth,
                                                  road_width, road_pattern, sheet_size)
                
                results.append(result)
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.session_state.processed_results = results
            st.session_state.batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            status_text.empty()
            st.rerun()

# --- RESULTS ---
else:
    results = st.session_state.processed_results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    col1, col2 = st.columns(2)
    col1.metric("✅ Successful", len(successful))
    col2.metric("❌ Failed", len(failed))
    
    st.divider()
    
    if successful:
        st.subheader("📥 Download Files")
        
        for i, r in enumerate(successful):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{r['file']}** → `{r['filename']}`")
                
                # Mode-specific stats
                if r['type'] == 'boundary':
                    st.caption(f"{r['polygons']} polygons • {r['area']:,.0f} m²")
                elif r['type'] == 'contour':
                    st.caption(f"{r['num_points']} pts • {r['num_contours']} contours • {r['elevation_range'][0]:.1f}-{r['elevation_range'][1]:.1f}m")
                elif r['type'] == 'volume':
                    net = r['net_volume']
                    status = "CUT" if net > 0 else "FILL" if net < 0 else "BALANCED"
                    st.caption(f"Cut: {r['cut_volume']:,.0f} m³ | Fill: {r['fill_volume']:,.0f} m³ | Net: {abs(net):,.0f} m³ ({status})")
                else:
                    st.caption(f"{r['num_plots']} plots • Avg: {r['avg_area']:.0f} m²")
            
            with col3:
                try:
                    with open(r['path'], "rb") as f:
                        file_bytes = f.read()
                    st.download_button("⬇️ DXF", file_bytes, r['filename'], "application/dxf", key=f"dl_{i}", use_container_width=True)
                except FileNotFoundError:
                    st.error(f"File not found: {r['filename']}")
        
        if len(successful) > 1:
            st.divider()
            try:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for r in successful:
                        if os.path.exists(r['path']):
                            zf.write(r['path'], r['filename'])
                st.download_button(f"📦 Download All ({len(successful)})", zip_buffer.getvalue(),
                                  f"batch_{st.session_state.batch_id}.zip", "application/zip", use_container_width=True)
            except Exception as e:
                st.error(f"Error creating zip file: {str(e)}")

    if failed:
        st.divider()
        st.subheader("❌ Failed")
        for f in failed:
            st.error(f"**{f['file']}**: {f['error']}")
    
    st.divider()
    if st.button("⬅️ Process More", use_container_width=True):
        reset_state()
        st.rerun()
