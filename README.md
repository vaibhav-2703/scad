# 📐 Survey to CAD Professional

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Convert DGPS/Total Station survey data to professional AutoCAD DXF drawings. Built for civil engineers who need quick, accurate CAD outputs from field survey data.

---

## ✨ Features

| Mode | Description |
|------|-------------|
| **🗺️ Plot Boundaries** | Convert survey points to boundary drawings with dimensions, area labels, and summary tables |
| **📈 Contour Map** | Generate contour lines from elevation data with customizable intervals |
| **⛏️ Cut/Fill Volume** | Calculate earthwork volumes for grading with visual cut/fill zones |
| **🏗️ Subdivision** | Auto-subdivide land into plots with road layouts |

---

## 🚀 Quick Start

### Option 1: Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/survey-to-cad.git
cd survey-to-cad

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Option 2: Deploy to Streamlit Cloud (Free)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app" → Select your forked repo
5. Set main file path: `app.py`
6. Click "Deploy"

Your app will be live at: `https://your-username-survey-to-cad.streamlit.app`

---

## 📁 Supported Input Formats

| Format | Description |
|--------|-------------|
| `.DAT` | DGPS/Total Station format (Point_ID, Description, Easting, Northing, Elevation) |
| `.CSV` | Standard CSV with header row |
| `.TXT` | Text files with comma/space separated values |

### Sample Data Format

```
Point_ID, Easting, Northing, Elevation, Description
a001, 500100.000, 2850100.000, 150.25, Corner
a002, 500150.000, 2850100.000, 150.30, Corner
a003, 500150.000, 2850150.000, 150.15, Corner
```

---

## 📂 Project Structure

```
survey-to-cad/
├── app.py                  # Main Streamlit application
├── parsers.py              # Survey data file parsers
├── utils.py                # Geometry calculations & coordinate transforms
├── cad_generator_pro.py    # DXF generation for plot boundaries
├── contour_generator.py    # Contour map generation
├── volume_calculator.py    # Cut/fill volume analysis
├── subdivision_planner.py  # Land subdivision planning
├── requirements.txt        # Python dependencies
├── .streamlit/config.toml  # Streamlit configuration
└── LICENSE                 # MIT License
```

---

## 🔧 Requirements

- Python 3.10+
- Dependencies: streamlit, pandas, numpy, scipy, ezdxf, pyproj, matplotlib

---

## 📖 Usage

1. **Open the app** in your browser (localhost:8501 or deployed URL)
2. **Select processing mode** from the sidebar (Boundaries, Contour, Volume, Subdivision)
3. **Configure settings** (scale, intervals, project details, etc.)
4. **Upload** one or more survey files (.DAT, .CSV, .TXT)
5. **Click "Generate"** and download your DXF files

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- DXF generation powered by [ezdxf](https://ezdxf.mozman.at/)
- Coordinate transformations via [pyproj](https://pyproj4.github.io/pyproj/)

---

Made with ❤️ for Civil Engineers
