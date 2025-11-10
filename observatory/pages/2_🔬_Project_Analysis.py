"""
Project Analysis - Code metrics, git stats, dependencies.
Multi-page app component for Observatory Developer Suite.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add dev_dashboard components to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'dev_dashboard'))

from components.real_project_analyzer import RealProjectAnalyzer

# Page configuration
st.set_page_config(
    page_title="Project Analysis",
    page_icon="🔬",
    layout="wide"
)

# Apply Observatory styling
st.markdown("""
<style>
/* Match Observatory dark theme */
.stApp {
    background-color: #1a1a1a !important;
}

h1, h2, h3 {
    color: #FFD700 !important;
}

[data-testid="stMetricValue"] {
    color: #FFD700 !important;
}

/* Button styling */
.stButton > button {
    background-color: #2d2d2d !important;
    color: #ffffff !important;
    border: 1px solid #FFD700 !important;
}

.stButton > button:hover {
    background-color: #3d3d3d !important;
    color: #FFD700 !important;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<h1 style='text-align: center; color: #FFD700;'>
    🔬 Project Analysis
</h1>
<p style='text-align: center; color: #888;'>
    Code metrics, git statistics, and project structure
</p>
<hr style='border: 1px solid #FFD700;'>
""", unsafe_allow_html=True)

# Render project analyzer
analyzer = RealProjectAnalyzer()
analyzer.render()
