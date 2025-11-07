"""Real Project Analyzer - Shows ACTUAL project metrics."""

import streamlit as st
import os
from pathlib import Path
from datetime import datetime
from .real_metrics import RealProjectMetrics


class RealProjectAnalyzer:
    """Display REAL project analysis, not placeholders."""

    def __init__(self):
        self.metrics = RealProjectMetrics()

    def render(self):
        """Render real project analysis."""
        st.markdown("## 🔬 Real Project Analysis")
        st.markdown("*Analyzing actual files in your project - not demo data!*")

        # Add path selector
        col1, col2 = st.columns([3, 1])
        with col1:
            project_path = st.text_input(
                "Project Path",
                value=os.getcwd(),
                help="Enter the path to analyze"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Analyze", use_container_width=True):
                st.session_state.analyzing = True

        # Get real stats
        if project_path and os.path.exists(project_path):
            self.metrics.project_path = project_path
            stats = self.metrics.get_project_stats()

            # Display real metrics
            tabs = st.tabs([
                "📊 Overview",
                "📝 Code Analysis",
                "📌 TODOs",
                "🌿 Git Stats",
                "📦 Dependencies",
                "🏗️ Structure"
            ])

            with tabs[0]:
                self.render_overview(stats)

            with tabs[1]:
                self.render_code_analysis(stats['code_analysis'])

            with tabs[2]:
                self.render_todos(stats['todos'])

            with tabs[3]:
                self.render_git_stats(stats['git_stats'])

            with tabs[4]:
                self.render_dependencies(stats['dependencies'])

            with tabs[5]:
                self.render_structure(stats['project_structure'])

        else:
            st.error("Invalid project path!")

    def render_overview(self, stats):
        """Render overview metrics."""
        st.markdown("### 📊 Project Overview")

        # File statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Files", stats['files']['total'])

        with col2:
            st.metric("Python Files", stats['files']['python'])

        with col3:
            st.metric("Size (MB)", stats['files']['total_size_mb'])

        with col4:
            st.metric("TODOs", stats['todos']['total'])

        st.markdown("---")

        # Code statistics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Lines", f"{stats['code_analysis']['total_lines']:,}")

        with col2:
            st.metric("Code Lines", f"{stats['code_analysis']['code_lines']:,}")

        with col3:
            st.metric("Functions", stats['code_analysis']['functions'])

        with col4:
            st.metric("Classes", stats['code_analysis']['classes'])

        # Git overview if available
        if stats['git_stats']['initialized']:
            st.markdown("---")
            st.markdown("### 🌿 Git Summary")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Branch", stats['git_stats']['branch'])

            with col2:
                st.metric("Commits", stats['git_stats']['commits'])

            with col3:
                st.metric("Authors", len(stats['git_stats']['authors']))

            with col4:
                changes = stats['git_stats']['uncommitted_changes']
                st.metric("Uncommitted", changes, delta="⚠️" if changes > 0 else None)

            if stats['git_stats']['last_commit']:
                st.info(f"**Last commit:** {stats['git_stats']['last_commit']}")

    def render_code_analysis(self, analysis):
        """Render code analysis details."""
        st.markdown("### 📝 Code Analysis")

        # Line breakdown
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Line Breakdown:**")
            total = analysis['total_lines']
            if total > 0:
                code_pct = round(analysis['code_lines'] / total * 100, 1)
                comment_pct = round(analysis['comment_lines'] / total * 100, 1)
                blank_pct = round(analysis['blank_lines'] / total * 100, 1)

                st.progress(code_pct / 100)
                st.caption(f"Code: {analysis['code_lines']:,} lines ({code_pct}%)")

                st.progress(comment_pct / 100)
                st.caption(f"Comments: {analysis['comment_lines']:,} lines ({comment_pct}%)")

                st.progress(blank_pct / 100)
                st.caption(f"Blank: {analysis['blank_lines']:,} lines ({blank_pct}%)")

        with col2:
            st.markdown("**Code Structure:**")
            st.metric("Functions", analysis['functions'])
            st.metric("Classes", analysis['classes'])
            st.metric("Unique Imports", analysis['imports'])

        # Code quality indicators
        st.markdown("---")
        st.markdown("**Quality Indicators:**")

        # Calculate some basic metrics
        if analysis['code_lines'] > 0:
            comment_ratio = round(analysis['comment_lines'] / analysis['code_lines'] * 100, 1)
            avg_function_size = round(analysis['code_lines'] / max(analysis['functions'], 1), 1)

            col1, col2, col3 = st.columns(3)

            with col1:
                quality = "Good" if comment_ratio > 10 else "Low"
                st.metric("Comment Ratio", f"{comment_ratio}%", delta=quality)

            with col2:
                complexity = "Simple" if avg_function_size < 50 else "Complex"
                st.metric("Avg Function Size", f"{avg_function_size} lines", delta=complexity)

            with col3:
                modularity = "Good" if analysis['classes'] > 5 else "Consider OOP"
                st.metric("Class Count", analysis['classes'], delta=modularity)

    def render_todos(self, todos):
        """Render TODO analysis."""
        st.markdown("### 📌 TODO Analysis")

        if todos['total'] == 0:
            st.success("✨ No TODOs found in the project!")
            return

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("TODOs", len(todos['TODO']))

        with col2:
            st.metric("FIXMEs", len(todos['FIXME']))

        with col3:
            st.metric("HACKs", len(todos['HACK']))

        with col4:
            st.metric("NOTEs", len(todos['NOTE']))

        # Detailed list
        st.markdown("---")
        st.markdown("**TODO Details:**")

        for todo_type in ['TODO', 'FIXME', 'HACK', 'NOTE']:
            if todos[todo_type]:
                with st.expander(f"{todo_type} ({len(todos[todo_type])})"):
                    for item in todos[todo_type][:10]:  # Show first 10
                        st.markdown(f"""
                        **{item['file']}:{item['line']}**
                        ```
                        {item['text'][:100]}
                        ```
                        """)

    def render_git_stats(self, git_stats):
        """Render git statistics."""
        st.markdown("### 🌿 Git Statistics")

        if not git_stats['initialized']:
            st.warning("⚠️ Not a git repository")
            return

        # Repository info
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Repository Info:**")
            st.info(f"Branch: **{git_stats['branch']}**")
            st.info(f"Total Commits: **{git_stats['commits']}**")
            st.info(f"Contributors: **{len(git_stats['authors'])}**")

        with col2:
            st.markdown("**Working Tree:**")
            if git_stats['uncommitted_changes'] > 0:
                st.warning(f"⚠️ {git_stats['uncommitted_changes']} uncommitted changes")
            else:
                st.success("✅ Working tree clean")

            if git_stats['untracked_files'] > 0:
                st.warning(f"📁 {git_stats['untracked_files']} untracked files")

        # Authors
        if git_stats['authors']:
            st.markdown("**Top Contributors:**")
            for author in git_stats['authors'][:5]:
                st.caption(f"👤 {author}")

        # Branches
        if git_stats['branches']:
            st.markdown("**Branches:**")
            branch_list = ", ".join(git_stats['branches'][:5])
            st.caption(branch_list)

    def render_dependencies(self, deps):
        """Render dependency analysis."""
        st.markdown("### 📦 Dependencies")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Python Dependencies:**")
            if deps['python']['found']:
                st.success(f"✅ Found {deps['python']['count']} packages")
                if deps['python']['packages']:
                    st.markdown("**Top packages:**")
                    for pkg in deps['python']['packages'][:5]:
                        st.caption(f"• {pkg}")
            else:
                st.info("No Python dependencies found")

        with col2:
            st.markdown("**JavaScript Dependencies:**")
            if deps['javascript']['found']:
                st.success(f"✅ Found {deps['javascript']['count']} packages")
                if deps['javascript']['packages']:
                    st.markdown("**Top packages:**")
                    for pkg in deps['javascript']['packages'][:5]:
                        st.caption(f"• {pkg}")
            else:
                st.info("No JavaScript dependencies found")

    def render_structure(self, structure):
        """Render project structure analysis."""
        st.markdown("### 🏗️ Project Structure")

        # Key indicators
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Depth", structure['depth'])

        with col2:
            st.metric("Directories", structure['directories'])

        with col3:
            has_tests = "✅" if structure['has_tests'] else "❌"
            st.metric("Tests", has_tests)

        with col4:
            has_docs = "✅" if structure['has_docs'] else "❌"
            st.metric("Docs", has_docs)

        # Additional checks
        st.markdown("**Project Features:**")

        features = []
        if structure['has_tests']:
            features.append("✅ Has test directory")
        else:
            features.append("⚠️ No tests found")

        if structure['has_docs']:
            features.append("✅ Has documentation")
        else:
            features.append("⚠️ No docs directory")

        if structure['has_ci']:
            features.append("✅ CI/CD configured")

        if structure['has_docker']:
            features.append("🐳 Docker support")

        for feature in features:
            st.caption(feature)

        # Config files
        if structure['config_files']:
            st.markdown("**Configuration Files:**")
            for cfg in structure['config_files'][:5]:
                st.caption(f"• {cfg}")

        # Key directories
        if structure['key_directories']:
            st.markdown("**Main Directories:**")
            dirs = ", ".join(structure['key_directories'][:8])
            st.info(dirs)