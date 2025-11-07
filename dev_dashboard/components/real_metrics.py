"""REAL metrics component that actually analyzes your project."""

import os
import subprocess
from pathlib import Path
from datetime import datetime
import re
from typing import Dict, List, Tuple
import json


class RealProjectMetrics:
    """Actually analyze the real project for meaningful metrics."""

    def __init__(self, project_path: str = None):
        self.project_path = project_path or os.getcwd()

    def get_project_stats(self) -> Dict:
        """Get REAL statistics about the current project."""
        stats = {
            'files': self.count_files(),
            'code_analysis': self.analyze_code(),
            'todos': self.find_todos(),
            'git_stats': self.get_git_stats(),
            'dependencies': self.analyze_dependencies(),
            'project_structure': self.analyze_structure()
        }
        return stats

    def count_files(self) -> Dict:
        """Count actual files in the project."""
        file_stats = {
            'python': 0,
            'javascript': 0,
            'markdown': 0,
            'json': 0,
            'yaml': 0,
            'other': 0,
            'total': 0,
            'total_size_mb': 0
        }

        total_size = 0

        for root, dirs, files in os.walk(self.project_path):
            # Skip hidden directories and common ignore patterns
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv']]

            for file in files:
                if file.startswith('.'):
                    continue

                file_path = Path(root) / file
                file_stats['total'] += 1

                # Get file size
                try:
                    size = file_path.stat().st_size
                    total_size += size
                except:
                    pass

                # Categorize by extension
                ext = file_path.suffix.lower()
                if ext == '.py':
                    file_stats['python'] += 1
                elif ext in ['.js', '.jsx', '.ts', '.tsx']:
                    file_stats['javascript'] += 1
                elif ext == '.md':
                    file_stats['markdown'] += 1
                elif ext == '.json':
                    file_stats['json'] += 1
                elif ext in ['.yml', '.yaml']:
                    file_stats['yaml'] += 1
                else:
                    file_stats['other'] += 1

        file_stats['total_size_mb'] = round(total_size / (1024 * 1024), 2)
        return file_stats

    def analyze_code(self) -> Dict:
        """Analyze actual code in the project."""
        analysis = {
            'total_lines': 0,
            'code_lines': 0,
            'comment_lines': 0,
            'blank_lines': 0,
            'functions': 0,
            'classes': 0,
            'imports': set()
        }

        for root, dirs, files in os.walk(self.project_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv']]

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()

                        for line in lines:
                            analysis['total_lines'] += 1
                            stripped = line.strip()

                            if not stripped:
                                analysis['blank_lines'] += 1
                            elif stripped.startswith('#'):
                                analysis['comment_lines'] += 1
                            else:
                                analysis['code_lines'] += 1

                            # Count functions and classes
                            if stripped.startswith('def '):
                                analysis['functions'] += 1
                            elif stripped.startswith('class '):
                                analysis['classes'] += 1

                            # Track imports
                            if stripped.startswith('import ') or stripped.startswith('from '):
                                import_match = re.match(r'(?:from|import)\s+(\w+)', stripped)
                                if import_match:
                                    analysis['imports'].add(import_match.group(1))
                    except:
                        pass

        analysis['imports'] = len(analysis['imports'])
        return analysis

    def find_todos(self) -> Dict:
        """Find all TODO, FIXME, HACK comments in the project."""
        todos = {
            'TODO': [],
            'FIXME': [],
            'HACK': [],
            'NOTE': [],
            'total': 0
        }

        patterns = {
            'TODO': re.compile(r'#\s*TODO[:|\s](.*)'),
            'FIXME': re.compile(r'#\s*FIXME[:|\s](.*)'),
            'HACK': re.compile(r'#\s*HACK[:|\s](.*)'),
            'NOTE': re.compile(r'#\s*NOTE[:|\s](.*)')
        }

        for root, dirs, files in os.walk(self.project_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv']]

            for file in files:
                if file.endswith(('.py', '.js', '.jsx', '.ts', '.tsx')):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                for todo_type, pattern in patterns.items():
                                    match = pattern.search(line)
                                    if match:
                                        todos[todo_type].append({
                                            'file': str(file_path.relative_to(self.project_path)),
                                            'line': line_num,
                                            'text': match.group(1).strip()
                                        })
                                        todos['total'] += 1
                    except:
                        pass

        return todos

    def get_git_stats(self) -> Dict:
        """Get real git statistics."""
        stats = {
            'initialized': False,
            'branch': 'unknown',
            'commits': 0,
            'authors': [],
            'last_commit': None,
            'uncommitted_changes': 0,
            'untracked_files': 0,
            'branches': []
        }

        try:
            # Check if git repo
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                capture_output=True,
                text=True,
                cwd=self.project_path
            )

            if result.returncode == 0:
                stats['initialized'] = True

                # Current branch
                result = subprocess.run(
                    ['git', 'branch', '--show-current'],
                    capture_output=True,
                    text=True,
                    cwd=self.project_path
                )
                stats['branch'] = result.stdout.strip() or 'detached'

                # Total commits
                result = subprocess.run(
                    ['git', 'rev-list', '--count', 'HEAD'],
                    capture_output=True,
                    text=True,
                    cwd=self.project_path
                )
                if result.returncode == 0:
                    stats['commits'] = int(result.stdout.strip())

                # List of authors
                result = subprocess.run(
                    ['git', 'log', '--format=%an', '--all'],
                    capture_output=True,
                    text=True,
                    cwd=self.project_path
                )
                if result.returncode == 0:
                    authors = set(result.stdout.strip().split('\n'))
                    stats['authors'] = list(authors)[:10]  # Top 10

                # Last commit
                result = subprocess.run(
                    ['git', 'log', '-1', '--format=%h - %s (%ar)'],
                    capture_output=True,
                    text=True,
                    cwd=self.project_path
                )
                if result.returncode == 0:
                    stats['last_commit'] = result.stdout.strip()

                # Uncommitted changes
                result = subprocess.run(
                    ['git', 'diff', '--numstat'],
                    capture_output=True,
                    text=True,
                    cwd=self.project_path
                )
                if result.stdout:
                    stats['uncommitted_changes'] = len(result.stdout.strip().split('\n'))

                # Untracked files
                result = subprocess.run(
                    ['git', 'ls-files', '--others', '--exclude-standard'],
                    capture_output=True,
                    text=True,
                    cwd=self.project_path
                )
                if result.stdout:
                    stats['untracked_files'] = len(result.stdout.strip().split('\n'))

                # All branches
                result = subprocess.run(
                    ['git', 'branch', '-a'],
                    capture_output=True,
                    text=True,
                    cwd=self.project_path
                )
                if result.returncode == 0:
                    branches = [b.strip().replace('* ', '') for b in result.stdout.strip().split('\n')]
                    stats['branches'] = branches[:10]  # Top 10

        except:
            pass

        return stats

    def analyze_dependencies(self) -> Dict:
        """Analyze project dependencies."""
        deps = {
            'python': {'found': False, 'packages': [], 'count': 0},
            'javascript': {'found': False, 'packages': [], 'count': 0},
        }

        # Check for Python dependencies
        requirements_files = ['requirements.txt', 'Pipfile', 'pyproject.toml', 'setup.py']
        for req_file in requirements_files:
            req_path = Path(self.project_path) / req_file
            if req_path.exists():
                deps['python']['found'] = True
                try:
                    with open(req_path, 'r') as f:
                        lines = f.readlines()

                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Extract package name
                            pkg = re.split(r'[=<>!~]', line)[0].strip()
                            if pkg:
                                deps['python']['packages'].append(pkg)
                except:
                    pass

        deps['python']['count'] = len(deps['python']['packages'])
        deps['python']['packages'] = deps['python']['packages'][:10]  # Top 10

        # Check for JavaScript dependencies
        package_json = Path(self.project_path) / 'package.json'
        if package_json.exists():
            deps['javascript']['found'] = True
            try:
                with open(package_json, 'r') as f:
                    data = json.load(f)

                all_deps = []
                if 'dependencies' in data:
                    all_deps.extend(data['dependencies'].keys())
                if 'devDependencies' in data:
                    all_deps.extend(data['devDependencies'].keys())

                deps['javascript']['packages'] = all_deps[:10]  # Top 10
                deps['javascript']['count'] = len(all_deps)
            except:
                pass

        return deps

    def analyze_structure(self) -> Dict:
        """Analyze project structure."""
        structure = {
            'depth': 0,
            'directories': 0,
            'has_tests': False,
            'has_docs': False,
            'has_ci': False,
            'has_docker': False,
            'config_files': [],
            'key_directories': []
        }

        max_depth = 0
        dir_count = 0

        for root, dirs, files in os.walk(self.project_path):
            # Calculate depth
            depth = root[len(self.project_path):].count(os.sep)
            max_depth = max(max_depth, depth)

            # Count directories
            dir_count += len(dirs)

            # Check for key directories
            for d in dirs:
                d_lower = d.lower()
                if 'test' in d_lower:
                    structure['has_tests'] = True
                if d_lower in ['docs', 'documentation']:
                    structure['has_docs'] = True

            # Check for key files
            for f in files:
                f_lower = f.lower()
                if f_lower == 'dockerfile':
                    structure['has_docker'] = True
                if f_lower in ['.travis.yml', '.github', 'jenkinsfile', '.gitlab-ci.yml']:
                    structure['has_ci'] = True
                if f_lower in ['config.json', 'config.yaml', 'config.yml', '.env', 'settings.py']:
                    structure['config_files'].append(f)

        structure['depth'] = max_depth
        structure['directories'] = dir_count

        # Get key directories in root
        root_path = Path(self.project_path)
        key_dirs = []
        for item in root_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                key_dirs.append(item.name)

        structure['key_directories'] = key_dirs[:10]  # Top 10

        return structure