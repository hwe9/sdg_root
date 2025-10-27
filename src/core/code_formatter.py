# src/core/code_formatter.py
"""
Centralized code formatting and linting for SDG project
This module ensures consistent code formatting across all services
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import List
from typing import Dict
from typing import Any
from typing import Optional
from typing import Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FormatResult:
    """Result of formatting operation"""
    file_path: str
    success: bool
    changes_made: bool = False
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class CodeFormatter:
    """Centralized code formatting and linting"""

    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.python_files = self._find_python_files()
        self.config_files = self._create_config_files()

    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the project"""
        python_files = []

        # Service directories
        service_dirs = [
            "src/api",
            "src/auth",
            "src/data_processing",
            "src/data_retrieval",
            "src/vectorization",
            "src/content_extraction",
            "src/core"
        ]

        for service_dir in service_dirs:
            service_path = self.base_dir / service_dir
            if service_path.exists():
                for file_path in service_path.rglob("*.py"):
                    if not any(part.startswith('.') for part in file_path.parts):
                        python_files.append(file_path)

        return python_files

    def _create_config_files(self) -> Dict[str, str]:
        """Create configuration files for formatting tools"""
        configs = {}

        # Black configuration
        configs["pyproject.toml"] = """[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']
include = '\.pyi?$'
extend-exclude = '''
/(
    \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["src"]
"""

        # Flake8 configuration
        configs[".flake8"] = """[flake8]
max-line-length = 88
extend-ignore = E203, W503, E501
exclude =
    .git,
    __pycache__,
    .venv,
    .tox,
    build,
    dist,
    *.egg-info
"""

        # MyPy configuration
        configs["mypy.ini"] = """[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = False
disallow_incomplete_defs = False
check_untyped_defs = True
disallow_untyped_decorators = False
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True
show_error_codes = True

[mypy-tests.*]
ignore_errors = True

[mypy-src.core.*]
disallow_untyped_defs = True
"""

        return configs

    def format_python_files(self, dry_run: bool = False) -> List[FormatResult]:
        """Format all Python files using black and isort"""
        results = []

        logger.info(f"Formatting {len(self.python_files)} Python files...")

        # Check if tools are available
        tools_available = self._check_tools_available()

        if not tools_available:
            logger.warning("Code formatting tools (black, isort) not found. Using manual formatting.")
            return self._manual_format_python_files(dry_run)

        for file_path in self.python_files:
            try:
                # Run isort first
                isort_result = subprocess.run(
                    ["isort", str(file_path), "--check-only", "--diff"],
                    capture_output=True,
                    text=True,
                    cwd=self.base_dir
                )

                # Run black
                black_result = subprocess.run(
                    ["black", str(file_path), "--check", "--diff"],
                    capture_output=True,
                    text=True,
                    cwd=self.base_dir
                )

                changes_needed = isort_result.returncode != 0 or black_result.returncode != 0

                if not dry_run and changes_needed:
                    # Apply formatting
                    subprocess.run(["isort", str(file_path)], cwd=self.base_dir, check=True)
                    subprocess.run(["black", str(file_path)], cwd=self.base_dir, check=True)

                results.append(FormatResult(
                    file_path=str(file_path.relative_to(self.base_dir)),
                    success=True,
                    changes_made=changes_needed
                ))

            except subprocess.CalledProcessError as e:
                results.append(FormatResult(
                    file_path=str(file_path.relative_to(self.base_dir)),
                    success=False,
                    errors=[f"Formatting failed: {e}"]
                ))
            except Exception as e:
                results.append(FormatResult(
                    file_path=str(file_path.relative_to(self.base_dir)),
                    success=False,
                    errors=[f"Unexpected error: {e}"]
                ))

        return results

    def _check_tools_available(self) -> bool:
        """Check if formatting tools are available"""
        try:
            subprocess.run(["black", "--version"], capture_output=True, check=True)
            subprocess.run(["isort", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _manual_format_python_files(self, dry_run: bool = False) -> List[FormatResult]:
        """Manual formatting when tools are not available"""
        results = []

        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()

                formatted_content = self._apply_manual_formatting(original_content)

                changes_made = formatted_content != original_content

                if not dry_run and changes_made:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(formatted_content)

                results.append(FormatResult(
                    file_path=str(file_path.relative_to(self.base_dir)),
                    success=True,
                    changes_made=changes_made
                ))

            except Exception as e:
                results.append(FormatResult(
                    file_path=str(file_path.relative_to(self.base_dir)),
                    success=False,
                    errors=[f"Manual formatting failed: {e}"]
                ))

        return results

    def _apply_manual_formatting(self, content: str) -> str:
        """Apply basic manual formatting rules"""
        lines = content.split('\n')
        formatted_lines = []

        for line in lines:
            # Fix multiple imports on one line (basic fix)
            if line.strip().startswith('import ') and ',' in line and 'from' not in line:
                # Split multiple imports
                parts = line.split('import ')
                if len(parts) == 2:
                    base_import = parts[0] + 'import '
                    imports = [imp.strip() for imp in parts[1].split(',')]
                    for imp in imports:
                        formatted_lines.append(base_import + imp)
                    continue

            # Fix multiple imports in from statements
            elif line.strip().startswith('from ') and 'import ' in line and ',' in line.split('import ')[1]:
                parts = line.split('import ')
                if len(parts) == 2:
                    from_part = parts[0] + 'import '
                    imports = [imp.strip() for imp in parts[1].split(',')]
                    for imp in imports:
                        formatted_lines.append(from_part + imp)
                    continue

            formatted_lines.append(line)

        return '\n'.join(formatted_lines)

    def lint_python_files(self) -> List[FormatResult]:
        """Lint all Python files using flake8 and mypy"""
        results = []

        logger.info(f"Linting {len(self.python_files)} Python files...")

        for file_path in self.python_files:
            errors = []

            try:
                # Run flake8
                flake8_result = subprocess.run(
                    ["flake8", str(file_path)],
                    capture_output=True,
                    text=True,
                    cwd=self.base_dir
                )

                if flake8_result.returncode != 0:
                    errors.extend(flake8_result.stdout.strip().split('\n'))

                # Run mypy (with timeout)
                try:
                    mypy_result = subprocess.run(
                        ["mypy", str(file_path), "--no-error-summary"],
                        capture_output=True,
                        text=True,
                        cwd=self.base_dir,
                        timeout=30  # 30 second timeout per file
                    )

                    if mypy_result.returncode != 0:
                        errors.extend(mypy_result.stdout.strip().split('\n'))

                except subprocess.TimeoutExpired:
                    errors.append("MyPy timeout (30s)")

            except Exception as e:
                errors.append(f"Linting error: {e}")

            results.append(FormatResult(
                file_path=str(file_path.relative_to(self.base_dir)),
                success=len(errors) == 0,
                errors=errors if errors else []
            ))

        return results

    def check_imports_consistency(self) -> List[FormatResult]:
        """Check for import statement consistency"""
        results = []

        for file_path in self.python_files:
            errors = []

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                lines = content.split('\n')
                imports_section = []
                in_imports = False

                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('import ') or stripped.startswith('from '):
                        in_imports = True
                        imports_section.append(line)
                    elif in_imports and (stripped == '' or stripped.startswith('#')):
                        continue  # Allow empty lines and comments in import section
                    elif in_imports and stripped:
                        break  # End of imports section

                # Check for issues
                if imports_section:
                    # Check for multiple imports on one line
                    for line in imports_section:
                        if ',' in line and 'import' in line and 'from' not in line:
                            errors.append(f"Multiple imports on one line: {line.strip()}")

                    # Check for unsorted imports (basic check)
                    import_lines = [line for line in imports_section if not line.strip().startswith('#')]
                    if len(import_lines) > 1:
                        # Simple check: stdlib imports should come before third-party
                        stdlib_imports = []
                        third_party_imports = []

                        for line in import_lines:
                            if line.strip().startswith('import ') and not line.strip().startswith('import src'):
                                stdlib_imports.append(line)
                            else:
                                third_party_imports.append(line)

                        if third_party_imports and stdlib_imports and third_party_imports[0] in stdlib_imports:
                            errors.append("Third-party imports should come after standard library imports")

            except Exception as e:
                errors.append(f"Import check error: {e}")

            results.append(FormatResult(
                file_path=str(file_path.relative_to(self.base_dir)),
                success=len(errors) == 0,
                errors=errors if errors else []
            ))

        return results

    def generate_config_files(self) -> List[str]:
        """Generate configuration files for formatting tools"""
        generated = []

        for config_file, content in self.config_files.items():
            config_path = self.base_dir / config_file
            try:
                if config_path.exists():
                    # Backup existing file
                    backup_path = config_path.with_suffix(config_path.suffix + '.backup')
                    config_path.rename(backup_path)
                    logger.info(f"Backed up {config_file} to {backup_path}")

                config_path.write_text(content)
                generated.append(config_file)
                logger.info(f"Generated {config_file}")

            except Exception as e:
                logger.error(f"Failed to generate {config_file}: {e}")

        return generated

    def create_pre_commit_hooks(self) -> bool:
        """Create pre-commit hooks for consistent formatting"""
        try:
            hooks_dir = self.base_dir / ".git" / "hooks"
            if not hooks_dir.exists():
                logger.warning("Git hooks directory not found, skipping pre-commit setup")
                return False

            pre_commit_hook = hooks_dir / "pre-commit"

            hook_content = """#!/bin/bash
# SDG Project Pre-commit Hook
# Ensures consistent code formatting before commits

echo "🔍 Running SDG code formatting checks..."

# Check if formatting tools are available
if ! command -v black &> /dev/null; then
    echo "❌ Black not found. Install with: pip install black"
    exit 1
fi

if ! command -v isort &> /dev/null; then
    echo "❌ isort not found. Install with: pip install isort"
    exit 1
fi

if ! command -v flake8 &> /dev/null; then
    echo "❌ flake8 not found. Install with: pip install flake8"
    exit 1
fi

# Get list of staged Python files
STAGED_PY_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$')

if [ -z "$STAGED_PY_FILES" ]; then
    echo "✅ No Python files to check"
    exit 0
fi

echo "📝 Formatting staged Python files..."
echo "$STAGED_PY_FILES" | while read -r file; do
    if [ -f "$file" ]; then
        echo "  Formatting $file"
        isort "$file" || exit 1
        black "$file" || exit 1
        git add "$file"
    fi
done

echo "🔍 Running lint checks..."
echo "$STAGED_PY_FILES" | while read -r file; do
    if [ -f "$file" ]; then
        echo "  Linting $file"
        flake8 "$file" || exit 1
    fi
done

echo "✅ All formatting and linting checks passed!"
"""

            pre_commit_hook.write_text(hook_content)
            pre_commit_hook.chmod(0o755)

            logger.info("Created pre-commit hook")
            return True

        except Exception as e:
            logger.error(f"Failed to create pre-commit hook: {e}")
            return False

# Global code formatter instance
code_formatter = CodeFormatter()
