#!/usr/bin/env python3
# scripts/migrate_logging.py
"""
Migrate all services to use centralized logging
"""

import os
import re
from pathlib import Path

def update_service_logging(service_path: Path, service_name: str):
    """Update a service to use centralized logging"""
    main_file = service_path / "main.py"

    if not main_file.exists():
        return False

    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already using centralized logging
    if 'from ..core.logging_config import get_logger' in content:
        print(f"✅ {service_name}: Already using centralized logging")
        return True

    # Find logging imports and logger initialization
    lines = content.split('\n')
    updated_lines = []
    logging_import_added = False
    logger_replaced = False

    for line in lines:
        # Replace logging.basicConfig and logger initialization
        if line.strip().startswith('logging.basicConfig('):
            # Remove basicConfig line
            continue
        elif re.match(r'\s*logger\s*=\s*logging\.getLogger\(.*\)', line):
            # Replace with centralized logger
            if not logging_import_added:
                # Find a good place to add the import
                # Look for the last core import
                pass
            updated_lines.append(f'logger = get_logger("{service_name}")')
            logger_replaced = True
        else:
            updated_lines.append(line)

    # Add the import if we replaced a logger
    if logger_replaced and 'from ..core.logging_config import get_logger' not in content:
        # Find where to insert the import (after other core imports)
        insert_index = -1
        for i, line in enumerate(updated_lines):
            if line.startswith('from ..core.') and 'logging_config' not in line:
                insert_index = i + 1

        if insert_index > 0:
            updated_lines.insert(insert_index, 'from ..core.logging_config import get_logger')
        else:
            # Fallback: add after the last import
            for i in range(len(updated_lines) - 1, -1, -1):
                if updated_lines[i].startswith(('import ', 'from ')):
                    updated_lines.insert(i + 1, 'from ..core.logging_config import get_logger')
                    break

    # Write back the file
    new_content = '\n'.join(updated_lines)
    if new_content != content:
        with open(main_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✅ {service_name}: Updated to use centralized logging")
        return True
    else:
        print(f"ℹ️  {service_name}: No changes needed")
        return True

def main():
    """Main migration function"""
    print("🔄 Migrating services to use centralized logging...")

    base_dir = Path(__file__).parent.parent / "src"
    services = {
        "api": base_dir / "api",
        "auth": base_dir / "auth",
        "data_processing": base_dir / "data_processing",
        "data_retrieval": base_dir / "data_retrieval",
        "vectorization": base_dir / "vectorization",
        "content_extraction": base_dir / "content_extraction"
    }

    success_count = 0
    total_count = len(services)

    for service_name, service_path in services.items():
        try:
            if update_service_logging(service_path, service_name):
                success_count += 1
        except Exception as e:
            print(f"❌ {service_name}: Failed to update - {e}")

    print(f"\n📊 Migration Results: {success_count}/{total_count} services updated")
    return success_count == total_count

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
