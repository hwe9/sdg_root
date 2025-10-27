#!/usr/bin/env python3
# scripts/manage_requirements.py
"""
SDG Requirements Management CLI Tool
Usage: python manage_requirements.py [command] [options]
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.requirements_manager import requirements_manager

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def update_requirements():
    """Update all requirements.txt files"""
    print("🔄 Updating requirements files...")
    
    results = requirements_manager.update_all_requirements()
    
    print(f"\n📊 Update Results:")
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    for service_name, success in results.items():
        status = "✅ Updated" if success else "❌ Failed"
        print(f"  - {service_name}: {status}")
    
    print(f"\nSummary: {success_count}/{total_count} services updated successfully")
    return success_count == total_count

def validate_requirements():
    """Validate requirements consistency"""
    print("🔍 Validating requirements consistency...")
    
    issues = requirements_manager.validate_requirements()
    
    if not issues:
        print("✅ All requirements files are consistent!")
        return True
    
    print(f"\n❌ Found issues in {len(issues)} services:")
    for service_name, service_issues in issues.items():
        print(f"\n  {service_name}:")
        for issue in service_issues:
            print(f"    - {issue}")
    
    return False

def show_dependencies():
    """Show all core dependencies"""
    print("📋 Core Dependencies:")
    print("")
    
    for dep_name, dep in requirements_manager.core_dependencies.items():
        print(f"  {dep.name}=={dep.version}")
        if dep.comment:
            print(f"    # {dep.comment}")
        print("")

def generate_base_requirements():
    """Generate base requirements.txt for project root"""
    print("📝 Generating base requirements.txt...")
    
    content = requirements_manager.generate_base_requirements()
    base_file = Path(__file__).parent.parent / "requirements.txt"
    
    # Backup existing file
    if base_file.exists():
        backup_file = base_file.with_suffix(".txt.backup")
        base_file.rename(backup_file)
        print(f"Backed up existing file to {backup_file}")
    
    base_file.write_text(content)
    print(f"✅ Generated base requirements.txt")
    return True

def check_version_conflicts():
    """Check for version conflicts across services"""
    print("🔍 Checking for version conflicts...")
    
    conflicts = {}
    
    # Collect all dependencies from all services
    all_deps = {}
    
    for service_name in requirements_manager.service_dependencies.keys():
        requirements_file = Path(__file__).parent.parent / "src" / service_name / "requirements.txt"
        
        if not requirements_file.exists():
            continue
        
        try:
            content = requirements_file.read_text()
            lines = content.split('\n')
            
            for line in lines:
                if '==' in line and not line.strip().startswith('#'):
                    dep_name = line.split('==')[0].strip()
                    version = line.split('==')[1].split()[0].strip()
                    
                    if dep_name not in all_deps:
                        all_deps[dep_name] = {}
                    
                    all_deps[dep_name][service_name] = version
        
        except Exception as e:
            print(f"Error reading {requirements_file}: {e}")
    
    # Check for conflicts
    for dep_name, versions in all_deps.items():
        unique_versions = set(versions.values())
        if len(unique_versions) > 1:
            conflicts[dep_name] = versions
    
    if conflicts:
        print(f"\n❌ Found version conflicts:")
        for dep_name, versions in conflicts.items():
            print(f"\n  {dep_name}:")
            for service, version in versions.items():
                print(f"    - {service}: {version}")
        return False
    else:
        print("✅ No version conflicts found!")
        return True

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="SDG Requirements Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_requirements.py update                    # Update all requirements files
  python manage_requirements.py validate                  # Validate requirements consistency
  python manage_requirements.py show-deps                # Show core dependencies
  python manage_requirements.py generate-base            # Generate base requirements.txt
  python manage_requirements.py check-conflicts          # Check for version conflicts
        """
    )
    
    parser.add_argument(
        'command',
        choices=['update', 'validate', 'show-deps', 'generate-base', 'check-conflicts'],
        help='Requirements management command to execute'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    try:
        if args.command == 'update':
            success = update_requirements()
            sys.exit(0 if success else 1)
        
        elif args.command == 'validate':
            success = validate_requirements()
            sys.exit(0 if success else 1)
        
        elif args.command == 'show-deps':
            show_dependencies()
            sys.exit(0)
        
        elif args.command == 'generate-base':
            success = generate_base_requirements()
            sys.exit(0 if success else 1)
        
        elif args.command == 'check-conflicts':
            success = check_version_conflicts()
            sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

