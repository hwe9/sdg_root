#!/usr/bin/env python3
# scripts/format_code.py
"""
SDG Code Formatting and Linting CLI Tool
Usage: python format_code.py [command] [options]
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.code_formatter import code_formatter

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def format_code(dry_run: bool = False):
    """Format all Python code"""
    print("🔧 Formatting Python code...")

    results = code_formatter.format_python_files(dry_run=dry_run)

    success_count = sum(1 for r in results if r.success)
    changed_count = sum(1 for r in results if r.changes_made)

    print(f"\n📊 Formatting Results:")
    print(f"  Files processed: {len(results)}")
    print(f"  Successfully formatted: {success_count}")
    print(f"  Files with changes: {changed_count}")

    if any(not r.success for r in results):
        print(f"\n❌ Failed files:")
        for result in results:
            if not result.success:
                print(f"  - {result.file_path}: {', '.join(result.errors)}")

    return success_count == len(results)

def lint_code():
    """Lint all Python code"""
    print("🔍 Linting Python code...")

    results = code_formatter.lint_python_files()

    success_count = sum(1 for r in results if r.success)

    print(f"\n📊 Linting Results:")
    print(f"  Files processed: {len(results)}")
    print(f"  Files with no issues: {success_count}")

    if success_count < len(results):
        print(f"\n⚠️  Files with issues:")
        for result in results:
            if not result.success and result.errors:
                print(f"  - {result.file_path}:")
                for error in result.errors[:5]:  # Show first 5 errors
                    print(f"    {error}")
                if len(result.errors) > 5:
                    print(f"    ... and {len(result.errors) - 5} more")

    return success_count == len(results)

def check_imports():
    """Check import statement consistency"""
    print("📋 Checking import consistency...")

    results = code_formatter.check_imports_consistency()

    success_count = sum(1 for r in results if r.success)

    print(f"\n📊 Import Check Results:")
    print(f"  Files processed: {len(results)}")
    print(f"  Files with consistent imports: {success_count}")

    if success_count < len(results):
        print(f"\n⚠️  Files with import issues:")
        for result in results:
            if not result.success and result.errors:
                print(f"  - {result.file_path}:")
                for error in result.errors:
                    print(f"    {error}")

    return success_count == len(results)

def generate_configs():
    """Generate configuration files"""
    print("📝 Generating configuration files...")

    generated = code_formatter.generate_config_files()

    print(f"\n📊 Configuration Generation Results:")
    print(f"  Files generated: {len(generated)}")

    if generated:
        print(f"\n✅ Generated files:")
        for file in generated:
            print(f"  - {file}")

    return len(generated) == len(code_formatter.config_files)

def setup_hooks():
    """Setup pre-commit hooks"""
    print("🔗 Setting up pre-commit hooks...")

    success = code_formatter.create_pre_commit_hooks()

    if success:
        print("✅ Pre-commit hook created successfully")
        print("  Hook location: .git/hooks/pre-commit")
    else:
        print("❌ Failed to create pre-commit hook")

    return success

def run_all_checks():
    """Run all formatting and linting checks"""
    print("🚀 Running all code quality checks...")

    checks = [
        ("Import consistency", check_imports),
        ("Linting", lint_code),
        ("Formatting", lambda: format_code(dry_run=True)),
    ]

    results = []
    for name, check_func in checks:
        print(f"\n--- {name} ---")
        try:
            success = check_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ Error running {name}: {e}")
            results.append((name, False))

    print(f"\n📊 Overall Results:")
    all_passed = True
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {name}: {status}")
        if not success:
            all_passed = False

    if all_passed:
        print(f"\n🎉 All checks passed!")
    else:
        print(f"\n⚠️  Some checks failed. Run individual commands for details.")

    return all_passed

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="SDG Code Formatting and Linting Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python format_code.py format                # Format all Python code
  python format_code.py format --dry-run      # Check formatting without changes
  python format_code.py lint                  # Lint all Python code
  python format_code.py check-imports         # Check import consistency
  python format_code.py generate-configs      # Generate config files
  python format_code.py setup-hooks           # Setup pre-commit hooks
  python format_code.py check-all             # Run all checks
        """
    )

    parser.add_argument(
        'command',
        choices=['format', 'lint', 'check-imports', 'generate-configs', 'setup-hooks', 'check-all'],
        help='Code formatting command to execute'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Check formatting without making changes (for format command)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    try:
        if args.command == 'format':
            success = format_code(dry_run=args.dry_run)
            sys.exit(0 if success else 1)

        elif args.command == 'lint':
            success = lint_code()
            sys.exit(0 if success else 1)

        elif args.command == 'check-imports':
            success = check_imports()
            sys.exit(0 if success else 1)

        elif args.command == 'generate-configs':
            success = generate_configs()
            sys.exit(0 if success else 1)

        elif args.command == 'setup-hooks':
            success = setup_hooks()
            sys.exit(0 if success else 1)

        elif args.command == 'check-all':
            success = run_all_checks()
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
