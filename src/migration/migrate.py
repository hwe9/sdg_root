#!/usr/bin/env python3
# src/migration/migrate.py
"""
SDG Database Migration CLI Tool
Usage: python migrate.py [command] [options]
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.migration.migration_manager import migration_manager

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def run_migrations():
    """Run all pending migrations"""
    print("🔄 Running database migrations...")
    
    results = migration_manager.run_all_pending_migrations()
    
    print(f"\n📊 Migration Results:")
    print(f"✅ Applied: {len(results['applied'])}")
    print(f"❌ Failed: {len(results['failed'])}")
    print(f"⏭️  Skipped: {len(results['skipped'])}")
    
    if results['applied']:
        print(f"\n✅ Successfully applied migrations:")
        for migration in results['applied']:
            print(f"  - {migration}")
    
    if results['failed']:
        print(f"\n❌ Failed migrations:")
        for migration in results['failed']:
            print(f"  - {migration}")
        return False
    
    return True

def rollback_migration(migration_name: str):
    """Rollback a specific migration"""
    print(f"🔄 Rolling back migration: {migration_name}")
    
    success = migration_manager.rollback_migration(migration_name)
    
    if success:
        print(f"✅ Successfully rolled back: {migration_name}")
    else:
        print(f"❌ Failed to rollback: {migration_name}")
    
    return success

def validate_schema():
    """Validate database schema"""
    print("🔍 Validating database schema...")
    
    results = migration_manager.validate_database_schema()
    
    print(f"\n📊 Schema Validation Results:")
    print(f"Status: {'✅ VALID' if results['valid'] else '❌ INVALID'}")
    
    if results['issues']:
        print(f"\n❌ Issues found:")
        for issue in results['issues']:
            print(f"  - {issue}")
    
    if results['tables']:
        print(f"\n📋 Table Status:")
        for table, status in results['tables'].items():
            print(f"  - {table}: {status}")
    
    return results['valid']

def list_migrations():
    """List all migrations and their status"""
    print("📋 Migration Status:")
    
    applied_migrations = migration_manager.get_applied_migrations()
    
    # Define all migrations
    all_migrations = [
        "001_consolidate_users"
    ]
    
    for migration in all_migrations:
        status = "✅ Applied" if migration in applied_migrations else "⏳ Pending"
        print(f"  - {migration}: {status}")
    
    print(f"\nTotal: {len(applied_migrations)}/{len(all_migrations)} applied")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="SDG Database Migration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python migrate.py migrate                    # Run all pending migrations
  python migrate.py rollback 001_consolidate_users  # Rollback specific migration
  python migrate.py validate                   # Validate database schema
  python migrate.py list                       # List migration status
        """
    )
    
    parser.add_argument(
        'command',
        choices=['migrate', 'rollback', 'validate', 'list'],
        help='Migration command to execute'
    )
    
    parser.add_argument(
        'migration_name',
        nargs='?',
        help='Migration name (required for rollback command)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without executing (not implemented yet)'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    try:
        if args.command == 'migrate':
            success = run_migrations()
            sys.exit(0 if success else 1)
        
        elif args.command == 'rollback':
            if not args.migration_name:
                print("❌ Error: Migration name is required for rollback command")
                sys.exit(1)
            success = rollback_migration(args.migration_name)
            sys.exit(0 if success else 1)
        
        elif args.command == 'validate':
            success = validate_schema()
            sys.exit(0 if success else 1)
        
        elif args.command == 'list':
            list_migrations()
            sys.exit(0)
    
    except KeyboardInterrupt:
        print("\n⚠️  Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

