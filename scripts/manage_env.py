#!/usr/bin/env python3
# scripts/manage_env.py
"""
SDG Environment Variable Management CLI Tool
Usage: python manage_env.py [command] [options]
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.env_manager import env_manager

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def validate_environment():
    """Validate current environment configuration"""
    print("🔍 Validating environment configuration...")
    
    try:
        env_manager._validate_environment()
        print("✅ Environment validation passed!")
        return True
    except ValueError as e:
        print(f"❌ Environment validation failed: {e}")
        return False

def show_environment():
    """Show current environment variables"""
    print("📋 Current Environment Variables:")
    print("")
    
    env_vars = env_manager.get_all_env_vars()
    
    # Group by category
    categories = {
        "Database": ["DATABASE_URL", "DB_HOST", "DB_PORT", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"],
        "Redis": ["REDIS_URL", "REDIS_PASSWORD"],
        "Weaviate": ["WEAVIATE_URL", "WEAVIATE_API_KEY", "WEAVIATE_TRANSFORMER_URL"],
        "Security": ["SECRET_KEY", "SECRET_KEY_ENCRYPTED", "ENCRYPTION_SALT"],
        "Service Configuration": ["ENVIRONMENT", "LOG_LEVEL", "DEBUG", "ALLOWED_ORIGINS"],
        "Service URLs": ["API_SERVICE_URL", "AUTH_SERVICE_URL", "DATA_PROCESSING_URL", "DATA_RETRIEVAL_URL", "VECTORIZATION_SERVICE_URL", "CONTENT_EXTRACTION_URL"],
        "Dependency Management": ["DEPENDENCY_MANAGER_ENABLED", "HEALTH_CHECK_INTERVAL", "MAX_STARTUP_TIME"],
        "Directories": ["CONFIG_DIR", "SECRET_STORE_DIR", "DATA_ROOT", "RAW_DATA_DIR", "PROCESSED_DATA_DIR", "IMAGES_DIR"],
        "PgAdmin": ["PGADMIN_DEFAULT_EMAIL", "PGADMIN_DEFAULT_PASSWORD"],
    }
    
    for category, var_names in categories.items():
        print(f"## {category}")
        for var_name in var_names:
            if var_name in env_vars:
                value = env_vars[var_name]
                if "password" in var_name.lower() or "secret" in var_name.lower():
                    # Mask sensitive values
                    display_value = "***" if value and not value.startswith("ERROR:") else value
                else:
                    display_value = value
                print(f"  {var_name}={display_value}")
        print("")

def generate_template():
    """Generate environment template file"""
    print("📝 Generating environment template...")
    
    template_content = env_manager.generate_env_template()
    template_file = Path(__file__).parent.parent / ".env.template"
    
    template_file.write_text(template_content)
    print(f"✅ Generated environment template: {template_file}")
    return True

def check_missing_vars():
    """Check for missing required environment variables"""
    print("🔍 Checking for missing required environment variables...")
    
    missing_vars = []
    
    for name, config in env_manager.env_configs.items():
        if config.required:
            try:
                value = env_manager.get_env_var(name)
                if not value:
                    missing_vars.append(name)
            except ValueError:
                missing_vars.append(name)
    
    if missing_vars:
        print(f"❌ Missing required environment variables:")
        for var in missing_vars:
            config = env_manager.env_configs[var]
            print(f"  - {var}: {config.description}")
        return False
    else:
        print("✅ All required environment variables are set!")
        return True

def show_env_info():
    """Show environment variable information"""
    print("📋 Environment Variable Information:")
    print("")
    
    for name, config in env_manager.env_configs.items():
        print(f"## {name}")
        print(f"  Description: {config.description}")
        print(f"  Required: {'Yes' if config.required else 'No'}")
        if config.default:
            print(f"  Default: {config.default}")
        print("")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="SDG Environment Variable Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_env.py validate                    # Validate environment configuration
  python manage_env.py show                        # Show current environment variables
  python manage_env.py generate-template           # Generate .env.template file
  python manage_env.py check-missing               # Check for missing required variables
  python manage_env.py info                        # Show environment variable information
        """
    )
    
    parser.add_argument(
        'command',
        choices=['validate', 'show', 'generate-template', 'check-missing', 'info'],
        help='Environment management command to execute'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    try:
        if args.command == 'validate':
            success = validate_environment()
            sys.exit(0 if success else 1)
        
        elif args.command == 'show':
            show_environment()
            sys.exit(0)
        
        elif args.command == 'generate-template':
            success = generate_template()
            sys.exit(0 if success else 1)
        
        elif args.command == 'check-missing':
            success = check_missing_vars()
            sys.exit(0 if success else 1)
        
        elif args.command == 'info':
            show_env_info()
            sys.exit(0)
    
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

