#!/usr/bin/env python3
# scripts/manage_docker.py
"""
SDG Docker Configuration Management CLI Tool
Usage: python manage_docker.py [command] [options]
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.docker_config import docker_config_manager

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def generate_compose():
    """Generate docker-compose.yml"""
    print("📝 Generating docker-compose.yml...")
    
    content = docker_config_manager.generate_docker_compose()
    compose_file = Path(__file__).parent.parent / "docker-compose.yml"
    
    # Backup existing file
    if compose_file.exists():
        backup_file = compose_file.with_suffix(".yml.backup")
        compose_file.rename(backup_file)
        print(f"Backed up existing file to {backup_file}")
    
    compose_file.write_text(content)
    print(f"✅ Generated docker-compose.yml")
    return True

def validate_config():
    """Validate Docker configuration"""
    print("🔍 Validating Docker configuration...")
    
    issues = docker_config_manager.validate_docker_config()
    
    if not issues:
        print("✅ Docker configuration is valid!")
        return True
    
    print(f"\n❌ Found issues in {len(issues)} services:")
    for service_name, service_issues in issues.items():
        print(f"\n  {service_name}:")
        for issue in service_issues:
            print(f"    - {issue}")
    
    return False

def show_services():
    """Show all Docker services"""
    print("📋 Docker Services:")
    print("")
    
    for service_name, config in docker_config_manager.services.items():
        print(f"## {service_name}")
        print(f"  Build Context: {config.build_context}")
        print(f"  Dockerfile: {config.dockerfile}")
        print(f"  Ports: {', '.join(config.ports) if config.ports else 'None'}")
        print(f"  Depends On: {', '.join(config.depends_on) if config.depends_on else 'None'}")
        print(f"  Networks: {', '.join(config.networks) if config.networks else 'None'}")
        print(f"  Restart Policy: {config.restart}")
        if config.healthcheck:
            print(f"  Health Check: Enabled")
        if config.deploy:
            print(f"  Deploy: Configured")
        print("")

def show_dependencies():
    """Show service dependency graph"""
    print("📊 Service Dependencies:")
    print("")
    
    for service_name, config in docker_config_manager.services.items():
        if config.depends_on:
            print(f"{service_name}:")
            for dep in config.depends_on:
                print(f"  └── {dep}")
            print("")
        else:
            print(f"{service_name}: (no dependencies)")
            print("")

def check_consistency():
    """Check Docker configuration consistency"""
    print("🔍 Checking Docker configuration consistency...")
    
    issues = []
    
    # Check for circular dependencies
    for service_name, config in docker_config_manager.services.items():
        for dep in config.depends_on:
            if dep in docker_config_manager.services:
                dep_config = docker_config_manager.services[dep]
                if service_name in dep_config.depends_on:
                    issues.append(f"Circular dependency: {service_name} <-> {dep}")
    
    # Check for missing services
    all_services = set(docker_config_manager.services.keys())
    referenced_services = set()
    
    for config in docker_config_manager.services.values():
        referenced_services.update(config.depends_on)
    
    missing_services = referenced_services - all_services
    if missing_services:
        issues.append(f"Missing services: {', '.join(missing_services)}")
    
    # Check for port conflicts
    port_usage = {}
    for service_name, config in docker_config_manager.services.items():
        for port in config.ports:
            if ':' in port:
                host_port = port.split(':')[0]
                if host_port in port_usage:
                    issues.append(f"Port conflict: {host_port} used by {port_usage[host_port]} and {service_name}")
                port_usage[host_port] = service_name
    
    if issues:
        print(f"\n❌ Found {len(issues)} consistency issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Docker configuration is consistent!")
        return True

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="SDG Docker Configuration Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_docker.py generate-compose          # Generate docker-compose.yml
  python manage_docker.py validate                  # Validate Docker configuration
  python manage_docker.py show-services             # Show all services
  python manage_docker.py show-dependencies          # Show dependency graph
  python manage_docker.py check-consistency         # Check configuration consistency
        """
    )
    
    parser.add_argument(
        'command',
        choices=['generate-compose', 'validate', 'show-services', 'show-dependencies', 'check-consistency'],
        help='Docker management command to execute'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    try:
        if args.command == 'generate-compose':
            success = generate_compose()
            sys.exit(0 if success else 1)
        
        elif args.command == 'validate':
            success = validate_config()
            sys.exit(0 if success else 1)
        
        elif args.command == 'show-services':
            show_services()
            sys.exit(0)
        
        elif args.command == 'show-dependencies':
            show_dependencies()
            sys.exit(0)
        
        elif args.command == 'check-consistency':
            success = check_consistency()
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

