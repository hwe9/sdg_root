#!/usr/bin/env python3
# scripts/manage_logging.py
"""
SDG Logging Management CLI Tool
Usage: python manage_logging.py [command] [options]
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logging_config import get_logger, loggers

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    )

def show_logging_config():
    """Show current logging configuration"""
    print("📋 Current Logging Configuration:")
    print("")

    for service_name, logger in loggers.items():
        print(f"## {service_name}")
        print(f"  Configured: {'Yes' if logger._configured else 'No'}")
        if hasattr(logger.logger, 'level'):
            level_name = logging.getLevelName(logger.logger.level)
            print(f"  Level: {level_name}")
        print(f"  Handlers: {len(logger.logger.handlers)}")

        for i, handler in enumerate(logger.logger.handlers):
            handler_type = type(handler).__name__
            if hasattr(handler, 'baseFilename'):
                print(f"    Handler {i+1}: {handler_type} -> {handler.baseFilename}")
            else:
                print(f"    Handler {i+1}: {handler_type}")
        print("")

def test_logging():
    """Test logging functionality"""
    print("🧪 Testing Logging Functionality...")
    print("")

    test_messages = [
        ("DEBUG", "This is a debug message"),
        ("INFO", "This is an info message"),
        ("WARNING", "This is a warning message"),
        ("ERROR", "This is an error message"),
    ]

    for service_name in ["api", "auth", "core"]:
        logger = get_logger(service_name)
        print(f"Testing {service_name} logger:")

        for level_name, message in test_messages:
            level = getattr(logging, level_name)
            if logger.isEnabledFor(level):
                print(f"  [{level_name}] {message}")

                # Actually log the message
                if level_name == "DEBUG":
                    logger.debug(message)
                elif level_name == "INFO":
                    logger.info(message)
                elif level_name == "WARNING":
                    logger.warning(message)
                elif level_name == "ERROR":
                    logger.error(message)
        print("")

def check_logging_consistency():
    """Check logging consistency across services"""
    print("🔍 Checking Logging Consistency...")
    print("")

    issues = []

    # Check that all services have loggers
    expected_services = ["api", "auth", "data_processing", "data_retrieval", "vectorization", "content_extraction", "core"]

    for service in expected_services:
        if service not in loggers:
            issues.append(f"Missing logger for service: {service}")

    # Check configuration consistency
    configured_loggers = [name for name, logger in loggers.items() if logger._configured]
    if len(set(configured_loggers)) != len(configured_loggers):
        issues.append("Duplicate configured loggers found")

    # Check handler consistency
    handler_counts = {}
    for name, logger in loggers.items():
        handler_count = len(logger.logger.handlers)
        handler_counts[name] = handler_count

    min_handlers = min(handler_counts.values()) if handler_counts else 0
    max_handlers = max(handler_counts.values()) if handler_counts else 0

    if min_handlers != max_handlers:
        issues.append(f"Inconsistent handler counts: {handler_counts}")

    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Logging configuration is consistent!")
        return True

def update_service_logging(service_name: str, log_level: str = None, log_to_file: bool = None, log_dir: str = None):
    """Update logging configuration for a specific service"""
    if service_name not in loggers:
        print(f"❌ Unknown service: {service_name}")
        return False

    logger = loggers[service_name]

    # Get current configuration
    current_level = os.getenv("LOG_LEVEL", "INFO")
    current_log_to_file = os.getenv("LOG_TO_FILE", "true").lower() == "true"
    current_log_dir = os.getenv("LOG_DIR", "/var/log/sdg")

    # Update with provided values
    if log_level:
        current_level = log_level
    if log_to_file is not None:
        current_log_to_file = log_to_file
    if log_dir:
        current_log_dir = log_dir

    # Reconfigure logger
    logger._configured = False  # Force reconfiguration
    logger.configure(
        level=current_level,
        log_to_file=current_log_to_file,
        log_dir=current_log_dir
    )

    print(f"✅ Updated logging for {service_name}:")
    print(f"  Level: {current_level}")
    print(f"  Log to file: {current_log_to_file}")
    print(f"  Log directory: {current_log_dir}")

    return True

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="SDG Logging Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_logging.py show                    # Show current logging configuration
  python manage_logging.py test                    # Test logging functionality
  python manage_logging.py check                   # Check logging consistency
  python manage_logging.py update api --level DEBUG  # Update API service logging
        """
    )

    parser.add_argument(
        'command',
        choices=['show', 'test', 'check', 'update'],
        help='Logging management command to execute'
    )

    parser.add_argument(
        'service',
        nargs='?',
        help='Service name (required for update command)'
    )

    parser.add_argument(
        '--level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Log level'
    )

    parser.add_argument(
        '--log-to-file',
        action='store_true',
        default=None,
        help='Enable file logging'
    )

    parser.add_argument(
        '--no-log-to-file',
        action='store_true',
        help='Disable file logging'
    )

    parser.add_argument(
        '--log-dir',
        help='Log directory path'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Handle log-to-file flag
    log_to_file = None
    if args.log_to_file:
        log_to_file = True
    elif args.no_log_to_file:
        log_to_file = False

    try:
        if args.command == 'show':
            show_logging_config()

        elif args.command == 'test':
            test_logging()

        elif args.command == 'check':
            success = check_logging_consistency()
            sys.exit(0 if success else 1)

        elif args.command == 'update':
            if not args.service:
                print("❌ Service name is required for update command")
                sys.exit(1)

            success = update_service_logging(
                args.service,
                log_level=args.level,
                log_to_file=log_to_file,
                log_dir=args.log_dir
            )
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
