"""
Prepare Open Source Distribution

This script prepares the Papr Memory codebase for open source distribution by:
1. Removing cloud-specific files and directories
2. Cleaning sensitive data and credentials
3. Creating a clean OSS-ready package

Usage:
    poetry run python scripts/utils/prepare_open_source.py --output ../memory-opensource

This will create a clean copy of the repository ready for open source release.

The script will:
1. Copy all files except cloud-specific directories and files
2. Exclude cloud_plugins/ and cloud_scripts/ directories
3. Exclude cloud-specific scripts from scripts/ subfolders
4. Remove sensitive files (.env, secrets, etc.)
5. Rename docker-compose-open-source.yaml to docker-compose.yaml
6. Create OSS-specific files (SECURITY.md, TELEMETRY.md)
7. Scan for potential secrets and warn about them

After running, you can:
1. Create a new git repository: cd ../memory-opensource && git init
2. Add remote: git remote add origin <your-repo-url>
3. Commit and push: git add . && git commit -m "Initial OSS release" && git push
"""

import os
import shutil
import sys
import argparse
from pathlib import Path
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class OpenSourcePreparation:
    """Prepare the codebase for open source distribution"""
    
    # Directories to completely exclude from OSS
    EXCLUDE_DIRS = [
        'cloud_plugins',      # Cloud-only features
        'cloud_scripts',      # Cloud maintenance scripts
        '__pycache__',
        '.pytest_cache',
        '.mypy_cache',
        'logs',              # Log files
        'venv',
        'env',
        '.venv',
        'node_modules',
    ]
    
    # Files to exclude from OSS
    EXCLUDE_FILES = [
        '.env',              # Environment variables (has secrets)
        '.env.local',
        '.env.production',
        'azure-pipelines.yml',  # Cloud CI/CD
        'docker-compose.yaml',  # Cloud docker config (keep docker-compose-open-source.yaml)
        '*.log',
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '.DS_Store',
        'Thumbs.db',
    ]
    
    # Cloud-specific scripts to exclude (updated for new scripts folder structure)
    # These scripts are cloud-only and should not be in open source
    EXCLUDE_SCRIPTS = [
        # Cloud maintenance scripts (now in various subfolders)
        'scripts/maintenance/fix_duplicate_api_keys.py',  # Cloud-specific API key management
        'scripts/generators/generate_missing_api_keys.py',  # Cloud-specific key generation
        'scripts/testing/test_production_config.py',  # Production cloud config testing
        
        # Cloud data migration/backfill scripts
        'scripts/migration/sync_neo_to_parse.py',  # Cloud-specific sync
        'scripts/migration/mirror_parse_to_mongo.py',  # Cloud-specific mirroring
        'scripts/migration/copy_lost_data.py',  # Cloud data recovery
        'scripts/migration/backfill_memory_counters.py',  # Cloud counter backfill
        'scripts/migration/add_developer_flags.py',  # Cloud developer features
        'scripts/migration/update_feedback_analytics.py',  # Cloud analytics
        
        # Cloud data files
        'scripts/Cohort_dataloss_users.csv',  # Cloud user data
        
        # Note: cloud_scripts/ directory is excluded via EXCLUDE_DIRS
        # Note: cloud_plugins/ directory is excluded via EXCLUDE_DIRS
    ]
    
    # Config files to exclude
    EXCLUDE_CONFIGS = [
        'config/cloud.yaml',  # Cloud configuration
    ]
    
    # Patterns to search for and warn about (potential secrets)
    SECRET_PATTERNS = [
        r'sk_live_[a-zA-Z0-9]+',  # Stripe live keys
        r'pk_live_[a-zA-Z0-9]+',  # Stripe publishable keys
        r'price_[a-zA-Z0-9]+',     # Stripe price IDs (in config is OK)
        r'prod_[a-zA-Z0-9]+',      # Stripe product IDs
        r'[a-zA-Z0-9]{32,}',       # Long API keys (might be false positives)
    ]
    
    def __init__(self, source_dir: Path, output_dir: Path):
        self.source_dir = source_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.excluded_count = 0
        self.copied_count = 0
        self.warnings = []
    
    def prepare(self):
        """Main preparation process"""
        logger.info(f"Preparing open source distribution...")
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Output: {self.output_dir}")
        
        # Create output directory
        if self.output_dir.exists():
            logger.warning(f"Output directory exists: {self.output_dir}")
            response = input("Delete and recreate? (y/N): ")
            if response.lower() != 'y':
                logger.info("Aborted.")
                return
            shutil.rmtree(self.output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        logger.info("\n📦 Copying files...")
        self._copy_directory(self.source_dir, self.output_dir)
        
        # Clean up cloud-specific code
        logger.info("\n🧹 Cleaning cloud-specific code...")
        self._clean_cloud_code()
        
        # Scan for secrets
        logger.info("\n🔍 Scanning for potential secrets...")
        self._scan_for_secrets()
        
        # Create OSS-specific files
        logger.info("\n📝 Creating OSS-specific files...")
        self._create_oss_files()
        
        # Update documentation
        logger.info("\n📚 Updating documentation...")
        self._update_documentation()
        
        # Final report
        self._print_report()
    
    def _should_exclude(self, path: Path) -> bool:
        """Check if a path should be excluded"""
        relative_path = path.relative_to(self.source_dir)
        path_str = str(relative_path).replace('\\', '/')  # Normalize path separators
        
        # Check excluded directories
        for exclude_dir in self.EXCLUDE_DIRS:
            if exclude_dir in path.parts:
                return True
        
        # Check excluded files (pattern matching)
        for pattern in self.EXCLUDE_FILES:
            if path.match(pattern):
                return True
        
        # Check excluded scripts (exact match or starts with)
        for exclude_script in self.EXCLUDE_SCRIPTS:
            # Handle both exact matches and directory prefixes
            if path_str == exclude_script or path_str.startswith(exclude_script + '/'):
                return True
            # Also check if the script path matches
            if path_str.endswith(exclude_script) or exclude_script in path_str:
                return True
        
        # Check excluded configs
        if path_str in self.EXCLUDE_CONFIGS:
            return True
        
        return False
    
    def _copy_directory(self, src: Path, dst: Path):
        """Recursively copy directory with exclusions"""
        for item in src.iterdir():
            src_path = src / item.name
            dst_path = dst / item.name
            
            if self._should_exclude(src_path):
                self.excluded_count += 1
                logger.debug(f"Excluded: {src_path.relative_to(self.source_dir)}")
                continue
            
            if src_path.is_dir():
                dst_path.mkdir(exist_ok=True)
                self._copy_directory(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
                self.copied_count += 1
    
    def _clean_cloud_code(self):
        """Remove cloud-specific code patterns"""
        # This is a placeholder - you might want to:
        # 1. Remove cloud imports from __init__ files
        # 2. Comment out cloud-specific routes
        # 3. Update default configs
        pass
    
    def _scan_for_secrets(self):
        """Scan for potential hardcoded secrets"""
        python_files = list(self.output_dir.rglob("*.py"))
        
        for file_path in python_files:
            try:
                content = file_path.read_text()
                
                for pattern in self.SECRET_PATTERNS:
                    matches = re.findall(pattern, content)
                    if matches:
                        # Filter out common false positives
                        if 'price_' in pattern and 'config/cloud.yaml' not in str(file_path):
                            # Price IDs in config files are OK
                            continue
                        
                        warning = f"⚠️  Potential secret in {file_path.relative_to(self.output_dir)}"
                        self.warnings.append(warning)
                        logger.warning(warning)
            
            except Exception as e:
                logger.debug(f"Error scanning {file_path}: {e}")
    
    def _create_oss_files(self):
        """Create OSS-specific files"""
        # Rename docker-compose-open-source.yaml to docker-compose.yaml
        oss_compose = self.output_dir / 'docker-compose-open-source.yaml'
        if oss_compose.exists():
            target_compose = self.output_dir / 'docker-compose.yaml'
            shutil.move(oss_compose, target_compose)
            logger.info("✅ Renamed docker-compose-open-source.yaml → docker-compose.yaml")
        
        # Ensure .env.example exists
        env_example = self.output_dir / '.env.example'
        if not env_example.exists():
            logger.warning("⚠️  .env.example not found - should be created!")
        
        # Create SECURITY.md if not exists
        security_md = self.output_dir / 'SECURITY.md'
        if not security_md.exists():
            security_md.write_text(self._get_security_template())
            logger.info("✅ Created SECURITY.md")
        
        # Create docs/TELEMETRY.md if not exists
        telemetry_md = self.output_dir / 'docs' / 'TELEMETRY.md'
        telemetry_md.parent.mkdir(exist_ok=True)
        if not telemetry_md.exists():
            telemetry_md.write_text(self._get_telemetry_template())
            logger.info("✅ Created docs/TELEMETRY.md")
    
    def _update_documentation(self):
        """Update documentation for OSS release"""
        readme = self.output_dir / 'README.md'
        
        if readme.exists():
            content = readme.read_text()
            
            # Update any cloud-specific URLs or references
            content = content.replace(
                'https://github.com/your-org/memory',
                'https://github.com/Papr-ai/memory'
            )
            
            readme.write_text(content)
            logger.info("✅ Updated README.md")
    
    def _print_report(self):
        """Print final report"""
        logger.info("\n" + "="*60)
        logger.info("📊 Open Source Preparation Complete!")
        logger.info("="*60)
        logger.info(f"✅ Copied files: {self.copied_count}")
        logger.info(f"🚫 Excluded files: {self.excluded_count}")
        logger.info(f"⚠️  Warnings: {len(self.warnings)}")
        
        if self.warnings:
            logger.info("\n⚠️  Please review these warnings:")
            for warning in self.warnings:
                logger.info(f"   {warning}")
        
        logger.info(f"\n📁 OSS distribution ready at: {self.output_dir}")
        logger.info("\n📋 Next steps:")
        logger.info("   1. Review the warnings above")
        logger.info(f"   2. Test the OSS build: cd {self.output_dir} && docker-compose -f docker-compose.yaml up")
        logger.info("   3. Review .env.example for completeness")
        logger.info("   4. Review documentation for cloud references")
        logger.info("   5. Initialize git repository:")
        logger.info(f"      cd {self.output_dir}")
        logger.info("      git init")
        logger.info("      git add .")
        logger.info("      git commit -m 'Initial open source release'")
        logger.info("   6. Create private GitHub repo 'memory-opensource' and push:")
        logger.info("      git remote add origin <your-repo-url>")
        logger.info("      git branch -M main")
        logger.info("      git push -u origin main")
        logger.info("   7. When ready to make public, change repo visibility in GitHub settings")
    
    @staticmethod
    def _get_security_template() -> str:
        """Get SECURITY.md template"""
        return """# Security Policy

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**Please DO NOT create a public GitHub issue for security vulnerabilities.**

Instead, please email: **security@papr.ai**

Include the following in your report:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### What to Expect

- We will acknowledge your email within 48 hours
- We will provide an initial assessment within 5 business days
- We will keep you updated on the progress
- We will credit you in the security advisory (unless you prefer to remain anonymous)

### Disclosure Policy

- We aim to disclose vulnerabilities within 90 days of the report
- Critical vulnerabilities will be patched and disclosed as soon as possible
- We will coordinate with you on the disclosure timeline

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x     | :white_check_mark: |
| < 1.0   | :x:                |

## Security Best Practices

When self-hosting Papr Memory:
- Always use HTTPS in production
- Keep all dependencies up to date
- Use strong passwords for databases
- Regularly backup your data
- Implement rate limiting
- Monitor logs for suspicious activity
- Use firewall rules to restrict access

Thank you for helping keep Papr Memory secure!
"""
    
    @staticmethod
    def _get_telemetry_template() -> str:
        """Get docs/TELEMETRY.md template"""
        return """# Telemetry & Privacy Policy

## Overview

Papr Memory includes **optional, privacy-first telemetry** to help us improve the software. This document explains exactly what data is collected, how it's used, and how to opt out.

## Our Commitment

We are committed to privacy and transparency:
- ✅ Telemetry is **opt-out** (can be easily disabled)
- ✅ All data is **anonymous** by default
- ✅ **No personal information** is ever collected
- ✅ Open source users can **self-host their own analytics** with PostHog
- ✅ All telemetry code is **open source** and auditable

## What We Collect

### ✅ Data We DO Collect

1. **Feature Usage**
   - Which API endpoints are called
   - Which features are used
   - Feature adoption rates
   - Example: "search endpoint was called 100 times today"

2. **Performance Metrics**
   - Response times (bucketed, not exact)
   - Query performance
   - Error rates
   - Example: "Average search took 100-500ms"

3. **Error Information**
   - Anonymous error types
   - Error frequency
   - Stack traces (with PII removed)
   - Example: "Database connection error occurred 3 times"

4. **Technical Context**
   - Python version (major.minor only)
   - Edition (opensource or cloud)
   - Version number
   - Example: "Python 3.11, opensource edition, v1.0.0"

### ❌ Data We NEVER Collect

- ❌ **Memory content** - We never see what you store
- ❌ **Search queries** - Your searches are private
- ❌ **Personal information** - No emails, names, or user data
- ❌ **IP addresses** - Your location stays private
- ❌ **File paths or names** - Your file structure is private
- ❌ **Unique device identifiers** - No device tracking
- ❌ **User IDs** - Only hashed anonymous IDs

## How to Opt Out

You can disable telemetry in **multiple ways**:

### Method 1: Environment Variable
```bash
# Add to your .env file
TELEMETRY_ENABLED=false
```

### Method 2: Command Line Flag
```bash
# Start with telemetry disabled
TELEMETRY_ENABLED=false python main.py
```

### Method 3: Config File
```yaml
# Edit config/opensource.yaml
telemetry:
  enabled: false
```

### Verify Telemetry Status
```bash
# Check if telemetry is enabled
curl http://localhost:5001/telemetry/status
```

## Self-Hosted Analytics

Open source users can use **PostHog** for self-hosted analytics:

```bash
# Add to .env
TELEMETRY_PROVIDER=posthog
POSTHOG_HOST=http://your-posthog-instance:8000
POSTHOG_API_KEY=your-key

# Or disable completely
TELEMETRY_ENABLED=false
```

### Deploy Your Own PostHog

```bash
# Self-host PostHog with Docker
git clone https://github.com/PostHog/posthog
cd posthog
docker-compose up -d
```

## Transparency

All telemetry code is open source:
- `/core/services/telemetry.py` - Main telemetry service
- `/config/features.py` - Feature flags
- `/config/opensource.yaml` - Open source config

You can audit exactly what data is sent by reviewing these files.

## Data Retention

- Raw telemetry events: Deleted after **90 days**
- Aggregated statistics: Kept indefinitely (anonymous)
- No individual-level data is retained long-term

## Questions?

If you have questions about telemetry or privacy:
- Open an issue: https://github.com/Papr-ai/memory/issues
- Email: privacy@papr.ai

**Thank you for helping us improve Papr Memory!** 🙏
"""


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Prepare Papr Memory for open source distribution'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for OSS distribution'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='.',
        help='Source directory (default: current directory)'
    )
    
    args = parser.parse_args()
    
    source_dir = Path(args.source).resolve()
    output_dir = Path(args.output).resolve()
    
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        sys.exit(1)
    
    if source_dir == output_dir:
        logger.error("Source and output directories cannot be the same")
        sys.exit(1)
    
    # Run preparation
    prep = OpenSourcePreparation(source_dir, output_dir)
    prep.prepare()


if __name__ == '__main__':
    main()

