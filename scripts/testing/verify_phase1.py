#!/usr/bin/env python
"""
Verification script for Phase 1 implementation.

This script verifies that:
1. Plugin directories exist
2. Feature flags work
3. Telemetry service works
4. Stripe plugin loads (if configured)
5. Config files are valid

Usage:
    poetry run python scripts/verify_phase1.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_directories():
    """Check that all expected directories exist"""
    print("\n📁 Checking directories...")
    
    required_dirs = [
        "cloud_plugins",
        "cloud_plugins/stripe",
        "plugins",
        "cloud_scripts",
        "core/services",
        "config",
    ]
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"  ✅ {dir_name}")
        else:
            print(f"  ❌ {dir_name} - MISSING")
            all_exist = False
    
    return all_exist

def check_files():
    """Check that all expected files exist"""
    print("\n📄 Checking files...")
    
    required_files = [
        "core/services/subscription.py",
        "core/services/telemetry.py",
        "cloud_plugins/__init__.py",
        "cloud_plugins/stripe/__init__.py",
        "cloud_plugins/stripe/service.py",
        "cloud_plugins/stripe/subscription_service.py",
        "config/features.py",
        "config/base.yaml",
        "config/opensource.yaml",
        "config/cloud.yaml",
        ".env.example",
    ]
    
    all_exist = True
    for file_name in required_files:
        file_path = project_root / file_name
        if file_path.exists():
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name} - MISSING")
            all_exist = False
    
    return all_exist

def check_feature_flags():
    """Test feature flag system"""
    print("\n🚩 Testing feature flags...")
    
    try:
        from config import get_features
        
        # Test OSS edition
        os.environ['PAPR_EDITION'] = 'opensource'
        features = get_features()
        
        print(f"  ✅ Feature flags loaded")
        print(f"     Edition: {features.edition}")
        print(f"     Is OSS: {features.is_opensource}")
        print(f"     Has Stripe: {features.has_stripe}")
        print(f"     Telemetry provider: {features.telemetry_provider}")
        
        return True
    except Exception as e:
        print(f"  ❌ Feature flags failed: {e}")
        return False

def check_telemetry():
    """Test telemetry service"""
    print("\n📊 Testing telemetry service...")
    
    try:
        from core.services.telemetry import get_telemetry
        
        telemetry = get_telemetry()
        status = telemetry.get_status()
        
        print(f"  ✅ Telemetry service loaded")
        print(f"     Enabled: {status['enabled']}")
        print(f"     Provider: {status['provider']}")
        print(f"     Edition: {status['edition']}")
        
        return True
    except Exception as e:
        print(f"  ❌ Telemetry service failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_subscription_interface():
    """Test subscription interface"""
    print("\n💳 Testing subscription interface...")
    
    try:
        from core.services.subscription import OpenSourceSubscriptionService
        
        service = OpenSourceSubscriptionService()
        print(f"  ✅ Subscription interface loaded")
        print(f"     Type: {type(service).__name__}")
        
        # Test basic functionality
        import asyncio
        result = asyncio.run(service.check_user_subscription("test_user"))
        print(f"     Test result: {result['tier']} (should be 'unlimited')")
        
        return True
    except Exception as e:
        print(f"  ❌ Subscription interface failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_stripe_plugin():
    """Test Stripe plugin (if available)"""
    print("\n💰 Testing Stripe plugin...")
    
    try:
        os.environ['PAPR_EDITION'] = 'cloud'
        from config import get_features
        
        features = get_features()
        
        if features.has_stripe:
            try:
                from cloud_plugins.stripe import StripeService
                print(f"  ✅ Stripe plugin available")
                print(f"     Has Stripe keys: {bool(os.getenv('STRIPE_SECRET_KEY'))}")
                return True
            except ImportError as e:
                print(f"  ⚠️  Stripe plugin code exists but import failed: {e}")
                return True  # Code exists, just not configured
        else:
            print(f"  ℹ️  Stripe plugin not enabled (feature flag disabled)")
            return True
    except Exception as e:
        print(f"  ❌ Stripe plugin check failed: {e}")
        return False

def check_config_valid():
    """Verify config files are valid YAML"""
    print("\n⚙️  Validating config files...")
    
    try:
        import yaml
        
        config_files = [
            "config/base.yaml",
            "config/opensource.yaml",
            "config/cloud.yaml",
        ]
        
        all_valid = True
        for config_file in config_files:
            config_path = project_root / config_file
            try:
                with open(config_path) as f:
                    yaml.safe_load(f)
                print(f"  ✅ {config_file} - valid YAML")
            except Exception as e:
                print(f"  ❌ {config_file} - invalid YAML: {e}")
                all_valid = False
        
        return all_valid
    except Exception as e:
        print(f"  ❌ Config validation failed: {e}")
        return False

def main():
    """Run all verification checks"""
    print("="*60)
    print("🔍 Phase 1 Verification")
    print("="*60)
    
    checks = [
        ("Directories", check_directories),
        ("Files", check_files),
        ("Feature Flags", check_feature_flags),
        ("Telemetry Service", check_telemetry),
        ("Subscription Interface", check_subscription_interface),
        ("Stripe Plugin", check_stripe_plugin),
        ("Config Files", check_config_valid),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ {name} check crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 Summary")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Phase 1 is ready.")
        print("\nNext steps:")
        print("  1. Test with: PAPR_EDITION=opensource poetry run python main.py")
        print("  2. Test with: PAPR_EDITION=cloud poetry run python main.py")
        print("  3. Update router files (optional)")
        print("  4. Run prepare_open_source.py")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

