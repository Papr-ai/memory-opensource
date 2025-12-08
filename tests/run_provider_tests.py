"""
Run all provider simple tests
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def run_all_tests():
    """Run all provider tests"""
    print("\n" + "="*80)
    print("RUNNING ALL PROVIDER TESTS")
    print("="*80)
    
    results = {}
    
    # Test 1: TensorLake
    print("\n" + "="*80)
    print("1/4: Testing TensorLake Provider")
    print("="*80)
    try:
        from test_tensorlake_sdk_simple import main as tensorlake_test
        await tensorlake_test()
        results['TensorLake'] = '✅ PASSED'
    except Exception as e:
        print(f"❌ TensorLake test failed: {e}")
        results['TensorLake'] = f'❌ FAILED: {str(e)[:50]}'
    
    # Test 2: Gemini
    print("\n" + "="*80)
    print("2/4: Testing Gemini Provider")
    print("="*80)
    try:
        from test_gemini_provider_simple import main as gemini_test
        await gemini_test()
        results['Gemini'] = '✅ PASSED'
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        results['Gemini'] = f'❌ FAILED: {str(e)[:50]}'
    
    # Test 3: PaddleOCR
    print("\n" + "="*80)
    print("3/4: Testing PaddleOCR Provider")
    print("="*80)
    try:
        from test_paddleocr_provider_simple import main as paddle_test
        await paddle_test()
        results['PaddleOCR'] = '✅ PASSED'
    except Exception as e:
        print(f"❌ PaddleOCR test failed: {e}")
        results['PaddleOCR'] = f'❌ FAILED: {str(e)[:50]}'
    
    # Test 4: DeepSeek-OCR
    print("\n" + "="*80)
    print("4/4: Testing DeepSeek-OCR Provider")
    print("="*80)
    try:
        from test_deepseek_ocr_provider_simple import main as deepseek_test
        await deepseek_test()
        results['DeepSeek-OCR'] = '✅ PASSED'
    except Exception as e:
        print(f"❌ DeepSeek-OCR test failed: {e}")
        results['DeepSeek-OCR'] = f'❌ FAILED: {str(e)[:50]}'
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    for provider, result in results.items():
        print(f"  {provider:<20} {result}")
    print("="*80 + "\n")
    
    # Return exit code
    failed = [p for p, r in results.items() if 'FAILED' in r]
    if failed:
        print(f"⚠️  {len(failed)} provider(s) failed: {', '.join(failed)}")
        return 1
    else:
        print("✅ All provider tests passed!")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    sys.exit(exit_code)

