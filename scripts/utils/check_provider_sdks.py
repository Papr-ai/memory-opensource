"""
Check and display current versions of provider SDKs
"""

import subprocess
import sys

def check_package_version(package_name):
    """Check if package is installed and its version"""
    try:
        result = subprocess.run(
            ["poetry", "show", package_name],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            # Parse version from output
            for line in result.stdout.split('\n'):
                if line.startswith('version'):
                    version = line.split(':')[1].strip()
                    return version
            return "installed (version unknown)"
        else:
            return "not installed"
    except Exception as e:
        return f"error: {e}"


def check_pypi_latest(package_name):
    """Check latest version on PyPI"""
    try:
        result = subprocess.run(
            ["poetry", "search", package_name],
            capture_output=True,
            text=True,
            check=False,
            timeout=10
        )
        # Note: Poetry search might not work, this is just a helper
        return "use: poetry add package@latest"
    except Exception:
        return "N/A"


def main():
    print("\n" + "="*80)
    print("PROVIDER SDK VERSION CHECK")
    print("="*80)
    
    packages = {
        "tensorlake": "TensorLake Document AI SDK",
        "google-generativeai": "Google Gemini SDK",
        "paddleocr": "PaddleOCR (local OCR)",
        "httpx": "HTTP Client (for DeepSeek-OCR)",
        "certifi": "SSL Certificates"
    }
    
    print(f"\n{'Package':<25} {'Description':<35} {'Current Version':<20}")
    print("-"*80)
    
    for package, description in packages.items():
        version = check_package_version(package)
        print(f"{package:<25} {description:<35} {version:<20}")
    
    print("\n" + "="*80)
    print("UPDATE COMMANDS")
    print("="*80)
    print("\n# Update TensorLake SDK:")
    print("poetry update tensorlake")
    print("\n# Update Gemini SDK:")
    print("poetry update google-generativeai")
    print("\n# Update PaddleOCR:")
    print("poetry update paddleocr")
    print("\n# Update all:")
    print("poetry update")
    
    print("\n" + "="*80)
    print("LATEST VERSIONS (check manually)")
    print("="*80)
    print("\nTensorLake: https://pypi.org/project/tensorlake/")
    print("Google GenAI: https://pypi.org/project/google-generativeai/")
    print("PaddleOCR: https://pypi.org/project/paddleocr/")
    print("\n")


if __name__ == "__main__":
    main()

