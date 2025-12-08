#!/bin/bash

# Script to check API key status and provide guidance

echo "========================================="
echo "API Key Configuration Checker"
echo "========================================="
echo ""

# Check OpenAI
echo "🔍 Checking OpenAI API Key..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY is NOT set"
    echo "   → Get your key from: https://platform.openai.com/api-keys"
    echo "   → Then run: export OPENAI_API_KEY='your-key-here'"
else
    KEY_PREFIX=$(echo "$OPENAI_API_KEY" | head -c 10)
    echo "✅ OPENAI_API_KEY is set: ${KEY_PREFIX}..."
    
    # Check if it looks like a valid OpenAI key
    if [[ $OPENAI_API_KEY == sk-* ]]; then
        echo "   ✓ Format looks correct (starts with sk-)"
    else
        echo "   ⚠️  Warning: Key doesn't start with 'sk-' (might be invalid)"
    fi
fi
echo ""

# Check Gemini
echo "🔍 Checking Gemini API Key..."
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY is NOT set"
    echo "   → Get your key from: https://aistudio.google.com/app/apikey"
    echo "   → Then run: export GEMINI_API_KEY='your-key-here'"
else
    KEY_PREFIX=$(echo "$GEMINI_API_KEY" | head -c 10)
    echo "✅ GEMINI_API_KEY is set: ${KEY_PREFIX}..."
fi
echo ""

# Check if keys are in .env
echo "🔍 Checking .env file..."
if [ -f ".env" ]; then
    if grep -q "OPENAI_API_KEY" .env 2>/dev/null; then
        echo "✅ OPENAI_API_KEY found in .env"
    else
        echo "❌ OPENAI_API_KEY not found in .env"
    fi
    
    if grep -q "GEMINI_API_KEY" .env 2>/dev/null; then
        echo "✅ GEMINI_API_KEY found in .env"
    else
        echo "❌ GEMINI_API_KEY not found in .env"
    fi
else
    echo "⚠️  .env file not found"
fi
echo ""

# Check if keys are in ~/.zshrc
echo "🔍 Checking ~/.zshrc..."
if [ -f "$HOME/.zshrc" ]; then
    if grep -q "OPENAI_API_KEY" "$HOME/.zshrc" 2>/dev/null; then
        echo "✅ OPENAI_API_KEY found in ~/.zshrc"
    else
        echo "❌ OPENAI_API_KEY not found in ~/.zshrc"
    fi
    
    if grep -q "GEMINI_API_KEY" "$HOME/.zshrc" 2>/dev/null; then
        echo "✅ GEMINI_API_KEY found in ~/.zshrc"
    else
        echo "❌ GEMINI_API_KEY not found in ~/.zshrc"
    fi
else
    echo "⚠️  ~/.zshrc file not found"
fi
echo ""

echo "========================================="
echo "Summary & Next Steps"
echo "========================================="
echo ""

NEEDS_FIX=0

if [ -z "$OPENAI_API_KEY" ]; then
    NEEDS_FIX=1
    echo "1. Get OpenAI API key: https://platform.openai.com/api-keys"
    echo "   Then add to ~/.zshrc:"
    echo "   echo 'export OPENAI_API_KEY=\"sk-your-key-here\"' >> ~/.zshrc"
    echo ""
fi

if [ -z "$GEMINI_API_KEY" ]; then
    NEEDS_FIX=1
    echo "2. Get Gemini API key: https://aistudio.google.com/app/apikey"
    echo "   Then add to ~/.zshrc:"
    echo "   echo 'export GEMINI_API_KEY=\"your-key-here\"' >> ~/.zshrc"
    echo ""
fi

if [ $NEEDS_FIX -eq 1 ]; then
    echo "3. After updating ~/.zshrc, reload it:"
    echo "   source ~/.zshrc"
    echo ""
    echo "4. Or for quick testing, export in current shell:"
    echo "   export OPENAI_API_KEY='your-key-here'"
    echo "   export GEMINI_API_KEY='your-key-here'"
    echo ""
else
    echo "✅ All API keys appear to be configured!"
    echo ""
    echo "If tests are still failing with API errors, the keys might be:"
    echo "  • Expired or revoked"
    echo "  • Over quota/rate limited"
    echo "  • Invalid format"
    echo ""
    echo "Try regenerating them from the respective platforms."
fi

echo "========================================="

