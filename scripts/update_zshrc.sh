#!/bin/bash
# Script to update ~/.zshrc with required environment variables
# Run this script to automatically update your shell environment

echo "🔧 Updating ~/.zshrc with environment variables..."
echo ""

# Add USE_DOTENV=false to disable .env loading
echo "✅ Adding USE_DOTENV=false..."
echo 'export USE_DOTENV=false' >> ~/.zshrc

# Add missing environment variables (you need to edit these with your actual values)
echo "✅ Adding missing environment variables..."
cat >> ~/.zshrc << 'EOF'

# Parse Server Configuration
export PARSE_APPLICATION_ID='your-parse-app-id'        # Replace with your actual Parse App ID
export PARSE_REST_API_KEY='your-parse-rest-key'       # Replace with your actual Parse REST API Key
export PARSE_MASTER_KEY='your-parse-master-key'       # Replace with your actual Parse Master Key

# Web Application URLs
export WEB_APP_URL='https://your-web-app-url'         # Replace with your web app URL

# Database and Vector Services
export NEO4J_URL='neo4j://your-neo4j-url'             # Replace with your Neo4j connection URL
export PINECONE_KEY='your-pinecone-key'              # Replace with your Pinecone API key
export PINECONE_ENV='your-pinecone-env'              # Replace with your Pinecone environment
export QDRANT_URL='http://localhost:6333'            # Local Qdrant URL (usually doesn't change)

EOF

# Remove old MONGO_URI if it exists (we derive it from DATABASE_URI now)
echo "✅ Removing old MONGO_URI export (if exists)..."
sed -i '' '/export MONGO_URI=/d' ~/.zshrc

echo ""
echo "🎉 ~/.zshrc updated successfully!"
echo ""
echo "📝 NEXT STEPS:"
echo "1. Edit ~/.zshrc and replace all 'your-*' placeholders with actual values"
echo "2. Run: source ~/.zshrc"
echo "3. Test: python3 -c \"import os; print('PARSE_APPLICATION_ID:', os.getenv('PARSE_APPLICATION_ID'))\""
echo ""
echo "🔍 REQUIRED VALUES TO FILL IN:"
echo "- PARSE_APPLICATION_ID: Your Parse Server application ID"
echo "- PARSE_REST_API_KEY: Your Parse Server REST API key"
echo "- PARSE_MASTER_KEY: Your Parse Server master key"
echo "- WEB_APP_URL: Your web application URL (e.g., https://app.papr.ai)"
echo "- NEO4J_URL: Your Neo4j database URL (e.g., neo4j://localhost:7687)"
echo "- PINECONE_KEY: Your Pinecone API key"
echo "- PINECONE_ENV: Your Pinecone environment (e.g., 'us-west1-gcp')"
echo ""
echo "💡 TIP: You can find these values in your Parse Server dashboard,"
echo "   Pinecone console, and other service configurations."
