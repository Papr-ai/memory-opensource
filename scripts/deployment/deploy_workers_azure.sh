#!/bin/bash
# Deploy Temporal workers to Azure App Service

set -e

echo "🚀 Deploying PAPR Memory Workers to Azure"
echo "=========================================="

# Configuration
RESOURCE_GROUP="papr-memory"
WORKER_APP_NAME="papr-memory-workers"
WEB_APP_NAME="papr-memory-api"
LOCATION="eastus"
ACR_NAME="testpaprcontainer"
IMAGE_NAME="memory"
IMAGE_TAG="latest"

# Check if logged in to Azure
if ! az account show > /dev/null 2>&1; then
    echo "❌ Not logged in to Azure. Please run: az login"
    exit 1
fi

echo "✅ Azure CLI authenticated"
echo ""

# Step 1: Create App Service Plans if they don't exist
echo "📦 Creating/Updating App Service Plans..."

# Web Server Plan (P1v3: 8GB RAM, 2 vCPU)
az appservice plan create \
    --name papr-web-plan \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku P1v3 \
    --is-linux \
    --number-of-workers 1 \
    || echo "Web plan already exists"

# Workers Plan (P2v3: 16GB RAM, 4 vCPU)
az appservice plan create \
    --name papr-workers-plan \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku P2v3 \
    --is-linux \
    --number-of-workers 1 \
    || echo "Workers plan already exists"

echo "✅ App Service Plans ready"
echo ""

# Step 2: Create Web App (if doesn't exist)
echo "🌐 Creating/Updating Web Server App..."
az webapp create \
    --resource-group $RESOURCE_GROUP \
    --plan papr-web-plan \
    --name $WEB_APP_NAME \
    --deployment-container-image-name ${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG} \
    || echo "Web app already exists"

# Configure Web App
az webapp config appsettings set \
    --resource-group $RESOURCE_GROUP \
    --name $WEB_APP_NAME \
    --settings \
        WEBSITES_PORT=5001 \
        DOCKER_ENABLE_CI=true

az webapp config set \
    --resource-group $RESOURCE_GROUP \
    --name $WEB_APP_NAME \
    --startup-file "poetry run uvicorn main:app --host 0.0.0.0 --port 5001"

echo "✅ Web Server configured"
echo ""

# Step 3: Create Workers App
echo "⚙️  Creating/Updating Workers App..."
az webapp create \
    --resource-group $RESOURCE_GROUP \
    --plan papr-workers-plan \
    --name $WORKER_APP_NAME \
    --deployment-container-image-name ${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG} \
    || echo "Workers app already exists"

# Configure Workers App
az webapp config set \
    --resource-group $RESOURCE_GROUP \
    --name $WORKER_APP_NAME \
    --startup-file "poetry run python start_all_workers.py"

echo "✅ Workers configured"
echo ""

# Step 4: Configure environment variables from .env
echo "🔧 Configuring environment variables..."

if [ -f ".env" ]; then
    echo "   Loading from .env file..."
    
    # Read .env and set app settings (you'll need to do this manually or parse .env)
    echo "   ⚠️  MANUAL STEP: Copy these environment variables to Azure Portal:"
    echo "   - TEMPORAL_CLOUD_NAMESPACE"
    echo "   - TEMPORAL_CLOUD_ADDRESS"
    echo "   - TEMPORAL_MTLS_CERT"
    echo "   - TEMPORAL_MTLS_KEY"
    echo "   - MONGODB_URI"
    echo "   - NEO4J_URL"
    echo "   - (and all other .env variables)"
    echo ""
    echo "   Or use Azure CLI to set them manually:"
    echo "   az webapp config appsettings set --resource-group $RESOURCE_GROUP --name $WORKER_APP_NAME --settings KEY=VALUE"
else
    echo "   ⚠️  No .env file found. You'll need to set environment variables manually."
fi

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Check deployment status:"
echo "   Web Server:  https://${WEB_APP_NAME}.azurewebsites.net/health"
echo "   Workers:     az webapp log tail --name $WORKER_APP_NAME --resource-group $RESOURCE_GROUP"
echo ""
echo "🔍 View in Azure Portal:"
echo "   https://portal.azure.com/#@/resource/subscriptions/.../resourceGroups/$RESOURCE_GROUP"
echo ""

