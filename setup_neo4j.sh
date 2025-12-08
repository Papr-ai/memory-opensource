#!/bin/bash

# Get current user's UID and GID
CURRENT_UID=$(id -u)
CURRENT_GID=$(id -g)

# Stop and remove existing container if it exists
echo "Stopping and removing existing Neo4j container if it exists..."
docker stop neo4j_secure_sslv4 2>/dev/null
docker rm neo4j_secure_sslv4 2>/dev/null

# Create required directories
echo "Creating required directories..."
sudo mkdir -p /home/devmachine/neo4dbms/local
sudo mkdir -p /home/devmachine/paprmemory/memory/ssl/certs

# Set proper permissions
echo "Setting proper permissions..."
sudo chown -R $CURRENT_UID:$CURRENT_GID /home/devmachine/neo4dbms/local
sudo chown -R $CURRENT_UID:$CURRENT_GID /home/devmachine/paprmemory/memory/ssl/certs

# Run the Neo4j container
echo "Starting Neo4j container..."
docker run --name neo4j_secure_sslv4 \
    --publish=7473:7473 \
    --publish=7687:7687 \
    --env NEO4J_AUTH=neo4j/v1meWWJjhxLQxqEJUCZnHY6dtmTvfAgWlcigYjj7n5k \
    --env NEO4J_dbms_connector_https_enabled=true \
    --env NEO4J_dbms_connector_https_listen__address=0.0.0.0:7473 \
    --env NEO4J_dbms_ssl_policy_https_base__directory=/certs \
    --env NEO4J_dbms_ssl_policy_https_private__key=server.key \
    --env NEO4J_dbms_ssl_policy_https_public__certificate=server.crt \
    --env NEO4J_dbms_ssl_policy_https_client__auth=NONE \
    --env NEO4J_dbms_connector_bolt_enabled=true \
    --env NEO4J_dbms_connector_bolt_tls__level=REQUIRED \
    --env NEO4J_dbms_connector_bolt_listen__address=0.0.0.0:7687 \
    --env NEO4J_dbms_ssl_policy_bolt_enabled=true \
    --env NEO4J_dbms_ssl_policy_bolt_base__directory=/certs \
    --env NEO4J_dbms_ssl_policy_bolt_private__key=server.key \
    --env NEO4J_dbms_ssl_policy_bolt_public__certificate=server.crt \
    --env NEO4J_dbms_ssl_policy_bolt_client__auth=NONE \
    --env NEO4J_dbms_ssl_policy_bolt_revoked__dir=/certs/revoked \
    --env NEO4J_dbms_ssl_policy_bolt_trusted__dir=/certs/trusted \
    --volume=/home/devmachine/neo4dbms/local:/data \
    --volume=/home/devmachine/paprmemory/memory/ssl/certs:/certs \
    --user="$CURRENT_UID:$CURRENT_GID" \
    --memory=2g \
    --cpus=2 \
    --tmpfs /tmp \
    neo4j:latest

# Check if container started successfully
if [ $? -eq 0 ]; then
    echo "Neo4j container started successfully!"
    echo "Container logs will follow. Press Ctrl+C to exit logs (container will continue running)"
    sleep 2
    docker logs -f neo4j_secure_sslv4
else
    echo "Failed to start Neo4j container. Please check the error messages above."
    exit 1
fi 