# Open Source Scripts

Scripts specific to the open source setup and initialization.

## Scripts

- **bootstrap_opensource_user.py** - Bootstrap default user for open source
- **init_parse_schema_opensource.py** - Initialize Parse Server schemas
- **init_qdrant_collections_opensource.py** - Initialize Qdrant collections
- **docker_entrypoint_opensource.sh** - Docker entrypoint script (auto-runs on container start)

## Usage

These scripts are typically run automatically by the Docker entrypoint script, but can be run manually:

```bash
# Bootstrap user (creates default user, organization, namespace, API key)
python scripts/opensource/bootstrap_opensource_user.py \
  --email opensource@papr.ai \
  --name "Open Source User" \
  --organization "Papr Open Source"

# Initialize Parse schema
python scripts/opensource/init_parse_schema_opensource.py \
  --parse-url http://parse-server:1337/parse \
  --app-id papr-oss-app-id \
  --master-key papr-oss-master-key

# Initialize Qdrant collections
python scripts/opensource/init_qdrant_collections_opensource.py
```

## Docker Integration

The `docker_entrypoint_opensource.sh` script automatically:
1. Waits for services to be ready
2. Initializes Qdrant collections
3. Initializes Parse schemas (if missing)
4. Creates default user (on first run)
5. Generates API key
6. Starts the application

