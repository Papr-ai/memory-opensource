# Setup Scripts

Scripts for initializing services, creating indexes, and setting up the environment.

## Scripts

- **init_parse_schema_opensource.py** - Initialize Parse Server schemas for open source
- **init_qdrant_collections_opensource.py** - Initialize Qdrant collections
- **setup_api_operation_tracking.py** - Setup API operation tracking
- **create_api_key_index.py** - Create API key indexes in MongoDB
- **add_parse_indexes.py** - Add indexes to Parse Server classes

## Usage

```bash
# Initialize Parse schema
python scripts/setup/init_parse_schema_opensource.py \
  --parse-url http://localhost:1337/parse \
  --app-id papr-oss-app-id \
  --master-key papr-oss-master-key

# Create API key index
python scripts/setup/create_api_key_index.py
```

