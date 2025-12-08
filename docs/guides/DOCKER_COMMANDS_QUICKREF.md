# Docker Commands Quick Reference

## 🛑 Stop Everything

### Quick Stop (Recommended)
```bash
# Stop and clean up ALL services (both compose files)
docker-compose -f docker-compose.yaml down --remove-orphans
docker-compose -f docker-compose-split.yaml down --remove-orphans
```

### Or Use Script
```bash
chmod +x scripts/stop_all_services.sh
./scripts/stop_all_services.sh
```

### Force Stop (Nuclear Option)
```bash
# Stop ALL memory containers
docker ps -a | grep memory | awk '{print $1}' | xargs docker stop
docker ps -a | grep memory | awk '{print $1}' | xargs docker rm

# Remove networks
docker network rm memory_default memory_network
```

---

## 🚀 Start Services

### Option 1: All-in-One (Development)
```bash
# Simple - everything in one container
docker-compose up -d

# Or with logs (foreground)
docker-compose up

# Or use script
chmod +x scripts/start_services.sh
./scripts/start_services.sh docker-compose.yaml
```

### Option 2: Split Services (Production)
```bash
# Recommended - separate web and workers
docker-compose -f docker-compose-split.yaml up -d

# Or with logs (foreground)
docker-compose -f docker-compose-split.yaml up

# Or use script
chmod +x scripts/start_services.sh
./scripts/start_services.sh docker-compose-split.yaml --logs
```

---

## 📊 Monitor Services

### Check Status
```bash
# All-in-one
docker-compose ps

# Split services
docker-compose -f docker-compose-split.yaml ps

# Or check directly
docker ps | grep memory
```

### View Logs
```bash
# All services (all-in-one)
docker-compose logs -f

# All services (split)
docker-compose -f docker-compose-split.yaml logs -f

# Just web server
docker-compose -f docker-compose-split.yaml logs -f web

# Just workers
docker-compose -f docker-compose-split.yaml logs -f workers

# Last 50 lines only
docker-compose -f docker-compose-split.yaml logs --tail=50 web
```

### Check Resource Usage
```bash
docker stats

# Or formatted
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
```

### Check Health
```bash
# Web server health
curl http://localhost:5001/health

# Or with pretty JSON
curl http://localhost:5001/health | jq .
```

---

## 🔧 Troubleshooting

### Diagnose Issues
```bash
# Use diagnostic script
chmod +x scripts/diagnose_startup.sh
./scripts/diagnose_startup.sh

# Or manually check web container
docker-compose -f docker-compose-split.yaml logs web

# Check last 100 lines
docker-compose -f docker-compose-split.yaml logs --tail=100 web
```

### Rebuild Images
```bash
# All-in-one
docker-compose build --no-cache
docker-compose up -d

# Split services
docker-compose -f docker-compose-split.yaml build --no-cache
docker-compose -f docker-compose-split.yaml up -d
```

### Clean Slate
```bash
# Stop everything
docker-compose down --remove-orphans
docker-compose -f docker-compose-split.yaml down --remove-orphans

# Remove images
docker images | grep memory | awk '{print $3}' | xargs docker rmi -f

# Clean Docker system
docker system prune -a

# Rebuild and start
docker-compose -f docker-compose-split.yaml build --no-cache
docker-compose -f docker-compose-split.yaml up -d
```

---

## 🐛 Debug Mode

### Start in Foreground (See All Output)
```bash
# All-in-one
docker-compose up

# Split services
docker-compose -f docker-compose-split.yaml up

# Press Ctrl+C to stop
```

### Access Container Shell
```bash
# Get a shell in web container
docker-compose -f docker-compose-split.yaml exec web /bin/bash

# Get a shell in workers container
docker-compose -f docker-compose-split.yaml exec workers /bin/bash

# Run a command in container
docker-compose -f docker-compose-split.yaml exec web poetry run python -c "import temporal; print('OK')"
```

### Check Container Exit Codes
```bash
# See why container exited
docker ps -a | grep memory

# Common exit codes:
# 0   = Clean exit
# 1   = General error
# 3   = Command execution error (likely Python error)
# 137 = Out of memory (SIGKILL)
# 143 = Graceful shutdown (SIGTERM)
```

---

## 📋 Common Issues & Solutions

### Issue: "Orphan containers found"
```bash
# Solution: Use --remove-orphans
docker-compose -f docker-compose-split.yaml down --remove-orphans
docker-compose -f docker-compose-split.yaml up -d
```

### Issue: "Port 5001 already in use"
```bash
# Find what's using port 5001
lsof -i :5001

# Kill the process (if safe)
kill <PID>

# Or use different port in docker-compose
# Change: "5001:5001" to "5002:5001"
```

### Issue: "Container exits immediately (exit code 3)"
```bash
# Check logs
docker-compose -f docker-compose-split.yaml logs web

# Common causes:
# 1. Missing .env file
# 2. Invalid environment variables
# 3. Python import error
# 4. Database connection failure

# Start in foreground to see error
docker-compose -f docker-compose-split.yaml up web
```

### Issue: "Out of memory (exit code 137)"
```bash
# Increase Docker Desktop memory:
# Docker Desktop → Settings → Resources → Memory → 8GB

# Or reduce container memory limits in docker-compose:
# memory: 2G → memory: 1G
```

### Issue: "Workers not connecting to Temporal"
```bash
# Check Temporal credentials in .env
grep TEMPORAL .env

# Check worker logs
docker-compose -f docker-compose-split.yaml logs workers | grep -i temporal

# Test Temporal connection
docker-compose -f docker-compose-split.yaml exec workers \
  poetry run python -c "
from cloud_plugins.temporal.client import get_temporal_client
import asyncio
asyncio.run(get_temporal_client())
print('✅ Connected!')
"
```

---

## 🎯 Recommended Workflow

### Daily Development
```bash
# Start services
./scripts/start_services.sh docker-compose-split.yaml

# Check logs as needed
docker-compose -f docker-compose-split.yaml logs -f

# Stop when done
docker-compose -f docker-compose-split.yaml down
```

### Debugging Session
```bash
# Clean start
docker-compose -f docker-compose-split.yaml down --remove-orphans
docker-compose -f docker-compose-split.yaml up

# Watch logs, press Ctrl+C when done
```

### After Code Changes
```bash
# Rebuild and restart
docker-compose -f docker-compose-split.yaml down
docker-compose -f docker-compose-split.yaml build
docker-compose -f docker-compose-split.yaml up -d
```

---

## 📚 Additional Resources

- **Docker Compose Docs**: https://docs.docker.com/compose/
- **Temporal Cloud Docs**: https://docs.temporal.io/cloud
- **Project Guides**:
  - `docs/TEMPORAL_CLOUD_WORKERS_GUIDE.md` - Temporal setup
  - `docs/AZURE_DEPLOYMENT_ARCHITECTURE.md` - Production deployment

---

## 🆘 Quick Help

```bash
# I just want to start everything
./scripts/start_services.sh docker-compose-split.yaml

# I just want to stop everything
./scripts/stop_all_services.sh

# I want to see what's wrong
./scripts/diagnose_startup.sh

# I want to see logs
docker-compose -f docker-compose-split.yaml logs -f
```

