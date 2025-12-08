# Docker Quick Start Guide - PAPR Memory Server

## 🚀 TL;DR - Start Everything

```bash
# Stop any existing services
docker-compose -f docker-compose-split.yaml down --remove-orphans

# Start web + workers
docker-compose -f docker-compose-split.yaml up -d

# View logs
docker-compose -f docker-compose-split.yaml logs -f
```

---

## 📋 Architecture

### Split Configuration (Recommended)
```
docker-compose-split.yaml
├── web (memory-web-1)       → FastAPI server on port 5001
└── workers (memory-workers-1) → Temporal workers
    ├── Memory Worker          → memory-processing queue
    └── Document Worker        → document-processing queue
```

**Why split?**
- ✅ Matches production architecture
- ✅ Web API stays responsive during heavy processing
- ✅ Independent scaling
- ✅ Better debugging (separate logs)
- ✅ Worker crashes don't affect API

### All-in-One Configuration (Development)
```
docker-compose.yaml
└── web (memory-web-1)       → Everything in one container
    ├── FastAPI server
    ├── Memory Worker
    └── Document Worker
```

**Use when:** Quick local testing, simpler setup

---

## 🎯 Common Commands

### Start Services

```bash
# Production-like (split services)
docker-compose -f docker-compose-split.yaml up -d

# Development (all-in-one)
docker-compose up -d

# Foreground with logs (press Ctrl+C to stop)
docker-compose -f docker-compose-split.yaml up
```

### Stop Services

```bash
# Clean stop (removes containers and networks)
docker-compose -f docker-compose-split.yaml down

# Stop and remove orphans (recommended)
docker-compose -f docker-compose-split.yaml down --remove-orphans

# Force stop all memory containers
docker ps -a | grep memory | awk '{print $1}' | xargs docker stop
docker ps -a | grep memory | awk '{print $1}' | xargs docker rm
```

### View Logs

```bash
# ALL logs (web + workers mixed)
docker-compose -f docker-compose-split.yaml logs -f

# ONLY web server logs
docker-compose -f docker-compose-split.yaml logs -f web

# ONLY worker logs
docker-compose -f docker-compose-split.yaml logs -f workers

# Last 50 lines (no follow)
docker-compose -f docker-compose-split.yaml logs --tail=50 workers

# Grep for specific text
docker-compose -f docker-compose-split.yaml logs workers | grep "Temporal"

# Follow logs with grep (real-time)
docker-compose -f docker-compose-split.yaml logs -f workers | grep "Successfully connected"
```

### Check Status

```bash
# Container status
docker-compose -f docker-compose-split.yaml ps

# Expected output:
# NAME                  STATUS
# memory-web-1          Up (healthy)
# memory-workers-1      Up (healthy)

# Resource usage
docker stats

# Formatted
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
```

### Health Checks

```bash
# Web server health
curl http://localhost:5001/health

# Should return:
# {"status":"healthy","message":"Service is running"}

# API docs
open http://localhost:5001/docs
```

---

## ✅ Verify Everything Is Working

### 1. Check Container Status
```bash
docker-compose -f docker-compose-split.yaml ps

# Both should show "Up (healthy)" after 60 seconds
```

### 2. Check Web Server
```bash
curl http://localhost:5001/health

# Should return 200 OK
```

### 3. Check Worker Logs
```bash
docker-compose -f docker-compose-split.yaml logs workers | tail -20

# Look for:
# ✅ Successfully connected to Temporal
# 🔧 Starting Memory Worker on task queue: memory-processing
# 🔧 Starting Document Worker on task queue: document-processing
# ✅ Both workers configured successfully
```

### 4. Check Temporal Cloud
Go to: https://cloud.temporal.io → Your namespace → Task Queues

Should see:
```
Task Queue              | Pollers | Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
memory-processing       | 1       | 🟢 Active
document-processing     | 1       | 🟢 Active
```

---

## 🔧 Rebuild After Code Changes

```bash
# Stop services
docker-compose -f docker-compose-split.yaml down

# Rebuild images (no cache)
docker-compose -f docker-compose-split.yaml build --no-cache

# Start fresh
docker-compose -f docker-compose-split.yaml up -d

# Watch logs
docker-compose -f docker-compose-split.yaml logs -f
```

---

## 🐛 Troubleshooting

### Issue: Container exits immediately
```bash
# Check exit code
docker ps -a | grep memory

# View logs
docker-compose -f docker-compose-split.yaml logs web

# Common causes:
# - Missing .env file
# - Invalid environment variables
# - Database connection failure
```

### Issue: Healthcheck failing (unhealthy)
```bash
# Check logs for 405 errors
docker-compose -f docker-compose-split.yaml logs web | grep "405"

# Fix: Use GET request in healthcheck (already fixed in docker-compose-split.yaml)
# test: ["CMD-SHELL", "wget -O /dev/null http://localhost:5001/health"]
```

### Issue: Workers not connecting to Temporal
```bash
# Check Temporal credentials
grep TEMPORAL .env

# View worker connection logs
docker-compose -f docker-compose-split.yaml logs workers | grep -i temporal

# Should see:
# Successfully connected to Temporal at us-west-2.aws.api.temporal.io
```

### Issue: Out of memory (exit code 137)
```bash
# Increase Docker Desktop memory:
# Docker Desktop → Settings → Resources → Memory → 8GB

# Or reduce container memory limits in docker-compose-split.yaml
```

### Issue: Port 5001 already in use
```bash
# Find what's using the port
lsof -i :5001

# Kill the process
kill <PID>

# Or change port in docker-compose-split.yaml:
# ports: "5002:5001"  # External:Internal
```

---

## 📊 What Each Log Means

### Web Server Startup
```
✅ Stripe service initialized
✅ Parse Server connection
✅ MongoDB connection successful
✅ Qdrant client initialized
✅ Neo4j connection test successful
✅ Application startup complete
INFO: Uvicorn running on http://0.0.0.0:5001
```

### Workers Startup
```
✅ Successfully connected to Temporal
🔧 Starting Memory Worker on task queue: memory-processing
🔧 Starting Document Worker on task queue: document-processing
✅ Both workers configured successfully
📊 Memory Worker: 3 Workflows, 14 Activities
📄 Document Worker: 1 Workflow, 16 Activities
🚀 Starting both workers...
```

### Healthy Healthcheck
```
INFO: 127.0.0.1:42096 - "GET /health HTTP/1.1" 200 OK
```

### Unhealthy Healthcheck (BEFORE FIX)
```
INFO: 127.0.0.1:42096 - "HEAD /health HTTP/1.1" 405 Method Not Allowed
```

---

## 🎓 Best Practices

1. **Always use split configuration** for development (matches production)
2. **Use `--remove-orphans`** when switching between configs
3. **Check logs** before assuming something is broken
4. **Verify Temporal connection** in Cloud UI (Task Queues tab)
5. **Rebuild after dependency changes** in pyproject.toml
6. **Watch resource usage** with `docker stats`
7. **Keep Docker Desktop memory at 8GB+** for smooth operation

---

## 📚 Related Documentation

- `DOCKER_COMMANDS_QUICKREF.md` - Comprehensive command reference
- `docs/TEMPORAL_CLOUD_WORKERS_GUIDE.md` - Temporal setup and architecture
- `docs/AZURE_DEPLOYMENT_ARCHITECTURE.md` - Production deployment
- `agent.md` - Engineering learnings (Docker Deployment section)

---

## 🆘 Still Having Issues?

```bash
# Nuclear option: Clean slate
docker-compose -f docker-compose-split.yaml down --remove-orphans
docker system prune -a
docker-compose -f docker-compose-split.yaml build --no-cache
docker-compose -f docker-compose-split.yaml up

# Watch startup logs carefully for errors
```

---

## 💡 Pro Tips

### Multiple Terminal Setup
```bash
# Terminal 1: Web logs
docker-compose -f docker-compose-split.yaml logs -f web

# Terminal 2: Worker logs
docker-compose -f docker-compose-split.yaml logs -f workers

# Terminal 3: Commands
docker-compose -f docker-compose-split.yaml ps
curl http://localhost:5001/health
```

### Useful Aliases
```bash
# Add to ~/.zshrc or ~/.bashrc
alias dc='docker-compose -f docker-compose-split.yaml'
alias dcl='docker-compose -f docker-compose-split.yaml logs -f'
alias dcp='docker-compose -f docker-compose-split.yaml ps'
alias dcr='docker-compose -f docker-compose-split.yaml down && docker-compose -f docker-compose-split.yaml up -d'

# Usage:
dc ps                    # Status
dcl workers             # Worker logs
dcr                     # Restart all
```

### Grep Patterns
```bash
# Worker connections
docker-compose -f docker-compose-split.yaml logs workers | grep -E "✅|🔧"

# Errors only
docker-compose -f docker-compose-split.yaml logs | grep -i error

# Specific provider
docker-compose -f docker-compose-split.yaml logs workers | grep -i "tensorlake"

# Temporal activities
docker-compose -f docker-compose-split.yaml logs workers | grep "Activity"
```

---

**Last Updated**: October 2025  
**Tested With**: Docker Desktop 4.25+, Python 3.11, Temporal Cloud

