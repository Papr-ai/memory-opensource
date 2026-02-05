# Session Compress Endpoint Guide

## Overview

The Papr Memory Service provides a conversation compression endpoint that reduces full chat histories into hierarchical summaries, perfect for LLM context windows.

**Developer-Friendly Naming**: We use `/compress` because it clearly communicates what developers want - compressed context for their AI agents.

---

## Endpoints

### 🎯 Recommended: GET /v1/messages/sessions/{sessionId}/compress

```bash
GET https://api.papr.ai/v1/messages/sessions/{sessionId}/compress
```

**Headers:**
```
X-API-Key: your_api_key_here
```

**Response:**
```json
{
  "session_id": "chat_20260201_120000",
  "summaries": {
    "short_term": "User is building JWT authentication for a React task app...",
    "medium_term": "Over the past 50 messages, the user has...",
    "long_term": "Full session: User started by asking about React best practices...",
    "topics": ["React", "JWT", "Authentication", "TypeScript"],
    "last_updated": "2026-02-01T12:30:00Z"
  },
  "enhanced_fields": {
    "session_intent": "Build secure JWT authentication for task management app",
    "key_decisions": [
      "Use httpOnly cookies instead of localStorage for security",
      "Implement refresh token rotation"
    ],
    "current_state": "Auth flow working, testing refresh token logic",
    "next_steps": [
      "Add role-based access control",
      "Write integration tests",
      "Deploy to staging"
    ],
    "technical_details": [
      "Token expiry: 3600 seconds",
      "Refresh endpoint: POST /api/auth/refresh",
      "Cookie settings: httpOnly, secure, sameSite=strict"
    ],
    "files_accessed": {
      "read": ["src/api/auth.ts", "src/components/Login.tsx"],
      "modified": [
        {"path": "src/api/auth.ts", "description": "Added JWT generation"}
      ],
      "created": ["src/hooks/useAuth.ts", "src/utils/tokenStorage.ts"],
      "deleted": []
    },
    "project_context": {
      "project_name": "Task Management App",
      "project_id": "proj_task_management_app",
      "project_path": "/Users/user/projects/task-app",
      "tech_stack": ["React", "TypeScript", "Node.js", "Express", "JWT"],
      "current_task": "Implementing JWT authentication with httpOnly cookies",
      "git_repo": null
    }
  },
  "ai_agent_note": "To find more details about this conversation, search memories with metadata filter: sessionId='chat_20260201_120000'",
  "from_cache": true,
  "message_count": 47
}
```

---

## Why Three Levels of Summaries?

### 1. **Short-Term** (Last 15 messages)
- Most recent context
- Current task/problem
- Immediate conversation state
- **Use for**: Quick context, current task focus

### 2. **Medium-Term** (Last ~100 messages)
- Recent session history
- Multiple related tasks
- Pattern across recent work
- **Use for**: Session context, understanding flow

### 3. **Long-Term** (Full session)
- Complete conversation arc
- All major decisions
- Full project context
- **Use for**: High-level overview, project documentation

---

## Usage in Paprwork

### Automatically Called:

```javascript
// paprManager.js
async getSessionSummary(sessionId) {
  // Calls GET /v1/messages/sessions/{sessionId}/compress
  const response = await this._apiRequest(
    'GET', 
    `/v1/messages/sessions/${sessionId}/compress`
  );
  
  return {
    sessionId,
    short_term: response.summaries.short_term,
    medium_term: response.summaries.medium_term,
    long_term: response.summaries.long_term,
    topics: response.summaries.topics,
    session_intent: response.enhanced_fields.session_intent,
    key_decisions: response.enhanced_fields.key_decisions,
    current_state: response.enhanced_fields.current_state,
    next_steps: response.enhanced_fields.next_steps,
    technical_details: response.enhanced_fields.technical_details,
    files_accessed: response.enhanced_fields.files_accessed,
    project_context: response.enhanced_fields.project_context
  };
}
```

### When It's Called:

```javascript
// After local summarization (as a fallback check)
const paprSummary = await fetchSummaryFromPaprMemory(chatId);

if (paprSummary) {
  // Use rich Papr Memory summary
  console.log('✅ Using summary from Papr Memory');
  webContents.send('conversation-summary', {
    chatId,
    summary: paprSummary.summary,
    type: 'papr_memory'
  });
} else {
  // Fallback to local summary
  await extractAndSaveFromSummary(summaryResult.summary, chatId);
}
```

---

## Input Requirements

### **Just the session ID!**

That's it. No complex parameters needed:

```bash
# Paprwork sends session ID
GET /v1/messages/sessions/chat_20260201_120000/compress

# Papr Memory:
# 1. Looks up session in Parse Server
# 2. Checks if summaries exist
# 3. Returns existing summaries (fast!)
# 4. Or triggers summarization if needed
```

**Behind the scenes:**
- Authentication via `X-API-Key` header
- User/workspace extracted from API key
- Summaries are user-scoped (only your data)
- Organization-level isolation

---

## The Endpoint

```bash
GET /v1/messages/sessions/{sessionId}/compress
```

**Why GET?**
- ✅ Idempotent (safe to call multiple times)
- ✅ RESTful (reading data, not modifying)
- ✅ Fast (cached responses)
- ✅ Cacheable by browsers/CDNs

**Input:** Just the session ID
**Output:** Compressed context with rich metadata

---

## When Summaries Are Generated

### Automatic (Every 15 Messages):

```
Message 1  → Stored
Message 2  → Stored
...
Message 15 → Stored + Batch Analysis Triggered
  ↓
Papr Memory:
  - Analyzes all 15 messages with Groq LLM
  - Generates short/medium/long summaries
  - Extracts project context, tech stack, decisions
  - Saves to Parse ChatSession
  - Creates Neo4j nodes (MessageSession, Project, etc.)
  ↓
Future calls to /compress return instantly!
```

### On-Demand (Before 15 Messages):

```bash
# First call to /compress will trigger generation automatically
GET /v1/messages/sessions/{sessionId}/compress

# Papr Memory:
# - Returns 404 if no summary exists
# - Triggers generation in background
# - Next call returns the summary
```

---

## Response Fields Explained

### Basic Summaries (Always Present):

| Field | Description | Example |
|-------|-------------|---------|
| `short_term` | Last 15 messages | "User is implementing JWT auth..." |
| `medium_term` | Last ~100 messages | "User has been building authentication system..." |
| `long_term` | Full session | "Complete project: Task management app with React..." |
| `topics` | Key topics | `["React", "JWT", "TypeScript"]` |
| `last_updated` | When generated | `"2026-02-01T12:30:00Z"` |

### Enhanced Fields (From Batch Analysis):

| Field | Description | Example |
|-------|-------------|---------|
| `session_intent` | What user is trying to accomplish | "Build secure JWT authentication" |
| `key_decisions` | Important choices made | "Use httpOnly cookies for security" |
| `current_state` | Where we are now | "Auth flow working, testing refresh logic" |
| `next_steps` | What to do next | "Add RBAC, write tests, deploy" |
| `technical_details` | Config, endpoints, errors | "Token expiry: 3600s" |
| `files_accessed` | Files read/modified/created | Read: `auth.ts`, Created: `useAuth.ts` |
| `project_context` | Project info | Name, tech stack, current task |

---

## Error Handling

### No Summaries Exist:

```bash
GET /v1/messages/sessions/{sessionId}/compress

Response: 404
{
  "detail": "Session not found or no summary exists"
}
```

**What to do:**
1. Call /compress again in a few seconds (background processing)
2. Or wait for 15 messages to trigger auto-summarization

### Session Doesn't Exist:

```bash
GET /v1/messages/sessions/invalid_session/compress

Response: 404
{
  "detail": "Session not found"
}
```

### Not Authenticated:

```bash
GET /v1/messages/sessions/{sessionId}/compress
# (No X-API-Key header)

Response: 401
{
  "detail": "Authentication required"
}
```

---

## Use Cases

### 1. **Reduce Context Window Size**

```javascript
// Full history = 50,000 tokens
const fullHistory = await chatManager.loadChat(chatId);

// Compressed summary = 2,000 tokens
const summary = await paprManager.getSessionSummary(chatId);

// Use summary in system prompt instead of full history
const systemPrompt = `
Previous conversation context:
${summary.long_term}

Current task: ${summary.session_intent}
Current state: ${summary.current_state}
Next steps: ${summary.next_steps.join(', ')}
`;
```

**Savings**: 96% token reduction!

---

### 2. **Quick Conversation Overview**

```javascript
// Show user what this chat is about
const summary = await paprManager.getSessionSummary(chatId);

console.log(`Chat: ${summary.project_context.project_name}`);
console.log(`Goal: ${summary.session_intent}`);
console.log(`Status: ${summary.current_state}`);
console.log(`Next: ${summary.next_steps[0]}`);
```

---

### 3. **AI Agent Memory**

```javascript
// Give AI agent compressed context
const summary = await paprManager.getSessionSummary(chatId);

const agentPrompt = `
You are helping with: ${summary.project_context.project_name}
Tech stack: ${summary.project_context.tech_stack.join(', ')}

Recent work:
${summary.short_term}

Key decisions made:
${summary.key_decisions.map(d => `- ${d}`).join('\n')}

Current blocker: ${summary.current_state}

Please help with: ${summary.next_steps[0]}
`;
```

---

### 4. **Project Documentation**

```javascript
// Generate project snapshot
const summary = await paprManager.getSessionSummary(chatId);

const doc = `
# ${summary.project_context.project_name}

## Tech Stack
${summary.project_context.tech_stack.join(', ')}

## Current Status
${summary.current_state}

## Key Decisions
${summary.key_decisions.map((d, i) => `${i + 1}. ${d}`).join('\n')}

## Next Steps
${summary.next_steps.map((s, i) => `${i + 1}. ${s}`).join('\n')}

## Files Modified
${summary.files_accessed.modified.map(f => `- ${f.path}: ${f.description}`).join('\n')}
`;

fs.writeFileSync('project-snapshot.md', doc);
```

---

## Performance

### First Call (No Summary Exists):

```
GET /compress
  ↓
Check Parse Server (~50ms)
  ↓
No summary found → Return 404
  ↓
Trigger batch analysis (background - 3s)
  ↓
Summary saved to Parse
  ↓
Second call: Returns instantly!
```

**First call**: ~50ms (returns 404, triggers background processing)  
**Second call**: ~50ms (returns full summary)

### Subsequent Calls (Summary Exists):

```
GET /compress
  ↓
Check Parse Server (~50ms)
  ↓
Summary found! → Return immediately
  ↓
Total: ~50ms
```

**Every call after**: ~50ms (cached in Parse Server)

**No repeated LLM calls!** Summaries are generated once per 15 messages, then cached.

---

## Developer Tips

### ✅ DO:

```javascript
// Use GET /compress (idempotent, fast, cacheable)
const summary = await fetch(
  'https://api.papr.ai/v1/messages/sessions/chat_123/compress',
  { headers: { 'X-API-Key': apiKey } }
);

// Check from_cache to know if this was instant
if (summary.from_cache) {
  console.log('✅ Instant response from cache');
} else {
  console.log('🆕 Just generated');
}
```

### ❌ DON'T:

```javascript
// Don't fetch full history when you just need context
const fullHistory = await getFullHistory(); // 50k tokens
// Use compress instead:
const summary = await getCompress(); // 2k tokens

// Don't call compress on every message - it's expensive
// Wait until you need the context (e.g., new feature, different task)
```

---

## Summary

### What to Send:
✅ **Just the session ID!**
```javascript
GET /v1/messages/sessions/{sessionId}/compress
```

### What You Get:
✅ **Everything you need:**
- 3 levels of summaries (short/medium/long)
- Topics discussed
- Session intent & current state
- Key decisions & next steps
- Technical details
- Files accessed
- Project context (name, tech stack, current task)

### When to Use:
✅ **Every time you need conversation context**
- Feeding AI agents
- Reducing token usage
- Quick conversation overview
- Project documentation
- Context for new features

**Result**: 96% token savings, instant retrieval, rich context! 🚀
