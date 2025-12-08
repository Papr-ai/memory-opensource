# paprDB Cross-Platform Architecture

## 🎯 Overview

paprDB uses SQLite as the storage engine, which provides **excellent cross-platform support**. This document outlines how to implement paprDB SDKs for:
- ✅ Python (Desktop, Server)
- ✅ TypeScript/JavaScript (Web, macOS, Node.js)
- ✅ iOS (Native Swift)
- ✅ Android (Native Kotlin/Java)

---

## 📱 Platform Support Matrix

| Platform | SQLite Support | Recommended Library | Status |
|----------|---------------|---------------------|--------|
| **Python** | ✅ Native | `sqlite3` (built-in) | ✅ Perfect |
| **TypeScript/Node.js** | ✅ Native | `better-sqlite3` or `sql.js` | ✅ Perfect |
| **TypeScript/Web** | ⚠️ Limited | `sql.js` (WASM) or IndexedDB | ⚠️ Needs workaround |
| **macOS (Swift)** | ✅ Native | SQLite.swift or FMDB | ✅ Perfect |
| **iOS (Swift)** | ✅ Native | SQLite.swift or Core Data | ✅ Perfect |
| **Android (Kotlin)** | ✅ Native | SQLite (built-in) | ✅ Perfect |

---

## 🐍 Python SDK

### Implementation

```python
# Python SDK (papr-python)
import sqlite3
import json
from typing import Dict, List, Optional

class PaprDB:
    def __init__(self, db_path: str = "papr.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()
    
    def create_node(self, node_type: str, properties: Dict):
        # SQLite is built-in, works perfectly
        self.conn.execute(
            "INSERT INTO nodes (id, type, properties) VALUES (?, ?, ?)",
            (node_id, node_type, json.dumps(properties))
        )
```

**Status**: ✅ **Perfect** - SQLite is built into Python standard library

---

## 📘 TypeScript/Node.js SDK

### Option 1: better-sqlite3 (Recommended)

```typescript
// TypeScript SDK (papr-ts) - Node.js
import Database from 'better-sqlite3';

class PaprDB {
    private db: Database.Database;
    
    constructor(dbPath: string = 'papr.db') {
        this.db = new Database(dbPath);
        this.initSchema();
    }
    
    createNode(nodeType: string, properties: Record<string, any>): string {
        const nodeId = `${nodeType}_${Date.now()}`;
        this.db.prepare(
            'INSERT INTO nodes (id, type, properties) VALUES (?, ?, ?)'
        ).run(nodeId, nodeType, JSON.stringify(properties));
        return nodeId;
    }
    
    graphqlQuery(query: string): any {
        // GraphQL → SQL translation
        const sql = this.graphqlToSQL(query);
        return this.db.prepare(sql).all();
    }
}
```

**Pros**:
- ✅ Synchronous API (simpler)
- ✅ Fast (native bindings)
- ✅ Full SQLite features

**Cons**:
- ❌ Node.js only (not web browsers)

**Status**: ✅ **Perfect for Node.js/macOS**

---

### Option 2: sql.js (Web + Node.js)

```typescript
// TypeScript SDK (papr-ts) - Web Browser
import initSqlJs, { Database } from 'sql.js';

class PaprDB {
    private db: Database | null = null;
    
    async init(dbPath?: string): Promise<void> {
        const SQL = await initSqlJs();
        this.db = new SQL.Database();
        this.initSchema();
    }
    
    createNode(nodeType: string, properties: Record<string, any>): string {
        if (!this.db) throw new Error('Database not initialized');
        const nodeId = `${nodeType}_${Date.now()}`;
        this.db.run(
            'INSERT INTO nodes (id, type, properties) VALUES (?, ?, ?)',
            [nodeId, nodeType, JSON.stringify(properties)]
        );
        return nodeId;
    }
}
```

**Pros**:
- ✅ Works in web browsers (WASM)
- ✅ Same API as native SQLite
- ✅ Cross-platform (web + Node.js)

**Cons**:
- ⚠️ Larger bundle size (~1MB WASM)
- ⚠️ Slightly slower than native

**Status**: ✅ **Good for Web + Node.js**

---

## 🌐 Web Browser Considerations

### Challenge: Browser Storage Limits

**Problem**: Web browsers can't directly access SQLite files. Options:

1. **sql.js (WASM)** - In-memory SQLite
2. **IndexedDB** - Browser-native storage
3. **Hybrid** - sql.js + IndexedDB persistence

### Solution: Hybrid Approach (Recommended)

```typescript
// TypeScript SDK (papr-ts) - Web Browser with IndexedDB persistence
import initSqlJs, { Database } from 'sql.js';

class PaprDBWeb {
    private db: Database | null = null;
    private indexedDB: IDBDatabase | null = null;
    
    async init(): Promise<void> {
        // 1. Load SQLite WASM
        const SQL = await initSqlJs();
        this.db = new SQL.Database();
        
        // 2. Open IndexedDB for persistence
        this.indexedDB = await this.openIndexedDB();
        
        // 3. Load existing data from IndexedDB
        await this.loadFromIndexedDB();
        
        // 4. Initialize schema
        this.initSchema();
    }
    
    async createNode(nodeType: string, properties: Record<string, any>): Promise<string> {
        if (!this.db) throw new Error('Database not initialized');
        
        const nodeId = `${nodeType}_${Date.now()}`;
        
        // 1. Insert into SQLite (in-memory)
        this.db.run(
            'INSERT INTO nodes (id, type, properties) VALUES (?, ?, ?)',
            [nodeId, nodeType, JSON.stringify(properties)]
        );
        
        // 2. Persist to IndexedDB (background)
        await this.persistToIndexedDB();
        
        return nodeId;
    }
    
    private async persistToIndexedDB(): Promise<void> {
        if (!this.db || !this.indexedDB) return;
        
        // Export SQLite database to binary
        const data = this.db.export();
        
        // Store in IndexedDB
        const transaction = this.indexedDB.transaction(['papr'], 'readwrite');
        const store = transaction.objectStore('papr');
        await store.put(data, 'database');
    }
    
    private async loadFromIndexedDB(): Promise<void> {
        if (!this.indexedDB) return;
        
        const transaction = this.indexedDB.transaction(['papr'], 'readonly');
        const store = transaction.objectStore('papr');
        const data = await store.get('database');
        
        if (data) {
            // Load SQLite database from binary
            this.db!.run(data);
        }
    }
}
```

**Architecture**:
```
┌─────────────────────────────────────┐
│      Web Browser (papr-ts)         │
├─────────────────────────────────────┤
│                                     │
│  ┌──────────────────────────────┐ │
│  │   sql.js (WASM)               │ │
│  │   - In-memory SQLite          │ │
│  │   - GraphQL queries           │ │
│  │   - Fast operations           │ │
│  └───────────┬──────────────────┘ │
│              │                     │
│              │ Export/Import       │
│              │                     │
│  ┌───────────▼──────────────────┐ │
│  │   IndexedDB                   │ │
│  │   - Persistent storage        │ │
│  │   - Browser-native            │ │
│  │   - ~50MB limit (per domain)  │ │
│  └───────────────────────────────┘ │
└─────────────────────────────────────┘
```

**Status**: ✅ **Works well** - sql.js + IndexedDB provides SQLite-like experience

---

## 🍎 iOS SDK (Swift)

### Option 1: SQLite.swift (Recommended)

```swift
// iOS SDK (papr-ios) - Swift
import SQLite

class PaprDB {
    private var db: Connection
    
    init(dbPath: String = "papr.db") throws {
        self.db = try Connection(dbPath)
        try initSchema()
    }
    
    func createNode(nodeType: String, properties: [String: Any]) -> String {
        let nodeId = "\(nodeType)_\(Int(Date().timeIntervalSince1970 * 1000))"
        let propertiesJSON = try! JSONSerialization.data(withJSONObject: properties)
        let propertiesString = String(data: propertiesJSON, encoding: .utf8)!
        
        try! db.run(
            "INSERT INTO nodes (id, type, properties) VALUES (?, ?, ?)",
            nodeId, nodeType, propertiesString
        )
        
        return nodeId
    }
    
    func graphqlQuery(query: String) -> [[String: Any]] {
        // GraphQL → SQL translation
        let sql = graphqlToSQL(query)
        var results: [[String: Any]] = []
        
        for row in try! db.prepare(sql) {
            var dict: [String: Any] = [:]
            for (index, name) in row.columnNames.enumerated() {
                dict[name] = row[index]
            }
            results.append(dict)
        }
        
        return results
    }
}
```

**Pros**:
- ✅ Type-safe Swift API
- ✅ Native performance
- ✅ Full SQLite features

**Status**: ✅ **Perfect for iOS**

---

### Option 2: Core Data (Alternative)

```swift
// iOS SDK (papr-ios) - Using Core Data
import CoreData

class PaprDB {
    private var context: NSManagedObjectContext
    
    init() {
        let container = NSPersistentContainer(name: "PaprDB")
        container.loadPersistentStores { _, error in
            if let error = error {
                fatalError("Failed to load store: \(error)")
            }
        }
        self.context = container.viewContext
    }
    
    func createNode(nodeType: String, properties: [String: Any]) -> String {
        let node = Node(context: context)
        node.id = UUID().uuidString
        node.type = nodeType
        node.properties = try! JSONSerialization.data(withJSONObject: properties)
        
        try! context.save()
        return node.id!
    }
}
```

**Pros**:
- ✅ Apple's recommended approach
- ✅ Built-in sync (CloudKit)
- ✅ Type-safe models

**Cons**:
- ⚠️ More complex setup
- ⚠️ Not SQLite directly (uses SQLite under the hood)

**Status**: ✅ **Good alternative** - More iOS-native, but less portable

---

## 🤖 Android SDK (Kotlin)

```kotlin
// Android SDK (papr-android) - Kotlin
import android.database.sqlite.SQLiteDatabase
import org.json.JSONObject

class PaprDB(context: Context, dbPath: String = "papr.db") {
    private val db: SQLiteDatabase
    
    init {
        db = context.openOrCreateDatabase(dbPath, Context.MODE_PRIVATE, null)
        initSchema()
    }
    
    fun createNode(nodeType: String, properties: Map<String, Any>): String {
        val nodeId = "${nodeType}_${System.currentTimeMillis()}"
        val propertiesJSON = JSONObject(properties).toString()
        
        db.execSQL(
            "INSERT INTO nodes (id, type, properties) VALUES (?, ?, ?)",
            arrayOf(nodeId, nodeType, propertiesJSON)
        )
        
        return nodeId
    }
    
    fun graphqlQuery(query: String): List<Map<String, Any>> {
        val sql = graphqlToSQL(query)
        val cursor = db.rawQuery(sql, null)
        val results = mutableListOf<Map<String, Any>>()
        
        cursor.use {
            while (it.moveToNext()) {
                val row = mutableMapOf<String, Any>()
                for (i in 0 until it.columnCount) {
                    row[it.getColumnName(i)] = it.getString(i)
                }
                results.add(row)
            }
        }
        
        return results
    }
}
```

**Status**: ✅ **Perfect** - SQLite is built into Android

---

## 🏗️ Unified API Design

### Cross-Platform API Contract

All SDKs should implement the same interface:

```typescript
// TypeScript interface (reference)
interface PaprDB {
    // Initialization
    init(dbPath?: string): Promise<void>;
    
    // Node operations
    createNode(nodeType: string, properties: Record<string, any>): Promise<string>;
    updateNode(nodeId: string, properties: Record<string, any>): Promise<void>;
    getNode(nodeId: string): Promise<Node | null>;
    
    // Constraint operations
    applyConstraints(nodeType: string, properties: Record<string, any>): Promise<Record<string, any>>;
    syncConstraints(): Promise<void>;
    
    // Graph operations
    createEdge(sourceId: string, targetId: string, edgeType: string): Promise<string>;
    getNeighbors(nodeId: string, edgeType?: string): Promise<Node[]>;
    
    // GraphQL
    graphqlQuery(query: string, variables?: Record<string, any>): Promise<any>;
    
    // Sync
    syncToCloud(): Promise<void>;
    syncFromCloud(cursor?: string): Promise<string>;
}
```

### Platform-Specific Implementations

```python
# Python
class PaprDB:
    def create_node(self, node_type: str, properties: Dict) -> str:
        # Implementation
```

```typescript
// TypeScript
class PaprDB {
    async createNode(nodeType: string, properties: Record<string, any>): Promise<string> {
        // Implementation
    }
}
```

```swift
// Swift
class PaprDB {
    func createNode(nodeType: String, properties: [String: Any]) -> String {
        // Implementation
    }
}
```

```kotlin
// Kotlin
class PaprDB {
    fun createNode(nodeType: String, properties: Map<String, Any>): String {
        // Implementation
    }
}
```

---

## 📦 Package Structure

```
papr-sdk/
├── papr-python/          # Python SDK
│   ├── paprdb/
│   │   ├── __init__.py
│   │   ├── database.py   # SQLite implementation
│   │   ├── constraints.py
│   │   └── graphql.py
│   └── setup.py
│
├── papr-ts/              # TypeScript SDK
│   ├── src/
│   │   ├── database.ts  # better-sqlite3 or sql.js
│   │   ├── constraints.ts
│   │   └── graphql.ts
│   ├── web/             # Web-specific (sql.js + IndexedDB)
│   └── package.json
│
├── papr-ios/            # iOS SDK
│   ├── PaprDB/
│   │   ├── PaprDB.swift
│   │   ├── Constraints.swift
│   │   └── GraphQL.swift
│   └── Package.swift
│
└── papr-android/        # Android SDK
    ├── src/main/kotlin/
    │   └── paprdb/
    │       ├── PaprDB.kt
    │       ├── Constraints.kt
    │       └── GraphQL.kt
    └── build.gradle
```

---

## 🔄 Sync Compatibility

### Cross-Platform Sync Format

All SDKs use the same sync format:

```json
{
  "nodes": [
    {
      "id": "node_123",
      "type": "Project",
      "properties": {"name": "Alpha", "status": "active"},
      "constraints_applied": true,
      "constraint_version": 1
    }
  ],
  "edges": [
    {
      "id": "edge_456",
      "source_id": "node_123",
      "target_id": "node_789",
      "type": "HAS_TASK"
    }
  ],
  "constraints": [
    {
      "id": "constraint_1",
      "node_type": "Project",
      "force": {"workspace_id": "ws_123"},
      "version": 1
    }
  ]
}
```

**Result**: Python, TypeScript, iOS, and Android can all sync with the same cloud API.

---

## ⚡ Performance Comparison

| Platform | Library | Performance | Bundle Size |
|----------|---------|-------------|-------------|
| **Python** | `sqlite3` | ⚡⚡⚡ Excellent | 0KB (built-in) |
| **TypeScript/Node** | `better-sqlite3` | ⚡⚡⚡ Excellent | ~500KB |
| **TypeScript/Web** | `sql.js` | ⚡⚡ Good | ~1MB (WASM) |
| **iOS** | `SQLite.swift` | ⚡⚡⚡ Excellent | ~200KB |
| **Android** | `SQLite` (built-in) | ⚡⚡⚡ Excellent | 0KB (built-in) |

---

## ✅ Recommendations

### For Each Platform

1. **Python**: Use `sqlite3` (built-in) ✅
2. **TypeScript/Node.js**: Use `better-sqlite3` ✅
3. **TypeScript/Web**: Use `sql.js` + IndexedDB ✅
4. **iOS**: Use `SQLite.swift` ✅
5. **Android**: Use built-in `SQLiteDatabase` ✅

### Unified Design

- ✅ Same API across all platforms
- ✅ Same sync format
- ✅ Same GraphQL support
- ✅ Same constraint system

---

## 🎯 Conclusion

**SQLite works excellently across all platforms:**

- ✅ **Python**: Built-in, perfect
- ✅ **TypeScript/Node.js**: `better-sqlite3`, perfect
- ✅ **TypeScript/Web**: `sql.js` + IndexedDB, works well
- ✅ **iOS**: `SQLite.swift`, perfect
- ✅ **Android**: Built-in, perfect

**Result**: You can build paprDB SDKs for all platforms with the same architecture! 🚀

