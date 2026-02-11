# Enterprise Document Processing UX Design

## Overview
Design for optimal developer experience leveraging Temporal's durable execution UI for enterprise document ingestion at scale.

## Current State
- Document v1 routes handle single document uploads
- No batch processing for documents
- No progress tracking for large operations
- Limited visibility into processing pipeline

## Proposed Enterprise UX

### 1. Document Batch API Integration

**Endpoint**: `POST /v1/documents/batch`

```json
{
  "documents": [
    {
      "filename": "contract_2024.pdf",
      "content": "base64_encoded_content",
      "metadata": {
        "document_type": "contract",
        "client": "acme_corp",
        "priority": "high"
      }
    }
  ],
  "webhook_url": "https://your-app.com/webhooks/documents",
  "webhook_secret": "your_secret_key",
  "processing_options": {
    "provider": "tensorlake",
    "store_in_memory": true,
    "generate_summaries": true
  }
}
```

**Response**:
```json
{
  "status": "processing",
  "batch_id": "batch_doc_f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "workflow_id": "document-batch-f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "temporal_ui_url": "https://temporal.papr.ai/namespaces/papr-memory.pq3ak/workflows/document-batch-f47ac10b",
  "estimated_completion_minutes": 15,
  "total_documents": 25
}
```

### 2. Temporal UI Integration

**Key Benefits**:
- **Real-time Progress**: See exactly which documents are being processed
- **Error Visibility**: Pinpoint which documents failed and why
- **Retry Management**: Restart failed documents without reprocessing successful ones
- **Performance Metrics**: Processing time per document, bottleneck identification
- **Audit Trail**: Complete history of all processing attempts

**Developer Access**:
1. **Direct Link**: API response includes Temporal UI URL for immediate access
2. **Embedded Widget**: Iframe integration in developer dashboard
3. **Status API**: RESTful endpoints for programmatic access

### 3. Document Ingestion Timeline UI

#### 3.1 Durable Execution Timeline (Temporal-Style)
Inspired by Temporal's timeline UI, show document → memory transformation:

```
┌─ Document Ingestion Timeline: contract_2024.pdf ──────────────────┐
│                                                                   │
│  📄 Document Upload         ✅ 00:00:12  [✓ Completed]           │
│  ├─ File validation         ✅ 00:00:02                          │
│  ├─ Security scan          ✅ 00:00:08                          │
│  └─ Metadata extraction    ✅ 00:00:02                          │
│                                                                   │
│  🔄 Document Processing     ✅ 00:02:34  [✓ Completed]           │
│  ├─ OCR extraction         ✅ 00:01:45                          │
│  ├─ Content structuring    ✅ 00:00:32                          │
│  ├─ Entity detection       ✅ 00:00:12                          │
│  └─ Summary generation     ✅ 00:00:05                          │
│                                                                   │
│  🧠 Memory Creation         ✅ 00:01:23  [✓ Completed]           │
│  ├─ Content chunking       ✅ 00:00:15                          │
│  ├─ Embedding generation   ✅ 00:00:45                          │
│  ├─ Vector indexing        ✅ 00:00:18                          │
│  └─ Graph relationship     ✅ 00:00:05                          │
│                                                                   │
│  🎯 Intelligence Layer     🔄 00:00:34  [⏳ In Progress]        │
│  ├─ Category prediction    ✅ 00:00:12  "Legal Contract"       │
│  ├─ Priority scoring       ✅ 00:00:08  "High Priority"        │
│  ├─ Auto-tagging          🔄 00:00:14  [⏳ Processing]         │
│  └─ Related mem. linking   ⏳ Pending                           │
│                                                                   │
│  📊 Final Status: 3/4 stages complete, 12 memories created      │
│                                                                   │
│  💡 Predicted Insights:                                          │
│  • Document Type: Contract (98% confidence)                      │
│  • Urgency: High (Due date: 2024-12-31)                        │
│  • Key Entities: [ACME Corp, $50,000, John Doe]                │
│  • Similar Docs: 3 related contracts found                      │
└───────────────────────────────────────────────────────────────────┘
```

#### 3.2 Memory-Centric Progress View
Show the end result: memories created and ready for search/retrieval:

```
┌─ Memory Creation Progress ─────────────────────────────────────────┐
│                                                                    │
│  📊 Batch: Legal Contracts Q4 2024                                │
│                                                                    │
│  📄 Documents → 🧠 Memories → 🔍 Ready for Search                 │
│                                                                    │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░ 15/20 docs processed                       │
│                                                                    │
│  💾 Memory Statistics:                                             │
│  ├─ Total memories created: 347                                    │
│  ├─ Vector embeddings: 347/347 ✅                                  │
│  ├─ Graph relationships: 89 connections                            │
│  ├─ Search index status: Ready ✅                                  │
│  └─ Average confidence: 94.3%                                      │
│                                                                    │
│  🎯 Intelligence Insights:                                         │
│  ├─ Auto-categorized: 15 contracts, 0 amendments                  │
│  ├─ Priority classified: 8 high, 7 medium                         │
│  ├─ Key entities extracted: 45 companies, 23 people               │
│  └─ Deadline predictions: 12 critical dates identified            │
│                                                                    │
│  🔍 Test Your Memories:                                           │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Query: "Show me high-priority contracts expiring in Q1"     │ │
│  │ [Search] → Expected Results: 3 memories                     │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

#### 3.3 Individual Document Drill-Down
Click any document to see its complete journey:

```
┌─ Document Journey: annual_report_2024.pdf ────────────────────────┐
│                                                                   │
│  🎯 Outcome: 23 memories indexed and searchable                   │
│                                                                   │
│  📊 Memory Breakdown:                                             │
│  ├─ 🏢 Company Info (5 memories)                                  │
│  │  └─ "ACME Corp revenue increased 23% in Q4..."               │
│  ├─ 📈 Financial Data (8 memories)                                │
│  │  └─ "Q4 2024 earnings exceeded expectations by..."           │
│  ├─ 👥 Leadership (4 memories)                                    │
│  │  └─ "CEO John Smith announced strategic initiatives..."       │
│  ├─ 📅 Key Dates (3 memories)                                     │
│  │  └─ "Annual shareholder meeting scheduled March 15..."        │
│  └─ 📋 Strategic Plans (3 memories)                               │
│     └─ "Five-year growth plan targets $500M revenue..."          │
│                                                                   │
│  🔗 Auto-Generated Relationships:                                 │
│  ├─ Links to: quarterly_report_q3.pdf (8 connections)            │
│  ├─ Related to: "financial_projections" tag (12 memories)        │
│  └─ Connected to: "John Smith" entity (6 memories)               │
│                                                                   │
│  🧠 Intelligence Applied:                                         │
│  ├─ Document Type: Annual Report (99.7% confidence)              │
│  ├─ Sentiment Analysis: Positive outlook (87% confidence)        │
│  ├─ Key Topics: [Revenue Growth, Leadership Changes, M&A]        │
│  └─ Action Items: 4 followup tasks identified                    │
│                                                                   │
│  ✅ Status: All memories ready for search and retrieval          │
│  🕒 Total Processing Time: 00:03:47                              │
└───────────────────────────────────────────────────────────────────┘
```

#### 3.4 Real-Time Memory Search Validation
Test memories as they're created:

```
┌─ Live Memory Testing ──────────────────────────────────────────────┐
│                                                                    │
│  🔍 Query: "What did the CEO say about Q4 performance?"           │
│                                                                    │
│  📊 Search Results: 3 memories found                              │
│  ├─ Memory #1: "CEO John Smith reported Q4 revenue..."           │
│  │  └─ Confidence: 94.2% | Source: annual_report_2024.pdf:p12   │
│  ├─ Memory #2: "Q4 performance exceeded all expectations..."      │
│  │  └─ Confidence: 91.8% | Source: annual_report_2024.pdf:p3    │
│  └─ Memory #3: "Leadership team highlighted Q4 achievements..."   │
│     └─ Confidence: 87.3% | Source: annual_report_2024.pdf:p45   │
│                                                                    │
│  ⚡ Response Time: 127ms | Vector Search: ✅ Graph Traverse: ✅   │
└────────────────────────────────────────────────────────────────────┘
```

#### 3.5 React Component Architecture

```typescript
// DocumentIngestionTimeline.tsx
import React, { useState, useEffect } from 'react';
import { useTemporalWorkflow } from '@/hooks/useTemporalWorkflow';

interface IngestionStage {
  id: string;
  name: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  duration: number;
  substages: SubStage[];
  insights?: PredictionInsight[];
}

interface PredictionInsight {
  type: 'category' | 'priority' | 'entity' | 'relationship';
  value: string;
  confidence: number;
  predictions?: string[];
}

const DocumentIngestionTimeline: React.FC<{
  documentId: string;
  workflowId: string;
}> = ({ documentId, workflowId }) => {
  const { workflow, realTimeStatus } = useTemporalWorkflow(workflowId);
  const [expandedStages, setExpandedStages] = useState<Set<string>>(new Set());

  const stages: IngestionStage[] = [
    {
      id: 'upload',
      name: '📄 Document Upload',
      status: realTimeStatus.uploadStage,
      duration: realTimeStatus.uploadDuration,
      substages: [
        { name: 'File validation', status: 'completed', duration: 2000 },
        { name: 'Security scan', status: 'completed', duration: 8000 },
        { name: 'Metadata extraction', status: 'completed', duration: 2000 }
      ]
    },
    {
      id: 'processing',
      name: '🔄 Document Processing',
      status: realTimeStatus.processingStage,
      duration: realTimeStatus.processingDuration,
      substages: [
        { name: 'OCR extraction', status: 'completed', duration: 105000 },
        { name: 'Content structuring', status: 'completed', duration: 32000 },
        { name: 'Entity detection', status: 'completed', duration: 12000 },
        { name: 'Summary generation', status: 'completed', duration: 5000 }
      ]
    },
    {
      id: 'memory',
      name: '🧠 Memory Creation',
      status: realTimeStatus.memoryStage,
      duration: realTimeStatus.memoryDuration,
      substages: [
        { name: 'Content chunking', status: 'completed', duration: 15000 },
        { name: 'Embedding generation', status: 'completed', duration: 45000 },
        { name: 'Vector indexing', status: 'completed', duration: 18000 },
        { name: 'Graph relationship', status: 'completed', duration: 5000 }
      ],
      insights: realTimeStatus.memoryInsights
    },
    {
      id: 'intelligence',
      name: '🎯 Intelligence Layer',
      status: realTimeStatus.intelligenceStage,
      duration: realTimeStatus.intelligenceDuration,
      substages: [
        { name: 'Category prediction', status: 'completed', duration: 12000 },
        { name: 'Priority scoring', status: 'completed', duration: 8000 },
        { name: 'Auto-tagging', status: 'in_progress', duration: 14000 },
        { name: 'Related mem. linking', status: 'pending', duration: 0 }
      ],
      insights: [
        {
          type: 'category',
          value: 'Legal Contract',
          confidence: 98,
          predictions: ['Contract', 'Legal Document', 'Agreement']
        },
        {
          type: 'priority',
          value: 'High Priority',
          confidence: 92,
          predictions: ['High', 'Medium', 'Low']
        }
      ]
    }
  ];

  return (
    <div className="document-timeline">
      <div className="timeline-header">
        <h2>Document Ingestion: {workflow.documentName}</h2>
        <div className="overall-progress">
          {workflow.completedStages}/4 stages complete, {workflow.memoriesCreated} memories created
        </div>
      </div>

      {stages.map((stage) => (
        <StageTimeline
          key={stage.id}
          stage={stage}
          expanded={expandedStages.has(stage.id)}
          onToggle={() => toggleStage(stage.id)}
          realTime={true}
        />
      ))}

      <MemorySearchTest
        documentId={documentId}
        memoriesReady={workflow.memoriesCreated}
      />

      <PredictionInsights insights={workflow.allInsights} />
    </div>
  );
};

const MemorySearchTest: React.FC<{
  documentId: string;
  memoriesReady: number;
}> = ({ documentId, memoriesReady }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [searchTime, setSearchTime] = useState(0);

  const testSearch = async () => {
    const startTime = Date.now();
    const response = await fetch(`/api/v1/memories/search`, {
      method: 'POST',
      body: JSON.stringify({
        query,
        filters: { document_id: documentId }
      })
    });
    const data = await response.json();
    setResults(data.memories);
    setSearchTime(Date.now() - startTime);
  };

  return (
    <div className="memory-test-panel">
      <h3>🔍 Test Your Memories ({memoriesReady} ready)</h3>
      <div className="search-input">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Query your newly created memories..."
        />
        <button onClick={testSearch} disabled={memoriesReady === 0}>
          Search
        </button>
      </div>

      {results.length > 0 && (
        <div className="search-results">
          <div className="results-meta">
            Found {results.length} memories in {searchTime}ms
          </div>
          {results.map((memory, i) => (
            <div key={i} className="memory-result">
              <div className="memory-content">{memory.content}</div>
              <div className="memory-meta">
                Confidence: {memory.confidence}% | Source: {memory.source}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default DocumentIngestionTimeline;
```

#### 3.6 Temporal UI Integration
  namespace="papr-memory.pq3ak"
  height="500px"
  features={['timeline', 'activities', 'errors']}
/>
```

#### 3.3 Enhanced Error Handling
```
┌─ Failed Document Details ──────────────────────────────────┐
│                                                            │
│  📄 contract_malformed.pdf                                 │
│  ❌ Error: Invalid PDF structure at byte offset 1247      │
│  🔧 Suggested Fix: Re-export PDF from original source     │
│  ⚡ Actions: [Retry] [Skip] [Download for Manual Review]  │
│                                                            │
│  📊 Processing Timeline:                                   │
│  ├─ 10:30 AM: Validation started                          │
│  ├─ 10:31 AM: PDF parsing failed                          │
│  └─ 10:31 AM: Marked as failed with retry recommendation │
└────────────────────────────────────────────────────────────┘
```

### 4. API Integration Points

#### 4.1 Enhanced Document v1 Routes

**Batch Upload**:
- `POST /v1/documents/batch` → Start Temporal workflow
- Returns workflow tracking information

**Status Checking**:
- `GET /v1/documents/batch/{batch_id}/status`
- `GET /v1/documents/batch/{batch_id}/results`

**Individual Document Status**:
- `GET /v1/documents/{document_id}/processing_status`

#### 4.2 Webhook Integration
```json
{
  "event": "batch.document.completed",
  "batch_id": "batch_doc_f47ac10b",
  "total_documents": 25,
  "successful": 23,
  "failed": 2,
  "processing_time_seconds": 847,
  "results": {
    "successful_documents": [
      {
        "filename": "contract_2024.pdf",
        "document_id": "doc_abc123",
        "memory_ids": ["mem_xyz789", "mem_abc456"],
        "processing_time_seconds": 34
      }
    ],
    "failed_documents": [
      {
        "filename": "corrupted.pdf",
        "error": "PDF parsing failed",
        "retry_recommended": true
      }
    ]
  }
}
```

### 5. Implementation Strategy

#### Phase 1: Core Temporal Integration
- [ ] Extend existing `start_temporal_worker.py` ✅
- [ ] Implement `POST /v1/documents/batch` endpoint
- [ ] Add workflow status endpoints

#### Phase 2: Developer Dashboard
- [ ] Create Temporal UI embedding components
- [ ] Build document processing dashboard
- [ ] Implement real-time status updates

#### Phase 3: Enterprise Features
- [ ] Advanced error handling and suggestions
- [ ] Bulk retry mechanisms
- [ ] Custom processing pipelines
- [ ] Analytics and reporting

### 6. Technical Architecture

#### 6.1 Workflow Integration
```python
# Enhanced document route
@router.post("/v1/documents/batch")
async def process_document_batch(
    request: DocumentBatchRequest,
    auth: dict = Depends(authenticate)
):
    # Start Temporal workflow
    workflow_id = await start_document_workflow(
        client=temporal_client,
        documents=request.documents,
        auth_response=auth,
        webhook_url=request.webhook_url
    )

    return {
        "status": "processing",
        "workflow_id": workflow_id,
        "temporal_ui_url": f"{TEMPORAL_UI_BASE_URL}/workflows/{workflow_id}",
        "estimated_completion_minutes": len(request.documents) * 0.6
    }
```

#### 6.2 Dashboard API
```python
@router.get("/v1/documents/batch/{batch_id}/temporal-embed")
async def get_temporal_embed_url(batch_id: str):
    return {
        "embed_url": f"{TEMPORAL_UI_BASE_URL}/embed/workflows/{batch_id}",
        "features": ["timeline", "activities", "input_output", "stack_trace"]
    }
```

### 7. Benefits for Enterprise Developers

1. **Transparency**: Full visibility into document processing pipeline
2. **Reliability**: Guaranteed processing with automatic retries
3. **Scalability**: Handle thousands of documents efficiently
4. **Debugging**: Pinpoint exact failure points with stack traces
5. **Integration**: Webhook notifications for downstream systems
6. **Monitoring**: Built-in metrics and performance tracking

### 8. Success Metrics

- **Processing Success Rate**: >99% for valid documents
- **Error Resolution Time**: <5 minutes average
- **Developer Satisfaction**: Measured via dashboard usage and feedback
- **Scale**: Support 10,000+ document batches per organization
- **Performance**: <30 seconds average processing per document

This design leverages Temporal's strengths while providing enterprise developers with the visibility and control they need for large-scale document processing operations.