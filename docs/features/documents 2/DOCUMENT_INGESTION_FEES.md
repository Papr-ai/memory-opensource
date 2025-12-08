# Document Ingestion Fees

## Overview

Document ingestion fees are **separate** from mini interaction costs and are billed based on document complexity and volume.

## Pricing Structure

Document ingestion fees are usage-based and calculated according to:
- **Volume**: Number of pages or tokens processed
- **Complexity**: Type of reasoning or processing required

### Current Pricing Tiers (per 100 pages)

1. **Basic documents** (text/images): **$1.50**
   - Simple text parsing
   - Basic image extraction
   - Standard PDFs

2. **Advanced documents** (financial/tables): **$2.75**
   - Complex table extraction
   - Financial document parsing
   - Structured data extraction

3. **Complex documents** (scanned/healthcare): **$5.00**
   - OCR processing
   - Scanned document reasoning
   - Healthcare/medical documents
   - Low-quality or handwritten documents

## Mini Interaction Costs

Document operations have **0 mini interactions** as they are billed separately:

```yaml
operation_costs:
  # Document Operations (charged separately via document ingestion fees)
  upload_document: 0  # Billed per page via document ingestion pricing
  get_document_status: 0  # Status check only
  cancel_document_processing: 0
```

## Implementation Strategy

### 1. Stripe Metering Events

For document ingestion fees, you should create **separate Stripe metering events** that track:

- **Document Type**: basic, advanced, or complex
- **Page Count**: Total pages processed
- **Workspace/Customer**: Who to bill

Example Stripe event:
```python
import stripe

# After document processing completes
stripe.billing.MeterEvent.create(
    event_name="document_ingestion",
    payload={
        "value": page_count,  # Number of pages
        "stripe_customer_id": customer_id,
    },
    identifier=f"{workspace_id}_{document_id}",
    timestamp=int(time.time())
)
```

### 2. Document Complexity Detection

You'll need to implement logic to determine document complexity:

```python
def determine_document_complexity(document_metadata: dict) -> str:
    """
    Determine document complexity tier based on characteristics
    
    Returns:
        'basic', 'advanced', or 'complex'
    """
    # Check if document is scanned/OCR required
    if document_metadata.get('requires_ocr'):
        return 'complex'
    
    # Check if document has complex tables/financial data
    if document_metadata.get('has_financial_tables'):
        return 'advanced'
    
    # Check if document is healthcare/medical
    if document_metadata.get('is_medical'):
        return 'complex'
    
    # Default to basic
    return 'basic'
```

### 3. Billing Integration

Create a separate metering system for document ingestion:

```python
async def bill_document_ingestion(
    workspace_id: str,
    document_id: str,
    page_count: int,
    complexity: str,
    stripe_customer_id: str
):
    """
    Bill for document ingestion based on complexity and page count
    
    This is SEPARATE from mini interaction billing
    """
    # Get the price per page based on complexity
    prices = {
        'basic': 0.015,     # $1.50 per 100 pages
        'advanced': 0.0275, # $2.75 per 100 pages
        'complex': 0.05     # $5.00 per 100 pages
    }
    
    price_per_page = prices.get(complexity, 0.015)
    total_cost = page_count * price_per_page
    
    # Send to Stripe metering
    stripe.billing.MeterEvent.create(
        event_name=f"document_ingestion_{complexity}",
        payload={
            "value": page_count,
            "stripe_customer_id": stripe_customer_id,
        },
        identifier=f"{workspace_id}_{document_id}_{int(time.time())}",
        timestamp=int(time.time())
    )
    
    logger.info(
        f"Document ingestion billed: workspace={workspace_id}, "
        f"pages={page_count}, complexity={complexity}, "
        f"cost=${total_cost:.2f}"
    )
```

## Interaction Tracking with operation_type

The `Interaction` class now includes an `operation_type` field to track which specific operation consumed interactions:

### Schema

```json
{
  "workspace": {"__type": "Pointer", "className": "WorkSpace", "objectId": "..."},
  "user": {"__type": "Pointer", "className": "_User", "objectId": "..."},
  "type": "mini",
  "month": 11,
  "year": 2025,
  "count": 4,
  "operation_type": "add_memory_v1",  // NEW FIELD
  "subscription": {"__type": "Pointer", "className": "Subscription", "objectId": "..."},
  "createdAt": "2025-11-10T...",
  "updatedAt": "2025-11-10T..."
}
```

### Operation Types

The `operation_type` field will contain values like:
- `add_memory_v1` (4 mini interactions)
- `search_v1` (1-3 mini interactions based on features)
- `update_memory_v1` (1 mini interaction)
- `get_sync_tiers` (3 mini interactions)
- etc.

This allows developers to see **exactly which operations** consumed their interaction quota.

## Developer Dashboard Display

Show developers their usage breakdown:

```
Mini Interactions This Month: 247 / 1000

Breakdown by Operation:
- add_memory_v1:       120 (30 calls × 4 interactions each)
- search_v1:            75 (50 calls × ~1.5 avg interactions each)
- get_sync_tiers:       36 (12 calls × 3 interactions each)
- update_memory_v1:     10 (10 calls × 1 interaction each)
- get_memory_v1:         6 (6 calls × 1 interaction each)

Document Ingestion This Month: $12.50

Breakdown by Document Type:
- Basic documents:     $4.50 (300 pages)
- Advanced documents:  $5.50 (200 pages)
- Complex documents:   $2.50 (50 pages)
```

## Next Steps

1. **Add Stripe metering events** for document ingestion in the document upload completion handler
2. **Implement document complexity detection** based on document characteristics
3. **Create dashboard UI** to show developers their usage breakdown by operation type
4. **Add tests** to verify operation_type is correctly tracked
5. **Update Parse Server schema** to include `operation_type` field in Interaction class (if not auto-created)

## Notes

- Document ingestion is billed **monthly based on actual usage**
- Mini interactions and document ingestion are **separate line items** on invoices
- The `operation_type` field helps developers optimize their API usage by identifying high-cost operations

