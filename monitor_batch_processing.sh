#!/bin/bash

# Monitor logs for batch processing activity
LOG_FILE="logs/start_all_workers.log"
BATCH_INDICATORS=("Processing.*memories in batches" "batch_add_memory_quick" "Executing.*batch activities")
INDIVIDUAL_INDICATOR="add_memory_quick.*Activity"

echo "🔍 Monitoring for batch processing activity..."
echo "✅ Looking for:"
echo "   - 'Processing.*memories in batches'"
echo "   - 'batch_add_memory_quick'"
echo "   - 'Executing.*batch activities'"
echo "❌ Will alert if we see: 'add_memory_quick.*Activity' (old code)"
echo ""

LAST_LINE_COUNT=0
while true; do
    CURRENT_LINE_COUNT=$(wc -l < "$LOG_FILE" 2>/dev/null || echo "0")
    
    if [ "$CURRENT_LINE_COUNT" -gt "$LAST_LINE_COUNT" ]; then
        NEW_LINES=$((CURRENT_LINE_COUNT - LAST_LINE_COUNT))
        tail -n "$NEW_LINES" "$LOG_FILE" | while IFS= read -r line; do
            # Check for batch processing indicators
            for pattern in "${BATCH_INDICATORS[@]}"; do
                if echo "$line" | grep -qiE "$pattern"; then
                    echo "✅✅✅ BATCH PROCESSING DETECTED: $line"
                fi
            done
            
            # Check for individual processing (old code)
            if echo "$line" | grep -qiE "$INDIVIDUAL_INDICATOR"; then
                echo "❌❌❌ INDIVIDUAL PROCESSING DETECTED (OLD CODE): $line"
            fi
        done
        
        LAST_LINE_COUNT=$CURRENT_LINE_COUNT
    fi
    
    sleep 2
done
