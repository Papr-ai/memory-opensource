#!/usr/bin/env python3
"""
Add a Workflow with Steps connected to SecurityBehaviors
Demonstrates workflow monitoring: detect skipped steps and their security impacts
"""

import requests
import json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = "f80c5a2940f21882420b41690522cb2c"
SESSION_TOKEN = "r:578db0db09b3159b7ec98e0043b2af9a"

HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
    "Authorization": f"Bearer {SESSION_TOKEN}"
}

content = """Customer Support Call Workflow - Security Best Practices

This workflow ensures customer support representatives follow security protocols when handling customer calls to prevent social engineering attacks and unauthorized access.

Workflow: Customer Support Call Handling
Steps:
1. Answer call professionally and introduce yourself
2. Verify caller identity using multi-factor authentication
3. Document call purpose and customer request
4. Provide requested information or assistance
5. Confirm customer satisfaction and close call

Security Behaviors Required:
- Step 2: SB080 - Verify caller legitimacy before sharing information
- Step 4: SB091 - Limit information disclosure to verified parties

Critical: Skipping Step 2 (caller verification) can lead to system compromise with medium-high probability."""

# Graph override with Workflow, WorkflowSteps, and SecurityBehaviors
graph_override = {
    "nodes": [
        # Workflow
        {
            "id": "workflow_customer_support",
            "label": "Workflow",
            "properties": {
                "id": "WF001",
                "name": "Customer Support Call Handling",
                "description": "Standard workflow for handling customer support calls with security protocols",
                "department": "Customer Support",
                "criticality": "high"
            }
        },

        # Workflow Steps
        {
            "id": "step_1_answer",
            "label": "WorkflowStep",
            "properties": {
                "id": "WF001_S1",
                "step_number": 1,
                "name": "Answer and greet caller",
                "description": "Answer call professionally and introduce yourself",
                "required": True,
                "security_critical": False
            }
        },
        {
            "id": "step_2_verify",
            "label": "WorkflowStep",
            "properties": {
                "id": "WF001_S2",
                "step_number": 2,
                "name": "Verify caller identity",
                "description": "Verify caller identity using multi-factor authentication",
                "required": True,
                "security_critical": True  # CRITICAL STEP
            }
        },
        {
            "id": "step_3_document",
            "label": "WorkflowStep",
            "properties": {
                "id": "WF001_S3",
                "step_number": 3,
                "name": "Document call purpose",
                "description": "Document call purpose and customer request",
                "required": True,
                "security_critical": False
            }
        },
        {
            "id": "step_4_assist",
            "label": "WorkflowStep",
            "properties": {
                "id": "WF001_S4",
                "step_number": 4,
                "name": "Provide assistance",
                "description": "Provide requested information or assistance",
                "required": True,
                "security_critical": True
            }
        },
        {
            "id": "step_5_close",
            "label": "WorkflowStep",
            "properties": {
                "id": "WF001_S5",
                "step_number": 5,
                "name": "Close call",
                "description": "Confirm customer satisfaction and close call",
                "required": True,
                "security_critical": False
            }
        },

        # Security Behaviors
        {
            "id": "sb080_verify_caller",
            "label": "SecurityBehavior",
            "properties": {
                "id": "SB080",
                "title": "Verifies the legitimacy of an unknown caller",
                "description": "Vishing – deceptive phone calls from cybercriminals impersonating trusted organizations or services – is a common tactic used to steal sensitive information, authorize fraudulent payments, or gain remote access.",
                "category": "Threat Detection & Prevention",
                "tier": "Tier 1",
                "source_url": "https://www.us-cert.gov/ncas/tips/ST04-011"
            }
        },
        {
            "id": "sb091_limit_disclosure",
            "label": "SecurityBehavior",
            "properties": {
                "id": "SB091",
                "title": "Limits information disclosure to verified parties",
                "description": "Only share sensitive information after proper verification to prevent unauthorized access",
                "category": "Information Protection",
                "tier": "Tier 1"
            }
        },

        # Impacts
        {
            "id": "imp001_system_compromise",
            "label": "Impact",
            "properties": {
                "id": "IMP001",
                "name": "System compromise",
                "description": "Complete compromise of system security leading to unauthorized access"
            }
        },
        {
            "id": "imp002_data_breach",
            "label": "Impact",
            "properties": {
                "id": "IMP002",
                "name": "Data breach",
                "description": "Unauthorized disclosure of sensitive customer information"
            }
        }
    ],

    "relationships": [
        # Workflow -> Steps
        {
            "source_node_id": "workflow_customer_support",
            "target_node_id": "step_1_answer",
            "relationship_type": "HAS_STEP",
            "properties": {"order": 1}
        },
        {
            "source_node_id": "workflow_customer_support",
            "target_node_id": "step_2_verify",
            "relationship_type": "HAS_STEP",
            "properties": {"order": 2}
        },
        {
            "source_node_id": "workflow_customer_support",
            "target_node_id": "step_3_document",
            "relationship_type": "HAS_STEP",
            "properties": {"order": 3}
        },
        {
            "source_node_id": "workflow_customer_support",
            "target_node_id": "step_4_assist",
            "relationship_type": "HAS_STEP",
            "properties": {"order": 4}
        },
        {
            "source_node_id": "workflow_customer_support",
            "target_node_id": "step_5_close",
            "relationship_type": "HAS_STEP",
            "properties": {"order": 5}
        },

        # Steps -> SecurityBehaviors (REQUIRES_BEHAVIOR)
        {
            "source_node_id": "step_2_verify",
            "target_node_id": "sb080_verify_caller",
            "relationship_type": "REQUIRES_BEHAVIOR",
            "properties": {
                "mandatory": True,
                "skip_risk": "high"
            }
        },
        {
            "source_node_id": "step_4_assist",
            "target_node_id": "sb091_limit_disclosure",
            "relationship_type": "REQUIRES_BEHAVIOR",
            "properties": {
                "mandatory": True,
                "skip_risk": "high"
            }
        },

        # SecurityBehaviors -> Impacts (LEADS_TO with plausibility)
        {
            "source_node_id": "sb080_verify_caller",
            "target_node_id": "imp001_system_compromise",
            "relationship_type": "LEADS_TO",
            "properties": {
                "severity": "high",
                "likelihood": "medium",
                "plausibility": 3,
                "reasoning": "An attacker posing as IT support over the phone can trick an unverified employee into installing malicious remote access software, leading to direct system compromise."
            }
        },
        {
            "source_node_id": "sb091_limit_disclosure",
            "target_node_id": "imp002_data_breach",
            "relationship_type": "LEADS_TO",
            "properties": {
                "severity": "high",
                "likelihood": "high",
                "plausibility": 4,
                "reasoning": "Without proper verification, sensitive customer data could be disclosed to unauthorized parties, resulting in a data breach."
            }
        }
    ]
}

memory_request = {
    "content": content,
    "type": "text",
    "metadata": {
        "topics": ["workflow", "security", "customer_support", "training"],
        "createdAt": datetime.now().isoformat() + "Z",
        "location": "training_materials",
        "emoji_tags": ["📋", "🔒", "📞"],
        "emotion_tags": ["procedural", "protective"],
        "conversationId": f"workflow_example_{int(datetime.now().timestamp())}",
        "external_user_id": "security_analyst_001",
        "customMetadata": {
            "workflow_id": "WF001",
            "department": "Customer Support",
            "type": "workflow_definition"
        }
    },
    "graph_override": graph_override
}

def add_workflow():
    """Add the workflow to memory"""
    print("📋 Adding Customer Support Workflow with Security Behaviors")
    print("=" * 70)

    print(f"\n🎯 Graph Structure:")
    print(f"   • 1 Workflow: Customer Support Call Handling")
    print(f"   • 5 WorkflowSteps")
    print(f"   • 2 SecurityBehaviors (SB080, SB091)")
    print(f"   • 2 Impacts (System Compromise, Data Breach)")
    print(f"   • Connections: Workflow → Steps → Behaviors → Impacts")

    print(f"\n🚀 Sending request to memory server...")

    try:
        response = requests.post(
            f"{BASE_URL}/v1/memory",
            headers=HEADERS,
            json=memory_request,
            timeout=90
        )

        print(f"\n📊 Response Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if result.get('data') and len(result['data']) > 0:
                memory_data = result['data'][0]
                memory_id = memory_data.get('memoryId')

                print(f"✅ SUCCESS!")
                print(f"   Memory ID: {memory_id}")

                print(f"\n💡 Example Queries:")
                print(f"\n1️⃣ Check what happens if Step 2 is skipped:")
                print(f"""
MATCH (w:Workflow {{id: 'WF001'}})-[:HAS_STEP]->(s:WorkflowStep {{step_number: 2}})
MATCH (s)-[:REQUIRES_BEHAVIOR]->(sb:SecurityBehavior)
MATCH (sb)-[r:LEADS_TO]->(impact:Impact)
RETURN s.name as skipped_step,
       sb.id as behavior_id,
       sb.title as behavior,
       sb.tier as tier,
       impact.name as potential_impact,
       r.plausibility as plausibility,
       r.reasoning as reasoning
""")

                print(f"\n2️⃣ Find all security-critical steps in workflow:")
                print(f"""
MATCH (w:Workflow {{id: 'WF001'}})-[:HAS_STEP]->(s:WorkflowStep)
WHERE s.security_critical = true
OPTIONAL MATCH (s)-[:REQUIRES_BEHAVIOR]->(sb:SecurityBehavior)
RETURN s.step_number, s.name, sb.id, sb.title, sb.tier
ORDER BY s.step_number
""")

                print(f"\n3️⃣ Get risk assessment for entire workflow:")
                print(f"""
MATCH (w:Workflow {{id: 'WF001'}})-[:HAS_STEP]->(s:WorkflowStep)
MATCH (s)-[:REQUIRES_BEHAVIOR]->(sb:SecurityBehavior)-[r:LEADS_TO]->(i:Impact)
RETURN w.name as workflow,
       count(DISTINCT s) as critical_steps,
       collect(DISTINCT sb.tier) as security_tiers,
       avg(r.plausibility) as avg_risk_score,
       collect(DISTINCT i.name) as potential_impacts
""")

                return memory_id
            else:
                print(f"⚠️ No data in response")
                print(json.dumps(result, indent=2))
        else:
            print(f"❌ FAILED!")
            print(f"   Error: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    add_workflow()
