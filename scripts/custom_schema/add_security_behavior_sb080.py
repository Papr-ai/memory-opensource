#!/usr/bin/env python3
"""
Add Security Behavior SB080 to memory using graph_override
Demonstrates the complete security schema with SecurityBehavior, Impact, MITRE tactics
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

# Structured content from your example
content = """Verifying the legitimacy of an unknown caller means an individual, when receiving an unexpected phone call, takes active steps to confirm the caller is genuine before sharing any sensitive information or complying with requests. This involves critically assessing the nature of the call (especially if it creates urgency or asks for personal data), questioning the caller to ascertain their identity and purpose, and, if any doubt exists, independently looking up an official contact number for the purported organization to initiate a call back for verification.

Security Behavior: SB080 - Verifies the legitimacy of an unknown caller

Vishing – deceptive phone calls from cybercriminals impersonating trusted organizations or services – is a common tactic used to steal sensitive information, authorize fraudulent payments, or gain remote access. Caller ID can be spoofed, making it unreliable for verification. Not confirming a caller's identity risks identity theft, financial loss, or the compromise of accounts and devices.

Source: https://www.us-cert.gov/ncas/tips/ST04-011
Category: Threat Detection & Prevention, Safe Online Practices & Communication
Tier: Tier 1
Impact: System compromise [IMP001]
Plausibility: 3/5
Reasoning: An attacker posing as IT support over the phone can trick an unverified employee into installing malicious remote access software, leading to direct system compromise."""

# Graph override structure following your security schema
graph_override = {
    "nodes": [
        {
            "id": "sb080_verify_caller",
            "label": "SecurityBehavior",
            "properties": {
                "id": "SB080",  # unique_identifier for MERGE
                "title": "Verifies the legitimacy of an unknown caller",
                "description": "Vishing – deceptive phone calls from cybercriminals impersonating trusted organizations or services – is a common tactic used to steal sensitive information, authorize fraudulent payments, or gain remote access. Caller ID can be spoofed, making it unreliable for verification.",
                "category": "Threat Detection & Prevention, Safe Online Practices & Communication",
                "tier": "Tier 1",
                "source_url": "https://www.us-cert.gov/ncas/tips/ST04-011"
            }
        },
        {
            "id": "imp001_system_compromise",
            "label": "Impact",
            "properties": {
                "id": "IMP001",  # unique_identifier for MERGE
                "name": "System compromise",
                "description": "Complete compromise of system security leading to unauthorized access"
            }
        },
        {
            "id": "ta0043_reconnaissance",
            "label": "Tactic",
            "properties": {
                "id": "TA0043",  # unique_identifier for MERGE
                "name": "Reconnaissance",
                "description": "Gathering information to plan future operations"
            }
        },
        {
            "id": "ta0001_initial_access",
            "label": "Tactic",
            "properties": {
                "id": "TA0001",  # unique_identifier for MERGE
                "name": "Initial Access",
                "description": "Trying to get into your network"
            }
        },
        {
            "id": "ta0006_credential_access",
            "label": "Tactic",
            "properties": {
                "id": "TA0006",  # unique_identifier for MERGE
                "name": "Credential Access",
                "description": "Stealing account names and passwords"
            }
        }
    ],
    "relationships": [
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
            "source_node_id": "sb080_verify_caller",
            "target_node_id": "ta0043_reconnaissance",
            "relationship_type": "MAPS_TO",
            "properties": {
                "attack_phase": "initial"
            }
        },
        {
            "source_node_id": "sb080_verify_caller",
            "target_node_id": "ta0001_initial_access",
            "relationship_type": "MAPS_TO",
            "properties": {
                "attack_phase": "entry"
            }
        },
        {
            "source_node_id": "sb080_verify_caller",
            "target_node_id": "ta0006_credential_access",
            "relationship_type": "MAPS_TO",
            "properties": {
                "attack_phase": "exploitation"
            }
        }
    ]
}

# Memory request
memory_request = {
    "content": content,
    "type": "text",
    "metadata": {
        "topics": ["security", "vishing", "phone_security", "social_engineering", "verification"],
        "createdAt": datetime.now().isoformat() + "Z",
        "location": "security_training",
        "emoji_tags": ["📞", "🛡️", "⚠️"],
        "emotion_tags": ["vigilant", "cautious", "protective"],
        "conversationId": f"security_behavior_{int(datetime.now().timestamp())}",
        "external_user_id": "security_analyst_001",  # Added for external user access
        "customMetadata": {
            "behavior_id": "SB080",
            "category": "Threat Detection & Prevention",
            "tier": "Tier 1",
            "source": "US-CERT",
            "plausibility_score": 3
        }
    },
    "graph_override": graph_override
}

def add_security_behavior():
    """Add the security behavior to memory using graph_override"""
    print("🔒 Adding Security Behavior SB080 to Memory")
    print("=" * 70)

    print(f"\n📝 Content Preview:")
    print(f"   {content[:150]}...")

    print(f"\n🎯 Graph Override Structure:")
    print(f"   Nodes: {len(graph_override['nodes'])}")
    for node in graph_override['nodes']:
        print(f"      🔵 {node['label']}: {node['properties'].get('name') or node['properties'].get('title')}")

    print(f"\n   Relationships: {len(graph_override['relationships'])}")
    for rel in graph_override['relationships']:
        print(f"      🔗 {rel['source_node_id']} --{rel['relationship_type']}--> {rel['target_node_id']}")

    print(f"\n🚀 Sending request to memory server...")

    try:
        response = requests.post(
            f"{BASE_URL}/v1/memory",
            headers=HEADERS,
            json=memory_request,
            timeout=30
        )

        print(f"\n📊 Response Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if result.get('data') and len(result['data']) > 0:
                memory_data = result['data'][0]
                memory_id = memory_data.get('memoryId')
                created_at = memory_data.get('createdAt')

                print(f"✅ SUCCESS!")
                print(f"   Memory ID: {memory_id}")
                print(f"   Created At: {created_at}")
                print(f"\n🔍 Graph Structure Created:")
                print(f"   • SecurityBehavior: SB080 (Verify Caller)")
                print(f"   • Impact: IMP001 (System Compromise)")
                print(f"   • MITRE Tactics: TA0043, TA0001, TA0006")
                print(f"   • Relationships: LEADS_TO, MAPS_TO")

                print(f"\n💡 You can now:")
                print(f"   1. View in Neo4j Browser")
                print(f"   2. Search: 'SB080 caller verification vishing'")
                print(f"   3. Query related security behaviors")

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
    add_security_behavior()
