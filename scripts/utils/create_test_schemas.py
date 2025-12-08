"""
Create test schemas for call center workflows and security.
Run this to generate schema_ids that can be used for document uploads.
"""

import asyncio
import httpx
import json
from dotenv import load_dotenv
import os

# Load environment variables conditionally
use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
if use_dotenv:
    load_dotenv()

# Get credentials
API_KEY = os.getenv("TEST_X_USER_API_KEY", "YQnxXIZPT0M9JVH3L0S0MNLicDaqJ4Vd")
# Use PYTHON_SERVER_URL (Memory Server) not PARSE_SERVER_URL (Parse Server)
BASE_URL = os.getenv("PYTHON_SERVER_URL", "http://localhost:8000")

# Schema 1: Customer Support & Workflows
# Format matches UserGraphSchema model in models/user_schemas.py
CUSTOMER_SUPPORT_WORKFLOW_SCHEMA = {
    "name": "Customer Support & Workflows",
    "description": "Schema for customer support conversations, workflows, steps, and evidence tracking",
    "version": "1.0.0",
    "status": "active",
    "scope": "workspace",
    "node_types": {
        "CallSession": {
            "name": "CallSession",
            "label": "Call Session",  # Required: Display name
            "description": "A customer support interaction session",
            "properties": {
                "session_id": {"type": "string", "required": True},
                "started_at": {"type": "datetime", "required": True},
                "channel": {"type": "string", "required": True},
                "summary": {"type": "string", "required": False}
            }
        },
        "Agent": {
            "name": "Agent",
            "label": "Agent",  # Required: Display name
            "description": "A customer support agent",
            "properties": {
                "name": {"type": "string", "required": True},
                "role": {"type": "string", "required": False}
            }
        },
        "Workflow": {
            "name": "Workflow",
            "label": "Workflow",  # Required: Display name
            "description": "A standardized process workflow",
            "properties": {
                "name": {"type": "string", "required": True},
                "version": {"type": "string", "required": False},
                "purpose": {"type": "string", "required": False}
            }
        },
        "Step": {
            "name": "Step",
            "label": "Step",  # Required: Display name
            "description": "A single step in a workflow",
            "properties": {
                "ordinal": {"type": "integer", "required": True},
                "name": {"type": "string", "required": True},
                "description": {"type": "string", "required": False},
                "required": {"type": "boolean", "required": False}
            }
        },
        "Tool": {
            "name": "Tool",
            "label": "Tool",  # Required: Display name
            "description": "A tool or system used",
            "properties": {
                "name": {"type": "string", "required": True},
                "vendor": {"type": "string", "required": False}
            }
        }
    },
    "relationship_types": {
        "HAS_STEP": {
            "name": "HAS_STEP",
            "label": "Has Step",  # Required: Display name
            "description": "Workflow contains steps",
            "allowed_source_types": ["Workflow"],  # Required: Use allowed_source_types not source_node_type
            "allowed_target_types": ["Step"]  # Required: Use allowed_target_types not target_node_type
        },
        "HANDLED_BY": {
            "name": "HANDLED_BY",
            "label": "Handled By",  # Required: Display name
            "description": "CallSession handled by agent",
            "allowed_source_types": ["CallSession"],
            "allowed_target_types": ["Agent"]
        },
        "USES_TOOL": {
            "name": "USES_TOOL",
            "label": "Uses Tool",  # Required: Display name
            "description": "Agent uses tool",
            "allowed_source_types": ["Agent"],
            "allowed_target_types": ["Tool"]
        },
        "PERFORMS": {
            "name": "PERFORMS",
            "label": "Performs",  # Required: Display name
            "description": "Agent performs step",
            "allowed_source_types": ["Agent"],
            "allowed_target_types": ["Step"]
        }
    }
}

# Schema 2: Security Behaviors & Risk
# Format matches UserGraphSchema model in models/user_schemas.py
SECURITY_SCHEMA = {
    "name": "Security Behaviors & Risk",
    "description": "Schema for security protocols, controls, risk indicators, and impacts",
    "version": "1.0.0",
    "status": "active",
    "scope": "workspace",
    "node_types": {
        "SecurityBehavior": {
            "name": "SecurityBehavior",
            "label": "Security Behavior",  # Required: Display name
            "description": "A security behavior or protocol",
            "properties": {
                "name": {"type": "string", "required": True},
                "description": {"type": "string", "required": False},
                "tier": {"type": "integer", "required": False}
            }
        },
        "Control": {
            "name": "Control",
            "label": "Control",  # Required: Display name
            "description": "A security control mechanism",
            "properties": {
                "name": {"type": "string", "required": True},
                "category": {"type": "string", "required": True}
            }
        },
        "RiskIndicator": {
            "name": "RiskIndicator",
            "label": "Risk Indicator",  # Required: Display name
            "description": "A risk indicator or signal",
            "properties": {
                "code": {"type": "string", "required": True},
                "name": {"type": "string", "required": True},
                "severity": {"type": "integer", "required": True}
            }
        },
        "Impact": {
            "name": "Impact",
            "label": "Impact",  # Required: Display name
            "description": "A potential security impact",
            "properties": {
                "name": {"type": "string", "required": True},
                "description": {"type": "string", "required": False}
            }
        },
        "VerificationMethod": {
            "name": "VerificationMethod",
            "label": "Verification Method",  # Required: Display name
            "description": "A security verification method (MFA, KBA, etc)",
            "properties": {
                "name": {"type": "string", "required": True},
                "type": {"type": "string", "required": True}
            }
        }
    },
    "relationship_types": {
        "REQUIRES_VERIFICATION": {
            "name": "REQUIRES_VERIFICATION",
            "label": "Requires Verification",  # Required: Display name
            "description": "Security behavior requires verification method",
            "allowed_source_types": ["SecurityBehavior"],  # Required: Use allowed_source_types not source_node_type
            "allowed_target_types": ["VerificationMethod"]  # Required: Use allowed_target_types not target_node_type
        },
        "TRIGGERS_RISK": {
            "name": "TRIGGERS_RISK",
            "label": "Triggers Risk",  # Required: Display name
            "description": "Behavior triggers risk",
            "allowed_source_types": ["SecurityBehavior"],
            "allowed_target_types": ["RiskIndicator"]
        },
        "LEADS_TO_IMPACT": {
            "name": "LEADS_TO_IMPACT",
            "label": "Leads To Impact",  # Required: Display name
            "description": "Risk leads to impact",
            "allowed_source_types": ["RiskIndicator"],
            "allowed_target_types": ["Impact"]
        },
        "MITIGATES": {
            "name": "MITIGATES",
            "label": "Mitigates",  # Required: Display name
            "description": "Control mitigates risk",
            "allowed_source_types": ["Control"],
            "allowed_target_types": ["RiskIndicator"]
        }
    }
}


async def create_schema(schema_data: dict, name: str) -> str:
    """Create a schema and return its ID"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        headers = {
            "X-API-Key": API_KEY,
            "Content-Type": "application/json",
            "X-Client-Type": "test_client"
        }
        
        print(f"\n🔨 Creating schema: {name}")
        print(f"   Endpoint: {BASE_URL}/v1/schemas")
        
        response = await client.post(
            f"{BASE_URL}/v1/schemas",
            headers=headers,
            json=schema_data
        )
        
        if response.status_code in [200, 201]:
            result = response.json()
            # Try different response formats
            schema_id = (
                result.get("id") or 
                result.get("schema_id") or 
                result.get("data", {}).get("id") or
                result.get("data", {}).get("schema_id")
            )
            
            if schema_id:
                print(f"   ✅ Created: {schema_id}")
                return schema_id
            else:
                print(f"   ⚠️  Response: {json.dumps(result, indent=2)}")
                raise Exception(f"Schema ID not found in response")
        else:
            print(f"   ❌ Failed: {response.status_code}")
            print(f"   Response: {response.text}")
            raise Exception(f"Failed to create schema: {response.text}")


async def main():
    """Create all schemas"""
    print("=" * 80)
    print("🚀 CREATING TEST SCHEMAS")
    print("=" * 80)
    print(f"API Key: {API_KEY[:20]}...")
    print(f"Base URL: {BASE_URL}")
    
    try:
        # Create Schema 1: Workflow
        workflow_schema_id = await create_schema(
            CUSTOMER_SUPPORT_WORKFLOW_SCHEMA,
            "Customer Support & Workflows"
        )
        
        # Create Schema 2: Security
        security_schema_id = await create_schema(
            SECURITY_SCHEMA,
            "Security Behaviors & Risk"
        )
        
        print("\n" + "=" * 80)
        print("✅ ALL SCHEMAS CREATED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\n📋 Schema IDs:")
        print(f"   1. Workflow Schema: {workflow_schema_id}")
        print(f"   2. Security Schema: {security_schema_id}")
        
        print(f"\n📝 To use these schemas with document upload:")
        print(f"\n   # Using workflow schema:")
        print(f"   curl -X POST '{BASE_URL}/v1/document' \\")
        print(f"     -H 'X-API-Key: {API_KEY}' \\")
        print(f"     -F 'file=@call_answering_sop.pdf' \\")
        print(f"     -F 'metadata={{\"schema_id\": \"{workflow_schema_id}\"}}'")
        
        print(f"\n   # Using security schema:")
        print(f"   curl -X POST '{BASE_URL}/v1/document' \\")
        print(f"     -H 'X-API-Key: {API_KEY}' \\")
        print(f"     -F 'file=@two-factor_authentication.pdf' \\")
        print(f"     -F 'metadata={{\"schema_id\": \"{security_schema_id}\"}}'")
        
        print(f"\n📊 To verify in Neo4j after upload:")
        print(f"   MATCH (n) WHERE n.upload_id = '<upload_id>' RETURN DISTINCT labels(n) as node_types")
        print(f"\n   Expected for workflow schema: CallSession, Agent, Workflow, Step, Tool")
        print(f"   Expected for security schema: SecurityBehavior, Control, RiskIndicator, Impact")
        print(f"   NOT expected: Memory, Goal, UseCase (system defaults)")
        
        # Save IDs to file for easy reference
        with open("/tmp/schema_ids.txt", "w") as f:
            f.write(f"workflow_schema_id={workflow_schema_id}\n")
            f.write(f"security_schema_id={security_schema_id}\n")
        
        print(f"\n💾 Schema IDs saved to: /tmp/schema_ids.txt")
        
        return {
            "workflow_schema_id": workflow_schema_id,
            "security_schema_id": security_schema_id
        }
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = asyncio.run(main())
    if result:
        print(f"\n🎉 Success! Use these schema IDs for testing.")
    else:
        print(f"\n💥 Failed to create schemas. Check errors above.")

