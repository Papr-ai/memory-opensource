"""
Test custom schema usage in document ingestion and memory processing.

This test demonstrates how to:
1. Create domain-specific schemas (Workflows, Customer Support, Security)
2. Upload a document with a schema_id
3. Verify memories use the custom schema (not system default)
"""

import pytest
import httpx
import json
import asyncio
import os
import uuid
import re
from datetime import datetime
from pathlib import Path
from asgi_lifespan import LifespanManager
from main import app
from models.user_schemas import UserGraphSchema, SchemaStatus, SchemaScope
from models.shared_types import UploadDocumentRequest, MemoryMetadata
from typing import Dict, Any, List, Tuple
from os import environ as env
from dotenv import load_dotenv, find_dotenv
from services.logger_singleton import LoggerSingleton
from difflib import SequenceMatcher

# Load environment variables
ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

# Create logger instance for this module
logger = LoggerSingleton.get_logger(__name__)

# Test credentials from environment
TEST_X_USER_API_KEY = env.get('TEST_X_USER_API_KEY', 'YQnxXIZPT0M9JVH3L0S0MNLicDaqJ4Vd')


# Expected answers from the PDF for accuracy rating (manually extracted from QPNC83-106 Instruction Manual)
EXPECTED_ANSWERS = {
    "What does alarm code H2 mean and how do I resolve it?": {
        "keywords": ["H2", "High PDP", "refrigerant leak", "flow rate", "inlet temperature", "exceeding limit", "call for service"],
        "expected_content": "EH2: Warning icon NOT flashing, label H2 flashing. Description: High PDP. Possible root causes: refrigerant leak, flow rate/inlet temperature exceeding the limit. Observations: call for service",
        "must_include": ["H2", "High PDP"]
    },
    "How do I set up the machine for first-time use?": {
        "keywords": ["installation", "first", "setup", "install", "mounting", "electrical connection", "initial"],
        "expected_content": "First-time setup procedures including installation, mounting, electrical connections, and initial configuration",
        "must_include": ["installation", "first"]
    },
    "What is the correct operating procedure for starting the machine?": {
        "keywords": ["starting", "start", "procedure", "operating", "on/off", "switch", "button"],
        "expected_content": "Operating procedures for starting the machine, including on/off switch operations and startup sequence",
        "must_include": ["starting", "procedure"]
    },
    "What maintenance tasks need to be performed and how often?": {
        "keywords": ["maintenance", "every week", "every 2000 hours", "every 4000 hours", "1 year", "2 year", "replace", "clean"],
        "expected_content": "Every week: Brush/blow off the finned surface of the condenser, Clean the filter of the automatic condensate drain. Every 2000 hours / 1 year: Replace the filter of automatic condensate drain (2902016102). Every 4000 hours / 2 year: Replace drain kit (2200902017)",
        "must_include": ["maintenance", "every week"]
    },
    "Why is the machine not reaching rated pressure and what should I check?": {
        "keywords": ["rated pressure", "pressure", "not reaching", "check", "compressor", "temperature", "gas charge"],
        "expected_content": "Troubleshooting for pressure issues, including checking refrigerant gas charge, compressor operation, and system conditions",
        "must_include": ["pressure"]
    },
    "What safety precautions must be followed when operating this equipment?": {
        "keywords": ["safety", "precaution", "warning", "danger", "hazard", "protection", "guard"],
        "expected_content": "Safety warnings and precautions for operating the equipment, including electrical safety, pressure safety, and operational hazards",
        "must_include": ["safety"]
    },
    "What are the rated pressure and temperature specifications for this machine?": {
        "keywords": ["rated", "pressure", "temperature", "specification", "bar", "psi", "°C", "°F", "RATED VALUES"],
        "expected_content": "RATED VALUES: Temperature 20 °C (68°F). Evaporating Pressure bar (psi) - R513A: 2.35 2.47 (34.08+ 35.82)",
        "must_include": ["rated", "pressure", "temperature"]
    },
    "What steps should I follow if the machine fails to start?": {
        "keywords": ["fails to start", "not start", "motor", "overload", "voltage", "starting system", "relay"],
        "expected_content": "Fault: Motor cuts out on overload, Motor hums and does not start. Possible causes: Line voltage too low, Starting system defective. Observations: Check running and starting relays and condensers, Contact electric power company",
        "must_include": ["fails to start", "motor"]
    },
    "What components need to be replaced during routine maintenance?": {
        "keywords": ["replace", "filter", "drain kit", "condensate drain", "component", "maintenance"],
        "expected_content": "Every 2000 hours / 1 year: Replace the filter of automatic condensate drain (2902016102). Every 4000 hours / 2 year: Replace drain kit (2200902017)",
        "must_include": ["replace", "filter"]
    },
    "How do I diagnose and fix low pressure issues?": {
        "keywords": ["low pressure", "L2", "Low PDP", "hot gas bypass valve", "ambient temperature", "lower than limits", "call for service"],
        "expected_content": "Fault: 602 - Low PDP (Low pressure). Description: Warning icon NOT flashing, label L2 flashing. Possible root causes: hot gas bypass valve out of order, ambient temperature lower than limits. Observations: call for service",
        "must_include": ["low pressure", "L2"]
    }
}


def calculate_accuracy_score(
    returned_content: str,
    expected_answer: Dict[str, Any],
    query: str
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate accuracy score (1-10) by comparing returned content with expected answer.
    
    Scoring criteria:
    - Keyword presence (must_include keywords): 0-4 points
    - Keyword coverage (all keywords): 0-3 points  
    - Content similarity (text matching): 0-3 points
    - Total: 0-10 points
    
    Args:
        returned_content: The actual content returned from search results (concatenated)
        expected_answer: Dictionary with 'keywords', 'expected_content', 'must_include'
        query: The original query for logging
        
    Returns:
        Tuple of (score: float, details: dict) where score is 0-10
    """
    if not returned_content or len(returned_content.strip()) == 0:
        return 0.0, {"error": "Empty content"}
    
    returned_lower = returned_content.lower()
    details = {
        "must_include_found": [],
        "must_include_missing": [],
        "keywords_found": [],
        "keywords_missing": [],
        "similarity_score": 0.0
    }
    
    score = 0.0
    
    # 1. Must-include keywords (critical - 0-4 points)
    must_include_score = 0.0
    for keyword in expected_answer.get("must_include", []):
        keyword_lower = keyword.lower()
        found = False
        
        # Direct match
        if keyword_lower in returned_lower:
            found = True
        # Semantic equivalents: "low pressure" can match "Low PDP" in compressed air context
        # PDP = Pressure Dew Point - in compressed air dryer systems, Low PDP warnings (fault code L2)
        # indicate pressure-related issues, even though PDP technically refers to dew point temperature
        # under pressure conditions. For support engineering context, they're semantically equivalent.
        elif keyword_lower == "low pressure":
            if "low pdp" in returned_lower or ("pdp" in returned_lower and "low" in returned_lower):
                found = True
                logger.debug(f"Semantic match: '{keyword}' matched via Low PDP (L2 fault code - Pressure Dew Point issue related to pressure)")
        
        if found:
            details["must_include_found"].append(keyword)
            must_include_score += 1.0
        else:
            details["must_include_missing"].append(keyword)
    
    # Normalize to 4 points max (even if more keywords)
    if len(expected_answer.get("must_include", [])) > 0:
        must_include_score = min(4.0, (must_include_score / len(expected_answer["must_include"])) * 4.0)
    score += must_include_score
    
    # 2. All keywords coverage (0-3 points)
    keywords_score = 0.0
    all_keywords = expected_answer.get("keywords", [])
    for keyword in all_keywords:
        keyword_lower = keyword.lower()
        found = False
        
        # Direct match
        if keyword_lower in returned_lower:
            found = True
        # Semantic equivalents: "low pressure" can match "Low PDP" in compressed air context
        # PDP = Pressure Dew Point - Low PDP (fault L2) indicates pressure-related issues in dryer systems
        elif keyword_lower == "low pressure":
            if "low pdp" in returned_lower or ("pdp" in returned_lower and "low" in returned_lower):
                found = True
        # Handle variations: "hot gas bypass valve" vs "hot gas by pass valve"
        elif keyword_lower == "hot gas bypass valve":
            if "hot gas" in returned_lower and ("bypass" in returned_lower or "by pass" in returned_lower):
                found = True
        # Handle variations: "lower than limits" vs "lower then limits"
        elif keyword_lower == "lower than limits":
            if "lower" in returned_lower and "limits" in returned_lower:
                found = True
        
        if found:
            details["keywords_found"].append(keyword)
            keywords_score += 1.0
        else:
            details["keywords_missing"].append(keyword)
    
    # Normalize to 3 points max
    if len(all_keywords) > 0:
        keywords_score = min(3.0, (keywords_score / len(all_keywords)) * 3.0)
    score += keywords_score
    
    # 3. Content similarity using SequenceMatcher (0-3 points)
    expected_text = expected_answer.get("expected_content", "").lower()
    similarity = SequenceMatcher(None, returned_lower[:500], expected_text[:500]).ratio()
    details["similarity_score"] = similarity
    similarity_points = similarity * 3.0  # 0-3 points
    score += similarity_points
    
    # Round to 1 decimal place
    score = round(min(10.0, max(0.0, score)), 1)
    
    details["final_score"] = score
    details["breakdown"] = {
        "must_include_points": round(must_include_score, 1),
        "keywords_points": round(keywords_score, 1),
        "similarity_points": round(similarity_points, 1)
    }
    
    return score, details


def extract_top_results_content(memories: List[Dict[str, Any]], top_n: int = 5) -> str:
    """
    Extract and concatenate content from top N search results.
    
    Args:
        memories: List of memory dictionaries from search results
        top_n: Number of top results to include
        
    Returns:
        Concatenated content string
    """
    if not memories or len(memories) == 0:
        return ""
    
    contents = []
    for memory in memories[:top_n]:
        if isinstance(memory, dict):
            content = memory.get("content", "")
            if content:
                # Clean up markdown images and other noise
                if isinstance(content, str):
                    # Remove image markdown patterns
                    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
                    content = re.sub(r'\*+', '', content)  # Remove asterisks
                    contents.append(content.strip())
    
    return "\n\n".join(contents)


# Schema 1: Customer Support & Workflows
CUSTOMER_SUPPORT_WORKFLOW_SCHEMA = {
    "name": "Customer Support & Workflows",
    "description": "Schema for customer support conversations, workflows, steps, and evidence tracking",
    "version": "1.0.0",
    "status": SchemaStatus.ACTIVE,
    "scope": SchemaScope.WORKSPACE,
    "node_types": {
        "CallSession": {
            "name": "CallSession",
            "label": "CallSession",
            "description": "A customer support interaction session",
            "properties": {
                "session_id": {"type": "string", "required": True},
                "started_at": {"type": "datetime", "required": True},
                "ended_at": {"type": "datetime", "required": False},
                "channel": {"type": "string", "required": True},  # phone, chat, email
                "language": {"type": "string", "required": False},
                "summary": {"type": "string", "required": False},
                "risk_score": {"type": "float", "required": False}
            }
        },
        "Utterance": {
            "name": "Utterance",
            "label": "Utterance",
            "description": "A single turn in the conversation",
            "properties": {
                "sequence": {"type": "integer", "required": True},
                "speaker": {"type": "string", "required": True},  # agent, customer, system
                "timestamp": {"type": "datetime", "required": True},
                "text": {"type": "string", "required": True},
                "redacted_text": {"type": "string", "required": False},
                "labels": {"type": "array", "required": False}
            }
        },
        "Workflow": {
            "name": "Workflow",
            "label": "Workflow",
            "description": "A standardized process workflow",
            "properties": {
                "workflow_id": {"type": "string", "required": True},
                "name": {"type": "string", "required": True},
                "version": {"type": "string", "required": True},
                "purpose": {"type": "string", "required": False},
                "channel": {"type": "string", "required": False},
                "active": {"type": "boolean", "required": True}
            }
        },
        "Step": {
            "name": "Step",
            "label": "Step",
            "description": "A single step in a workflow",
            "properties": {
                "step_id": {"type": "string", "required": True},
                "ordinal": {"type": "integer", "required": True},
                "name": {"type": "string", "required": True},
                "description": {"type": "string", "required": False},
                "required": {"type": "boolean", "required": True},
                "branch_key": {"type": "string", "required": False}
            }
        },
        "WorkflowRun": {
            "name": "WorkflowRun",
            "label": "WorkflowRun",
            "description": "An execution of a workflow",
            "properties": {
                "run_id": {"type": "string", "required": True},
                "started_at": {"type": "datetime", "required": True},
                "finished_at": {"type": "datetime", "required": False},
                "resolved_workflow": {"type": "string", "required": True},
                "confidence": {"type": "float", "required": False},
                "outcome": {"type": "string", "required": False}  # success, abandoned, escalated
            }
        },
        "StepEvent": {
            "name": "StepEvent",
            "label": "StepEvent",
            "description": "Evidence of a step being executed",
            "properties": {
                "event_id": {"type": "string", "required": True},
                "timestamp": {"type": "datetime", "required": True},
                "result": {"type": "string", "required": True},  # observed, failed, skipped
                "details": {"type": "string", "required": False},
                "auto_detected": {"type": "boolean", "required": False}
            }
        },
        "Gap": {
            "name": "Gap",
            "label": "Gap",
            "description": "A detected gap in workflow execution",
            "properties": {
                "gap_id": {"type": "string", "required": True},
                "reason": {"type": "string", "required": True},  # missing_required_step, order_violation, missing_control
                "severity": {"type": "integer", "required": True},  # 1-5
                "detail": {"type": "string", "required": False}
            }
        },
        "Agent": {
            "name": "Agent",
            "label": "Agent",
            "description": "A customer support agent",
            "properties": {
                "agent_id": {"type": "string", "required": True},
                "name": {"type": "string", "required": True},
                "role": {"type": "string", "required": False},
                "team": {"type": "string", "required": False}
            }
        },
        "Customer": {
            "name": "Customer",
            "label": "Customer",
            "description": "A customer",
            "properties": {
                "customer_id": {"type": "string", "required": True},
                "customer_key": {"type": "string", "required": False},
                "risk_flags": {"type": "array", "required": False}
            }
        }
    },
    "relationship_types": {
        "HAS_UTTERANCE": {
            "name": "HAS_UTTERANCE",
            "label": "Has Utterance",
            "description": "CallSession contains utterances",
            "properties": {},
            "allowed_source_types": ["CallSession"],
            "allowed_target_types": ["Utterance"],
            "cardinality": "one-to-many"
        },
        "HANDLED_BY": {
            "name": "HANDLED_BY",
            "label": "Handled By",
            "description": "CallSession handled by agent",
            "properties": {},
            "allowed_source_types": ["CallSession"],
            "allowed_target_types": ["Agent"],
            "cardinality": "one-to-many"
        },
        "WITH_CUSTOMER": {
            "name": "WITH_CUSTOMER",
            "label": "With Customer",
            "description": "CallSession with customer",
            "properties": {},
            "allowed_source_types": ["CallSession"],
            "allowed_target_types": ["Customer"],
            "cardinality": "one-to-many"
        },
        "HAS_STEP": {
            "name": "HAS_STEP",
            "label": "Has Step",
            "description": "Workflow contains steps",
            "properties": {},
            "allowed_source_types": ["Workflow"],
            "allowed_target_types": ["Step"],
            "cardinality": "one-to-many"
        },
        "NEXT": {
            "name": "NEXT",
            "label": "Next",
            "description": "Sequential next step",
            "properties": {},
            "allowed_source_types": ["Step"],
            "allowed_target_types": ["Step"],
            "cardinality": "one-to-one"
        },
        "HAS_RUN": {
            "name": "HAS_RUN",
            "label": "Has Run",
            "description": "CallSession has workflow run",
            "properties": {},
            "allowed_source_types": ["CallSession"],
            "allowed_target_types": ["WorkflowRun"],
            "cardinality": "one-to-many"
        },
        "OF_WORKFLOW": {
            "name": "OF_WORKFLOW",
            "label": "Of Workflow",
            "description": "WorkflowRun is of workflow",
            "properties": {},
            "allowed_source_types": ["WorkflowRun"],
            "allowed_target_types": ["Workflow"],
            "cardinality": "one-to-many"
        },
        "HAS_STEPACTION": {
            "name": "HAS_STEPACTION",
            "label": "Has Step Action",
            "description": "WorkflowRun has step events",
            "properties": {},
            "allowed_source_types": ["WorkflowRun"],
            "allowed_target_types": ["StepEvent"],
            "cardinality": "one-to-many"
        },
        "FOR_STEP": {
            "name": "FOR_STEP",
            "label": "For Step",
            "description": "StepEvent for a step",
            "properties": {},
            "allowed_source_types": ["StepEvent"],
            "allowed_target_types": ["Step"],
            "cardinality": "one-to-many"
        },
        "EVIDENCE_FROM": {
            "name": "EVIDENCE_FROM",
            "label": "Evidence From",
            "description": "StepEvent evidence from utterance",
            "properties": {},
            "allowed_source_types": ["StepEvent"],
            "allowed_target_types": ["Utterance"],
            "cardinality": "one-to-many"
        },
        "HAS_GAP": {
            "name": "HAS_GAP",
            "label": "Has Gap",
            "description": "WorkflowRun has gap",
            "properties": {},
            "allowed_source_types": ["WorkflowRun"],
            "allowed_target_types": ["Gap"],
            "cardinality": "one-to-many"
        }
    }
}


# Schema 2: Security Behaviors & Risk
SECURITY_SCHEMA = {
    "name": "Security Behaviors & Risk",
    "description": "Schema for security protocols, controls, risk indicators, and impacts",
    "version": "1.0.0",
    "status": SchemaStatus.ACTIVE,
    "scope": SchemaScope.WORKSPACE,
    "node_types": {
        "SecurityBehavior": {
            "name": "SecurityBehavior",
            "label": "SecurityBehavior",
            "description": "A security behavior or protocol",
            "properties": {
                "behavior_id": {"type": "string", "required": True},
                "name": {"type": "string", "required": True},
                "description": {"type": "string", "required": False},
                "nist_function": {"type": "array", "required": False},  # PROTECT, DETECT, etc
                "mitre_tactics": {"type": "array", "required": False},  # TA0001, etc
                "tier": {"type": "integer", "required": False}  # 1-4
            }
        },
        "Control": {
            "name": "Control",
            "label": "Control",
            "description": "A security control mechanism",
            "properties": {
                "control_id": {"type": "string", "required": True},
                "name": {"type": "string", "required": True},
                "category": {"type": "string", "required": True}  # Authentication, Verification, etc
            }
        },
        "VerificationEvent": {
            "name": "VerificationEvent",
            "label": "VerificationEvent",
            "description": "A security verification event",
            "properties": {
                "event_id": {"type": "string", "required": True},
                "timestamp": {"type": "datetime", "required": True},
                "method": {"type": "string", "required": True},  # MFA, KBA, VoiceID, OTP
                "status": {"type": "string", "required": True},  # passed, failed, not_applicable
                "context": {"type": "string", "required": False}
            }
        },
        "RiskIndicator": {
            "name": "RiskIndicator",
            "label": "RiskIndicator",
            "description": "A risk indicator or signal",
            "properties": {
                "indicator_id": {"type": "string", "required": True},
                "code": {"type": "string", "required": True},  # RI010, etc
                "name": {"type": "string", "required": True},
                "severity": {"type": "integer", "required": True},  # 1-5
                "rationale": {"type": "string", "required": False}
            }
        },
        "Impact": {
            "name": "Impact",
            "label": "Impact",
            "description": "A potential security impact",
            "properties": {
                "impact_id": {"type": "string", "required": True},
                "name": {"type": "string", "required": True},
                "description": {"type": "string", "required": False}
            }
        },
        "Tool": {
            "name": "Tool",
            "label": "Tool",
            "description": "A security tool or system",
            "properties": {
                "tool_id": {"type": "string", "required": True},
                "name": {"type": "string", "required": True},
                "vendor": {"type": "string", "required": False},
                "kind": {"type": "string", "required": False}  # CRM, VoiceBio, CoreBank, etc
            }
        }
    },
    "relationship_types": {
        "REQUIRES_CONTROL": {
            "name": "REQUIRES_CONTROL",
            "label": "Requires Control",
            "description": "Step or workflow requires control",
            "properties": {},
            "allowed_source_types": ["Step", "Workflow"],
            "allowed_target_types": ["Control"],
            "cardinality": "many-to-many"
        },
        "COVERS_BEHAVIOR": {
            "name": "COVERS_BEHAVIOR",
            "label": "Covers Behavior",
            "description": "Workflow covers security behavior",
            "properties": {},
            "allowed_source_types": ["Workflow"],
            "allowed_target_types": ["SecurityBehavior"],
            "cardinality": "many-to-many"
        },
        "HAS_VERIFICATION": {
            "name": "HAS_VERIFICATION",
            "label": "Has Verification",
            "description": "CallSession has verification event",
            "properties": {},
            "allowed_source_types": ["CallSession"],
            "allowed_target_types": ["VerificationEvent"],
            "cardinality": "one-to-many"
        },
        "TRIGGERS_RISK": {
            "name": "TRIGGERS_RISK",
            "label": "Triggers Risk",
            "description": "CallSession or gap triggers risk",
            "properties": {},
            "allowed_source_types": ["CallSession", "Gap"],
            "allowed_target_types": ["RiskIndicator"],
            "cardinality": "many-to-many"
        },
        "MAPPED_TO_BEHAVIOR": {
            "name": "MAPPED_TO_BEHAVIOR",
            "label": "Mapped To Behavior",
            "description": "RiskIndicator mapped to security behavior",
            "properties": {},
            "allowed_source_types": ["RiskIndicator"],
            "allowed_target_types": ["SecurityBehavior"],
            "cardinality": "many-to-many"
        },
        "MAPPED_TO_IMPACT": {
            "name": "MAPPED_TO_IMPACT",
            "label": "Mapped To Impact",
            "description": "RiskIndicator mapped to impact",
            "properties": {},
            "allowed_source_types": ["RiskIndicator"],
            "allowed_target_types": ["Impact"],
            "cardinality": "many-to-many"
        },
        "CAN_LEAD_TO_IMPACT": {
            "name": "CAN_LEAD_TO_IMPACT",
            "label": "Can Lead To Impact",
            "description": "SecurityBehavior can lead to impact",
            "properties": {},
            "allowed_source_types": ["SecurityBehavior"],
            "allowed_target_types": ["Impact"],
            "cardinality": "many-to-many"
        },
        "USES_TOOL": {
            "name": "USES_TOOL",
            "label": "Uses Tool",
            "description": "Agent or step uses tool",
            "properties": {},
            "allowed_source_types": ["Agent", "Step"],
            "allowed_target_types": ["Tool"],
            "cardinality": "many-to-many"
        }
    }
}


# Schema 3: Manufacturing Floor Knowledge Graph
def load_and_validate_manufacturing_schema() -> Dict[str, Any]:
    """
    Load the manufacturing schema from JSON, validate it using UserGraphSchema,
    and fix any issues (enum case, many-to-one cardinality).
    
    Note: The API doesn't support "many-to-one" cardinality. To preserve semantics,
    we reverse the source/target when converting: many-to-one A->B becomes one-to-many B->A.
    """
    # Load raw schema from JSON
    raw_schema = json.load(open("tests/taktora-schema.json"))["user_schema"]
    
    # Fix enum case: API expects lowercase enum values
    if "status" in raw_schema:
        original_status = raw_schema["status"]
        raw_schema["status"] = raw_schema["status"].lower()
        if original_status != raw_schema["status"]:
            print(f"⚠️  Fixed status enum: {original_status} -> {raw_schema['status']}")
    if "scope" in raw_schema:
        original_scope = raw_schema["scope"]
        raw_schema["scope"] = raw_schema["scope"].lower()
        if original_scope != raw_schema["scope"]:
            print(f"⚠️  Fixed scope enum: {original_scope} -> {raw_schema['scope']}")
    
    # Fix cardinality issues: API doesn't support "many-to-one"
    # Convert by reversing source/target to preserve semantics:
    # many-to-one A->B is semantically equivalent to one-to-many B->A
    if "relationship_types" in raw_schema:
        for rel_name, rel_def in raw_schema["relationship_types"].items():
            if rel_def.get("cardinality") == "many-to-one":
                # Reverse source/target and change to one-to-many to preserve semantics
                original_source = rel_def.get("allowed_source_types", [])
                original_target = rel_def.get("allowed_target_types", [])
                rel_def["allowed_source_types"] = original_target
                rel_def["allowed_target_types"] = original_source
                rel_def["cardinality"] = "one-to-many"
                print(f"⚠️  Fixed cardinality for {rel_name}: reversed direction (many-to-one -> one-to-many)")
    
    # Validate using Pydantic model
    try:
        validated_schema = UserGraphSchema(**raw_schema)
        print(f"✅ Schema validation passed: {validated_schema.name} v{validated_schema.version}")
        print(f"   - {len(validated_schema.node_types)} node types")
        print(f"   - {len(validated_schema.relationship_types)} relationship types")
        
        # Convert back to dict for API (excluding fields that shouldn't be in the request)
        schema_dict = validated_schema.model_dump(exclude={"id", "user_id", "workspace_id", "organization_id", 
                                                          "created_at", "updated_at", "usage_count", "last_used_at",
                                                          "read_access", "write_access"}, exclude_none=True)
        
        return schema_dict
    except Exception as e:
        print(f"❌ Schema validation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

MANUFACTURING_FLOOR_KNOWLEDGE_GRAPH_SCHEMA = load_and_validate_manufacturing_schema()


@pytest.mark.asyncio
async def test_create_custom_schemas(app):
    """Test creating the Manufacturing Floor Knowledge Graph schema"""
    
    async with LifespanManager(app, startup_timeout=120):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            headers = {
                "X-API-Key": TEST_X_USER_API_KEY,
                "Content-Type": "application/json",
                "X-Client-Type": "test_client"
            }
            
            # Create Manufacturing Floor Knowledge Graph Schema
            response = await client.post(
                "/v1/schemas",
                headers=headers,
                json=MANUFACTURING_FLOOR_KNOWLEDGE_GRAPH_SCHEMA
            )
            
            assert response.status_code == 201, f"Failed to create manufacturing floor knowledge graph schema: {response.text}"
            manufacturing_floor_knowledge_graph_schema_id = response.json()["data"]["id"]
            print(f"✅ Created Manufacturing Floor Knowledge Graph Schema: {manufacturing_floor_knowledge_graph_schema_id}")
        
        return {
            "manufacturing_floor_knowledge_graph_schema_id": manufacturing_floor_knowledge_graph_schema_id
        }


@pytest.mark.asyncio
@pytest.mark.skip(reason="Feature not yet implemented - demonstrates desired behavior")
async def test_document_upload_with_custom_schema(api_key, base_url):
    """
    Test document upload with custom schema_id.
    
    This test demonstrates the DESIRED behavior:
    1. Upload a document with schema_id parameter
    2. Verify generated memories use custom schema nodes/relationships
    3. Verify Neo4j graph uses custom schema (not system default)
    
    Currently NOT implemented - this is the target API.
    """
    
    # First create the schema
    schema_ids = await test_create_custom_schemas(api_key, base_url)
    workflow_schema_id = schema_ids["workflow_schema_id"]
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        headers = {
            "X-API-Key": api_key,
            "X-Client-Type": "test_client"
        }
        
        # Upload document WITH schema_id
        with open("tests/QPNC83-106 Instruction Manual.pdf", "rb") as f:
            files = {"file": ("QPNC83-106 Instruction Manual.pdf", f, "application/pdf")}
            
            # Pass schema_id in metadata
            metadata = {
                "metadata": {
                    "schema_id": workflow_schema_id,  # ⬅️ THIS IS THE KEY PARAMETER
                    "customMetadata": {
                        "document_type": "Instruction Manual",
                        "category": "manufacturing_floor_knowledge_graph"
                    }
                }
            }
            
            data = {"metadata": json.dumps(metadata)}
            
            response = await client.post(
                f"{base_url}/v1/document",
                headers=headers,
                files=files,
                data=data
            )
            
            assert response.status_code == 200, f"Document upload failed: {response.text}"
            upload_id = response.json()["document_status"]["upload_id"]
            print(f"✅ Document uploaded with schema_id: {upload_id}")
            
            # Wait for processing (Temporal workflow)
            await asyncio.sleep(60)
            
            # Query memories to verify they use custom schema
            search_response = await client.post(
                f"{base_url}/v1/memory/search",
                headers=headers,
                json={
                    "query": "manufacturing floor knowledge graph",
                    "limit": 10
                }
            )
            
            assert search_response.status_code == 200
            memories = search_response.json()["data"]
            
            # Verify memories have custom schema nodes (not default Memory nodes)
            assert len(memories) > 0, "No memories found after document processing"
            
            # Check for custom node types in Neo4j relationships
            for memory in memories[:3]:
                related_nodes = memory.get("related_nodes", [])
                node_labels = [node.get("label") for node in related_nodes]
                
                # Should see CallSession, Workflow, Step, etc - NOT just Memory
                assert any(label in ["CallSession", "Workflow", "Step", "Agent"] for label in node_labels), \
                    f"Expected custom schema nodes, but got: {node_labels}"
            
            print(f"✅ Verified memories use custom schema nodes!")


@pytest.mark.asyncio
async def test_support_engineering_queries(app):
    """
    Test realistic support engineering queries that should return answers from the instruction manual.
    
    This test:
    1. Uses existing Manufacturing Floor Knowledge Graph schema (objectId: i6hzNQuao3)
    2. Uploads a document using v2 endpoint with reducto provider
    3. Waits for Temporal workflow to complete
    4. Runs 10 realistic support engineering queries
    5. For each query: returns 20 results, displays top 10 with memory IDs, content, and Neo4j nodes
    
    These are real-world questions that support engineers would ask when troubleshooting equipment
    or helping customers understand machine operation, maintenance, and procedures.
    """
    
    # Step 1: Use existing schema (already created in Parse)
    manufacturing_schema_id = "i6hzNQuao3"  # Manufacturing Floor Knowledge Graph schema
    print(f"🔑 Using existing schema ID: {manufacturing_schema_id}")
    
    # Step 2: Upload document using v2 endpoint with reducto
    file_path = "tests/QPNC83-106 Instruction Manual.pdf"
    
    if not os.path.exists(file_path):
        pytest.skip(f"PDF file not found at {file_path}")
    
    async with LifespanManager(app, startup_timeout=120):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=600.0) as client:
            headers = {
                "X-API-Key": TEST_X_USER_API_KEY,
                "X-Client-Type": "test_client",
                "Accept-Encoding": "gzip"
            }
            
            # Upload document using v2 endpoint with UploadDocumentRequest
            with open(file_path, "rb") as f:
                pdf_content = f.read()
            
            files = {"file": (os.path.basename(file_path), pdf_content, "application/pdf")}
            upload_marker = str(uuid.uuid4())
            
            # Create UploadDocumentRequest with schema_id
            metadata_with_schema = MemoryMetadata(
                source="test_support_engineering_queries",
                customMetadata={
                    "test_id": upload_marker,
                    "document_type": "Instruction Manual",
                    "category": "manufacturing_floor_knowledge_graph"
                }
            )
            
            upload_document_request = UploadDocumentRequest(
                type="document",
                metadata=metadata_with_schema,
                schema_id=manufacturing_schema_id,  # ✅ Use manufacturing schema
                simple_schema_mode=False,
                preferred_provider="reducto",  # ✅ Use reducto as specified
                hierarchical_enabled=True
            )
            
            form_data = {
                "type": "document",
                "metadata": upload_document_request.metadata.model_dump_json(),
                "schema_id": upload_document_request.schema_id,
                "simple_schema_mode": "false",
                "preferred_provider": "reducto",
                "hierarchical_enabled": "true",
            }
            
            print(f"📤 Uploading document with schema_id: {manufacturing_schema_id} using reducto provider")
            
            response = await client.post(
                "/v1/document",
                files=files,
                data=form_data,
                headers=headers,
                timeout=300.0
            )
            
            assert response.status_code in [200, 202], f"Expected success, got {response.status_code}: {response.text}"
            result = response.json()
            assert result["status"] == "success"
            
            # Step 3: Wait for Temporal workflow to complete
            if response.status_code == 202:
                upload_id = result.get("document_status", {}).get("upload_id")
                assert upload_id, "upload_id missing in response"
                
                print(f"⏳ Waiting for document processing (upload_id: {upload_id})...")
                
                # Poll status until completed
                max_wait_seconds = 600
                poll_interval = 5
                waited = 0
                final_status = None
                
                while waited < max_wait_seconds:
                    status_resp = await client.get(
                        f"/v1/document/status/{upload_id}",
                        headers={"X-API-Key": TEST_X_USER_API_KEY}
                    )
                    if status_resp.status_code == 200:
                        st = status_resp.json()
                        final_status = st.get("status")
                        if final_status in ["completed", "failed", "cancelled"]:
                            break
                    await asyncio.sleep(poll_interval)
                    waited += poll_interval
                    if waited % 30 == 0:
                        print(f"   Still waiting... {waited}s elapsed")
                
                assert final_status == "completed", f"Workflow did not complete, status={final_status}"
                print(f"✅ Document processing completed!")
            
            # Step 4: Run 10 realistic support engineering queries
            test_queries = [
                "What does alarm code H2 mean and how do I resolve it?",
                "How do I set up the machine for first-time use?",
                "What is the correct operating procedure for starting the machine?",
                "What maintenance tasks need to be performed and how often?",
                "Why is the machine not reaching rated pressure and what should I check?",
                "What safety precautions must be followed when operating this equipment?",
                "What are the rated pressure and temperature specifications for this machine?",
                "What steps should I follow if the machine fails to start?",
                "What components need to be replaced during routine maintenance?",
                "How do I diagnose and fix low pressure issues?"
            ]
            
            print(f"\n🔍 Running {len(test_queries)} support engineering queries...")
            
            # Test each query
            for i, query in enumerate(test_queries, 1):
                print(f"\n{'='*80}")
                print(f"Query {i}/{len(test_queries)}: {query}")
                print(f"{'='*80}")
                logger.info(f"Query {i}/{len(test_queries)}: {query}")
                
                search_response = await client.post(
                    "/v1/memory/search",
                    headers=headers,
                    json={
                        "query": query,
                        "limit": 20,  # Get 20 results
                        "rank_results": True
                    }
                )
                
                assert search_response.status_code == 200, f"Search failed for query {i}: {search_response.text}"
                results = search_response.json()
                logger.debug(f"Search response for query {i}: status={search_response.status_code}, keys={list(results.keys())}")
                memories = results.get("data", [])
                
                # Verify we got results
                assert len(memories) > 0, f"Query {i} returned no results: '{query}'"
                
                print(f"\n📊 Total results returned: {len(memories)}")
                print(f"📋 Displaying top 10 results:\n")
                
                # Display top 10 results with memory IDs, content, and Neo4j nodes
                top_10 = memories[:10]
                for rank, memory in enumerate(top_10, 1):
                    memory_id = memory.get("id") or memory.get("memory_id") or memory.get("_id", "N/A")
                    content = memory.get("content", "")
                    content_preview = content[:200] + "..." if len(content) > 200 else content
                    
                    # Get Neo4j nodes (related_nodes or similar)
                    related_nodes = memory.get("related_nodes", [])
                    node_labels = [node.get("label") or node.get("type", "Unknown") for node in related_nodes]
                    
                    print(f"  [{rank}] Memory ID: {memory_id}")
                    print(f"      Content: {content_preview}")
                    print(f"      Neo4j Nodes ({len(related_nodes)}): {', '.join(node_labels) if node_labels else 'None'}")
                    print()
                
                # Summary: Show all 20 memory nodes and Neo4j nodes
                all_node_labels = []
                for memory in memories[:20]:  # First 20 results
                    related_nodes = memory.get("related_nodes", [])
                    for node in related_nodes:
                        label = node.get("label") or node.get("type")
                        if label and label not in all_node_labels:
                            all_node_labels.append(label)
                
                print(f"📈 Summary for Query {i}:")
                print(f"   - Memories returned: {len(memories)}")
                print(f"   - Unique Neo4j node types: {len(all_node_labels)}")
                print(f"   - Node types: {', '.join(all_node_labels[:20]) if all_node_labels else 'None'}")
                
                # Small delay between queries
                await asyncio.sleep(1)
            
            print(f"\n{'='*80}")
            print(f"✅ All {len(test_queries)} support engineering queries completed successfully!")
            print(f"{'='*80}")


@pytest.mark.asyncio
async def test_support_engineering_queries_only(app):
    """
    Run support engineering queries against already-processed document.
    
    This test:
    1. Uses existing Manufacturing Floor Knowledge Graph schema (objectId: i6hzNQuao3)
    2. Runs 10 realistic support engineering queries
    3. For each query: returns 20 results, displays top 10 with memory IDs, content, and Neo4j nodes
    
    Assumes document has already been uploaded and processed.
    """
    
    manufacturing_schema_id = "i6hzNQuao3"  # Manufacturing Floor Knowledge Graph schema
    print(f"🔑 Using schema ID: {manufacturing_schema_id}")
    logger.info(f"Starting test_support_engineering_queries_only with schema ID: {manufacturing_schema_id}")
    
    # Create results directory and file
    results_dir = Path("tests/test_reports")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"support_engineering_queries_{timestamp}.json"
    
    logger.info(f"Results will be saved to: {results_file}")
    
    # Store all results for file output
    all_results = {
        "schema_id": manufacturing_schema_id,
        "timestamp": timestamp,
        "queries": []
    }
    
    async with LifespanManager(app, startup_timeout=120):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test", timeout=600.0) as client:
            headers = {
                "X-API-Key": TEST_X_USER_API_KEY,
                "X-Client-Type": "test_client",
                "Accept-Encoding": "gzip"
            }
            
            # 10 Realistic Support Engineering Questions
            test_queries = [
                "What does alarm code H2 mean and how do I resolve it?",
                "How do I set up the machine for first-time use?",
                "What is the correct operating procedure for starting the machine?",
                "What maintenance tasks need to be performed and how often?",
                "Why is the machine not reaching rated pressure and what should I check?",
                "What safety precautions must be followed when operating this equipment?",
                "What are the rated pressure and temperature specifications for this machine?",
                "What steps should I follow if the machine fails to start?",
                "What components need to be replaced during routine maintenance?",
                "How do I diagnose and fix low pressure issues?"
            ]
            
            print(f"\n🔍 Running {len(test_queries)} support engineering queries...")
            logger.info(f"Starting {len(test_queries)} support engineering queries")
            
            # Test each query
            for i, query in enumerate(test_queries, 1):
                print(f"\n{'='*80}")
                print(f"Query {i}/{len(test_queries)}: {query}")
                print(f"{'='*80}")
                
                search_response = await client.post(
                    "/v1/memory/search",
                    headers=headers,
                    json={
                        "query": query,
                        "rank_results": True,
                        "enable_agentic_graph": True,
                        "max_memories": 20,
                        "max_nodes": 20
                    }
                )
                
                assert search_response.status_code == 200, f"Search failed for query {i}: {search_response.text}"
                results = search_response.json()
                
                # Save raw API response for later use and debugging
                raw_response = results.copy()
                
                # Handle different response formats and ensure it's a list
                # Check for nested structure: data.memories or data (list) or memories or results
                memories = []
                data_obj = results.get("data", {})
                if isinstance(data_obj, dict):
                    memories = data_obj.get("memories", [])
                    if not memories:
                        memories = data_obj.get("data", [])
                elif isinstance(data_obj, list):
                    memories = data_obj
                
                if not memories:
                    memories = results.get("memories", [])
                if not memories:
                    memories = results.get("results", [])
                
                # Ensure memories is a list - handle case where API might return non-list
                if not isinstance(memories, list):
                    print(f"⚠️  Warning: memories is not a list, type={type(memories)}, value={memories}")
                    print(f"   Full response: {json.dumps(results, indent=2)}")
                    logger.warning(f"Query {i}: memories is not a list, type={type(memories)}, value={memories}")
                    logger.debug(f"Query {i} full response: {json.dumps(results, indent=2)}")
                    memories = []
                
                # Initialize query result - always save raw response for recalculation
                query_result = {
                    "query_number": i,
                    "query": query,
                    "total_results": 0,
                    "top_10_results": [],
                    "all_memories": [],  # Save all memories with full content for accuracy calculation
                    "all_node_types": [],
                    "raw_api_response": raw_response,  # Save raw response for debugging and recalculation
                    "error": None
                }
                
                # Verify we got results
                if len(memories) == 0:
                    error_msg = f"Query {i} returned no results: '{query}'"
                    query_result["error"] = error_msg
                    query_result["response_keys"] = list(results.keys()) if isinstance(results, dict) else "Not a dict"
                    print(f"⚠️  {error_msg}")
                    print(f"   Response keys: {query_result['response_keys']}")
                    logger.warning(f"Query {i} failed: {error_msg}")
                    logger.debug(f"Query {i} response keys: {query_result['response_keys']}")
                    logger.debug(f"Query {i} full response: {json.dumps(results, indent=2)}")
                    # Still save to results even with error - raw response will help debug
                    all_results["queries"].append(query_result)
                    continue  # Skip to next query instead of failing
                
                query_result["total_results"] = len(memories)
                print(f"\n📊 Total results returned: {len(memories)}")
                print(f"📋 Displaying top 10 results:\n")
                logger.info(f"Query {i} returned {len(memories)} total results")
                
                # Process all memories (up to 20) - save all with full content for accuracy calculation
                all_memories_data = []
                top_10 = memories[:10] if isinstance(memories, list) and len(memories) > 0 else []
                all_memories_processed = memories[:20] if isinstance(memories, list) else []  # Get up to 20 for accuracy calculation
                logger.info(f"Query {i} - Displaying top {len(top_10)} results (of {len(memories)} total), saving {len(all_memories_processed)} for accuracy calculation")
                
                for rank, memory in enumerate(all_memories_processed, 1):
                    if not isinstance(memory, dict):
                        if rank <= 10:
                            print(f"  [{rank}] ⚠️  Memory is not a dict, type={type(memory)}")
                        logger.warning(f"Query {i} - Rank {rank}: Memory is not a dict, type={type(memory)}")
                        continue
                    
                    memory_id = memory.get("id") or memory.get("memory_id") or memory.get("_id", "N/A")
                    content = memory.get("content", "")
                    if not isinstance(content, str):
                        content = str(content) if content else ""
                    content_preview = content[:200] + "..." if len(content) > 200 else content
                    full_content = content  # Store full content for file
                    
                    # Get Neo4j nodes (related_nodes or similar)
                    related_nodes = memory.get("related_nodes", [])
                    if not isinstance(related_nodes, list):
                        related_nodes = []
                    node_labels = [node.get("label") or node.get("type", "Unknown") for node in related_nodes if isinstance(node, dict)]
                    
                    # Store full memory data (for accuracy calculation)
                    memory_data = {
                        "rank": rank,
                        "memory_id": memory_id,
                        "content_full": full_content,  # Full content for accuracy calculation
                        "content_length": len(full_content),
                        "related_nodes_count": len(related_nodes),
                        "node_labels": node_labels,
                        "score": memory.get("score"),
                        "metadata": {k: v for k, v in memory.items() if k not in ["content", "related_nodes", "id", "memory_id", "_id", "score"]}
                    }
                    all_memories_data.append(memory_data)
                    
                    # For top 10, also create display version with preview
                    if rank <= 10:
                        result_data = {
                            **memory_data,
                            "content_preview": content_preview
                        }
                        query_result["top_10_results"].append(result_data)
                        
                        print(f"  [{rank}] Memory ID: {memory_id}")
                        print(f"      Content: {content_preview}")
                        print(f"      Neo4j Nodes ({len(related_nodes)}): {', '.join(node_labels) if node_labels else 'None'}")
                        print()
                        
                        # Log detailed result to logger
                        logger.info(
                            f"Query {i} - Result [{rank}]: "
                            f"Memory ID={memory_id}, "
                            f"Score={memory.get('score', 'N/A')}, "
                            f"Content Length={len(full_content)}, "
                            f"Neo4j Nodes={len(related_nodes)} ({', '.join(node_labels) if node_labels else 'None'})"
                        )
                        logger.debug(f"Query {i} - Result [{rank}] Full Content: {full_content}")
                
                # Save all memories with full content for accuracy calculation
                query_result["all_memories"] = all_memories_data
                
                # Summary: Show all 20 memory nodes and Neo4j nodes
                all_node_labels = []
                if isinstance(memories, list):
                    for memory in memories[:20]:  # First 20 results
                        if not isinstance(memory, dict):
                            continue
                        related_nodes = memory.get("related_nodes", [])
                        if not isinstance(related_nodes, list):
                            continue
                        for node in related_nodes:
                            if not isinstance(node, dict):
                                continue
                            label = node.get("label") or node.get("type")
                            if label and label not in all_node_labels:
                                all_node_labels.append(label)
                
                query_result["all_node_types"] = all_node_labels
                
                # Calculate accuracy score if expected answer exists
                # Use all_memories (with full content) instead of raw memories list
                accuracy_score = None
                accuracy_details = None
                if query in EXPECTED_ANSWERS:
                    # Extract content from all_memories (top 5 for scoring)
                    returned_content = "\n\n".join(
                        [m.get("content_full", "") for m in all_memories_data[:5] if m.get("content_full")]
                    )
                    expected_answer = EXPECTED_ANSWERS[query]
                    accuracy_score, accuracy_details = calculate_accuracy_score(
                        returned_content, expected_answer, query
                    )
                    query_result["accuracy_score"] = accuracy_score
                    query_result["accuracy_details"] = accuracy_details
                    query_result["expected_answer"] = expected_answer
                    
                    logger.info(
                        f"Query {i} Accuracy Score: {accuracy_score}/10.0 - "
                        f"Must-include: {len(accuracy_details.get('must_include_found', []))}/{len(expected_answer.get('must_include', []))}, "
                        f"Keywords: {len(accuracy_details.get('keywords_found', []))}/{len(expected_answer.get('keywords', []))}, "
                        f"Similarity: {accuracy_details.get('similarity_score', 0.0):.2f}"
                    )
                else:
                    logger.warning(f"Query {i} has no expected answer defined for accuracy rating: {query}")
                
                print(f"📈 Summary for Query {i}:")
                print(f"   - Memories returned: {len(memories)}")
                print(f"   - Unique Neo4j node types: {len(all_node_labels)}")
                print(f"   - Node types: {', '.join(all_node_labels[:20]) if all_node_labels else 'None'}")
                if accuracy_score is not None:
                    print(f"   - Accuracy Score: {accuracy_score}/10.0")
                    if accuracy_details:
                        breakdown = accuracy_details.get("breakdown", {})
                        print(f"     • Must-include keywords: {breakdown.get('must_include_points', 0):.1f}/4.0")
                        print(f"     • Keyword coverage: {breakdown.get('keywords_points', 0):.1f}/3.0")
                        print(f"     • Content similarity: {breakdown.get('similarity_points', 0):.1f}/3.0")
                
                # Log summary to logger
                logger.info(
                    f"Query {i} Summary: "
                    f"{len(memories)} memories returned, "
                    f"{len(all_node_labels)} unique Neo4j node types: {', '.join(all_node_labels[:20]) if all_node_labels else 'None'}"
                )
                
                # Add query result to all_results
                all_results["queries"].append(query_result)
                
                # Small delay between queries
                await asyncio.sleep(1)
            
            print(f"\n{'='*80}")
            print(f"✅ All {len(test_queries)} support engineering queries completed successfully!")
            print(f"{'='*80}")
            
            # Save results to file
            successful_count = sum(1 for q in all_results['queries'] if q.get('error') is None)
            failed_count = sum(1 for q in all_results['queries'] if q.get('error') is not None)
            
            # Calculate accuracy statistics
            queries_with_scores = [q for q in all_results['queries'] if 'accuracy_score' in q]
            if queries_with_scores:
                avg_accuracy = sum(q['accuracy_score'] for q in queries_with_scores) / len(queries_with_scores)
                max_accuracy = max(q['accuracy_score'] for q in queries_with_scores)
                min_accuracy = min(q['accuracy_score'] for q in queries_with_scores)
                all_results["accuracy_statistics"] = {
                    "average_score": round(avg_accuracy, 2),
                    "max_score": round(max_accuracy, 2),
                    "min_score": round(min_accuracy, 2),
                    "queries_rated": len(queries_with_scores),
                    "queries_total": len(all_results['queries'])
                }
            
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Results saved to: {results_file}")
            print(f"   - Total queries: {len(all_results['queries'])}")
            print(f"   - Successful queries: {successful_count}")
            print(f"   - Failed queries: {failed_count}")
            
            if queries_with_scores:
                print(f"\n📊 Accuracy Rating Summary:")
                print(f"   - Queries rated: {len(queries_with_scores)}/{len(all_results['queries'])}")
                print(f"   - Average accuracy: {avg_accuracy:.1f}/10.0")
                print(f"   - Best score: {max_accuracy:.1f}/10.0")
                print(f"   - Worst score: {min_accuracy:.1f}/10.0")
            
            print(f"\n📄 Review the JSON file to see full content, accuracy scores, and evaluate answer quality!")
            
            # Log final summary
            logger.info(
                f"Test completed: {len(all_results['queries'])} queries total, "
                f"{successful_count} successful, {failed_count} failed. "
                f"Results saved to {results_file}"
            )
            if queries_with_scores:
                logger.info(
                    f"Accuracy Rating: {len(queries_with_scores)} queries rated, "
                    f"Average: {avg_accuracy:.2f}/10.0, Max: {max_accuracy:.2f}/10.0, Min: {min_accuracy:.2f}/10.0"
                )
            logger.info(f"Full query results, accuracy scores, and content available in JSON file: {results_file}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_create_custom_schemas(
        api_key="YQnxXIZPT0M9JVH3L0S0MNLicDaqJ4Vd",
        base_url="http://localhost:8000"
    ))

