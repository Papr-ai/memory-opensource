#!/usr/bin/env python3
"""
Comprehensive test suite for all memory processing paths.

This covers the different processing paths that memories can take:
1. Auto mode variations (default, schema_id, simple_schema_mode, property_overrides)
2. Manual mode (graph_override)
3. Error handling paths
4. Fallback scenarios
5. Edge cases
"""

import httpx
import asyncio
import os
import json
import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

BASE_URL = os.getenv("MEMORY_SERVER_URL", "http://localhost:8000")
TEST_API_KEY = os.getenv("PAPR_API_KEY", "f80c5a2940f21882420b41690522cb2c")
SCHEMA_ID = "IeskhPibBx"  # The security schema ID from previous tests

async def test_auto_mode_default():
    """Test 1: Auto mode with no graph_generation field (default behavior)"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    external_user_id = f"test_auto_default_{timestamp}"
    
    memory_data = {
        "type": "text",
        "content": f"[AUTO_DEFAULT {timestamp}] John completed the quarterly report for Project Alpha.",
        "metadata": {
            "event_type": "task_completion",
            "test_type": f"auto_default_{timestamp}",
            "external_user_id": external_user_id
        }
        # No graph_generation field = defaults to auto mode with AI selection
    }
    
    print(f"🧪 Test 1: Auto Mode Default (No graph_generation field)")
    print(f"   Expected Path: Auto mode with AI schema selection")
    
    return await make_memory_request(memory_data, external_user_id, "auto_default")

async def test_auto_mode_explicit():
    """Test 2: Auto mode explicitly specified with no config"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    external_user_id = f"test_auto_explicit_{timestamp}"
    
    memory_data = {
        "type": "text",
        "content": f"[AUTO_EXPLICIT {timestamp}] Sarah analyzed the customer feedback data.",
        "graph_generation": {
            "mode": "auto"
            # No auto config = pure AI
        },
        "metadata": {
            "event_type": "data_analysis",
            "test_type": f"auto_explicit_{timestamp}",
            "external_user_id": external_user_id
        }
    }
    
    print(f"🧪 Test 2: Auto Mode Explicit (Empty auto config)")
    print(f"   Expected Path: Auto mode with AI schema selection")
    
    return await make_memory_request(memory_data, external_user_id, "auto_explicit")

async def test_auto_mode_with_schema_id():
    """Test 3: Auto mode with specific schema_id"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    external_user_id = f"test_auto_schema_{timestamp}"
    
    memory_data = {
        "type": "text",
        "content": f"[AUTO_SCHEMA {timestamp}] Security alert: SQL injection detected on /api/users endpoint.",
        "graph_generation": {
            "mode": "auto",
            "auto": {
                "schema_id": SCHEMA_ID
            }
        },
        "metadata": {
            "event_type": "security_incident",
            "test_type": f"auto_schema_{timestamp}",
            "external_user_id": external_user_id
        }
    }
    
    print(f"🧪 Test 3: Auto Mode with Schema ID")
    print(f"   Schema ID: {SCHEMA_ID}")
    print(f"   Expected Path: Auto mode with specific schema enforcement")
    
    return await make_memory_request(memory_data, external_user_id, "auto_schema")

async def test_auto_mode_simple_schema():
    """Test 4: Auto mode with simple_schema_mode enabled"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    external_user_id = f"test_auto_simple_{timestamp}"
    
    memory_data = {
        "type": "text",
        "content": f"[AUTO_SIMPLE {timestamp}] Critical vulnerability found in authentication system.",
        "graph_generation": {
            "mode": "auto",
            "auto": {
                "schema_id": SCHEMA_ID,
                "simple_schema_mode": True
            }
        },
        "metadata": {
            "event_type": "security_incident",
            "test_type": f"auto_simple_{timestamp}",
            "external_user_id": external_user_id
        }
    }
    
    print(f"🧪 Test 4: Auto Mode with Simple Schema Mode")
    print(f"   Schema ID: {SCHEMA_ID}")
    print(f"   Simple Schema Mode: True")
    print(f"   Expected Path: Auto mode with limited schema complexity")
    
    return await make_memory_request(memory_data, external_user_id, "auto_simple")

async def test_auto_mode_property_overrides():
    """Test 5: Auto mode with property overrides"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    external_user_id = f"test_auto_props_{timestamp}"
    
    memory_data = {
        "type": "text",
        "content": f"[AUTO_PROPS {timestamp}] John completed the quarterly report for Project Alpha.",
        "graph_generation": {
            "mode": "auto",
            "auto": {
                "property_overrides": {
                    "Person": {
                        "id": f"person_john_{timestamp}",
                        "employee_id": "EMP001"
                    },
                    "Project": {
                        "id": f"project_alpha_{timestamp}",
                        "project_code": "PROJ-ALPHA-2024"
                    }
                }
            }
        },
        "metadata": {
            "event_type": "task_completion",
            "test_type": f"auto_props_{timestamp}",
            "external_user_id": external_user_id
        }
    }
    
    print(f"🧪 Test 5: Auto Mode with Property Overrides")
    print(f"   Property Overrides: Person.id, Person.employee_id, Project.id, Project.project_code")
    print(f"   Expected Path: Auto mode with developer property hints")
    
    return await make_memory_request(memory_data, external_user_id, "auto_props")

async def test_auto_mode_combined_config():
    """Test 6: Auto mode with all configuration options"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    external_user_id = f"test_auto_combined_{timestamp}"
    
    memory_data = {
        "type": "text",
        "content": f"[AUTO_COMBINED {timestamp}] Advanced persistent threat detected using social engineering tactics.",
        "graph_generation": {
            "mode": "auto",
            "auto": {
                "schema_id": SCHEMA_ID,
                "simple_schema_mode": True,
                "property_overrides": {
                    "SecurityBehavior": {
                        "id": f"behavior_apt_{timestamp}",
                        "severity": "critical"
                    },
                    "Tactic": {
                        "id": f"tactic_social_{timestamp}",
                        "category": "initial_access"
                    }
                }
            }
        },
        "metadata": {
            "event_type": "security_incident",
            "test_type": f"auto_combined_{timestamp}",
            "external_user_id": external_user_id
        }
    }
    
    print(f"🧪 Test 6: Auto Mode with Combined Configuration")
    print(f"   Schema ID: {SCHEMA_ID}")
    print(f"   Simple Schema Mode: True")
    print(f"   Property Overrides: SecurityBehavior, Tactic")
    print(f"   Expected Path: Auto mode with full configuration")
    
    return await make_memory_request(memory_data, external_user_id, "auto_combined")

async def test_manual_mode_minimal():
    """Test 7: Manual mode with minimal graph"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    external_user_id = f"test_manual_minimal_{timestamp}"
    
    memory_data = {
        "type": "text",
        "content": f"[MANUAL_MINIMAL {timestamp}] Simple structured data import.",
        "graph_generation": {
            "mode": "manual",
            "manual": {
                "nodes": [
                    {
                        "id": f"person_{timestamp}",
                        "label": "Person",
                        "properties": {
                            "name": "Jane Doe",
                            "role": "Analyst"
                        }
                    }
                ],
                "relationships": []  # No relationships
            }
        },
        "metadata": {
            "event_type": "data_import",
            "test_type": f"manual_minimal_{timestamp}",
            "external_user_id": external_user_id
        }
    }
    
    print(f"🧪 Test 7: Manual Mode Minimal (Single node, no relationships)")
    print(f"   Nodes: 1 (Person)")
    print(f"   Relationships: 0")
    print(f"   Expected Path: Manual mode with minimal graph")
    
    return await make_memory_request(memory_data, external_user_id, "manual_minimal")

async def test_manual_mode_complex():
    """Test 8: Manual mode with complex graph"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    external_user_id = f"test_manual_complex_{timestamp}"
    
    memory_data = {
        "type": "text",
        "content": f"[MANUAL_COMPLEX {timestamp}] Complex organizational structure import.",
        "graph_generation": {
            "mode": "manual",
            "manual": {
                "nodes": [
                    {
                        "id": f"person_manager_{timestamp}",
                        "label": "Person",
                        "properties": {
                            "name": "Alice Smith",
                            "role": "Manager",
                            "department": "Engineering"
                        }
                    },
                    {
                        "id": f"person_dev1_{timestamp}",
                        "label": "Person",
                        "properties": {
                            "name": "Bob Johnson",
                            "role": "Developer",
                            "skills": ["Python", "React"]
                        }
                    },
                    {
                        "id": f"person_dev2_{timestamp}",
                        "label": "Person",
                        "properties": {
                            "name": "Carol Wilson",
                            "role": "Developer",
                            "skills": ["Java", "Angular"]
                        }
                    },
                    {
                        "id": f"project_{timestamp}",
                        "label": "Project",
                        "properties": {
                            "name": "Mobile App Redesign",
                            "status": "in_progress",
                            "deadline": "2024-12-31"
                        }
                    }
                ],
                "relationships": [
                    {
                        "source_node_id": f"person_manager_{timestamp}",
                        "target_node_id": f"person_dev1_{timestamp}",
                        "relationship_type": "MANAGES"
                    },
                    {
                        "source_node_id": f"person_manager_{timestamp}",
                        "target_node_id": f"person_dev2_{timestamp}",
                        "relationship_type": "MANAGES"
                    },
                    {
                        "source_node_id": f"person_dev1_{timestamp}",
                        "target_node_id": f"project_{timestamp}",
                        "relationship_type": "WORKS_ON"
                    },
                    {
                        "source_node_id": f"person_dev2_{timestamp}",
                        "target_node_id": f"project_{timestamp}",
                        "relationship_type": "WORKS_ON"
                    }
                ]
            }
        },
        "metadata": {
            "event_type": "org_structure_import",
            "test_type": f"manual_complex_{timestamp}",
            "external_user_id": external_user_id
        }
    }
    
    print(f"🧪 Test 8: Manual Mode Complex (Multiple nodes and relationships)")
    print(f"   Nodes: 4 (3 Person, 1 Project)")
    print(f"   Relationships: 4 (MANAGES, WORKS_ON)")
    print(f"   Expected Path: Manual mode with complex graph structure")
    
    return await make_memory_request(memory_data, external_user_id, "manual_complex")

async def test_invalid_mode():
    """Test 9: Invalid mode (should fail gracefully)"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    external_user_id = f"test_invalid_mode_{timestamp}"
    
    memory_data = {
        "type": "text",
        "content": f"[INVALID_MODE {timestamp}] This should fail gracefully.",
        "graph_generation": {
            "mode": "invalid_mode",  # Invalid mode
            "auto": {
                "schema_id": SCHEMA_ID
            }
        },
        "metadata": {
            "event_type": "error_test",
            "test_type": f"invalid_mode_{timestamp}",
            "external_user_id": external_user_id
        }
    }
    
    print(f"🧪 Test 9: Invalid Mode (Error handling)")
    print(f"   Mode: invalid_mode")
    print(f"   Expected Path: Error handling / validation failure")
    
    return await make_memory_request(memory_data, external_user_id, "invalid_mode", expect_failure=True)

async def test_malformed_manual_config():
    """Test 10: Malformed manual configuration (should fail gracefully)"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    external_user_id = f"test_malformed_{timestamp}"
    
    memory_data = {
        "type": "text",
        "content": f"[MALFORMED {timestamp}] This has malformed manual config.",
        "graph_generation": {
            "mode": "manual",
            "manual": {
                "nodes": [
                    {
                        # Missing required fields
                        "label": "Person"
                        # No id or properties
                    }
                ],
                "relationships": [
                    {
                        # Missing required fields
                        "source_node_id": "nonexistent_id"
                        # No target_node_id or relationship_type
                    }
                ]
            }
        },
        "metadata": {
            "event_type": "error_test",
            "test_type": f"malformed_{timestamp}",
            "external_user_id": external_user_id
        }
    }
    
    print(f"🧪 Test 10: Malformed Manual Config (Error handling)")
    print(f"   Issues: Missing node id/properties, incomplete relationships")
    print(f"   Expected Path: Validation error / graceful failure")
    
    return await make_memory_request(memory_data, external_user_id, "malformed", expect_failure=True)

async def make_memory_request(memory_data, external_user_id, test_type, expect_failure=False):
    """Helper function to make memory request and handle response"""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{BASE_URL}/v1/memory",
                json=memory_data,
                params={"external_user_id": external_user_id},
                headers={
                    "Content-Type": "application/json",
                    "X-API-Key": TEST_API_KEY
                }
            )
            
            if expect_failure:
                if response.status_code != 200:
                    print(f"   ✅ EXPECTED FAILURE: {response.status_code}")
                    return {"success": True, "test_type": test_type, "expected_failure": True}
                else:
                    print(f"   ❌ UNEXPECTED SUCCESS: Expected failure but got 200")
                    return {"success": False, "error": "Expected failure but got success", "test_type": test_type}
            else:
                if response.status_code == 200:
                    result = response.json()
                    print(f"   ✅ SUCCESS!")
                    if result.get("data") and len(result["data"]) > 0:
                        memory_id = result["data"][0].get("memoryId")
                        print(f"   Memory ID: {memory_id}")
                        return {"success": True, "memory_id": memory_id, "test_type": test_type}
                    return {"success": True, "test_type": test_type}
                else:
                    print(f"   ❌ FAILED: {response.status_code}")
                    print(f"   Response: {response.text}")
                    return {"success": False, "error": response.text, "test_type": test_type}
                
    except Exception as e:
        print(f"   ❌ EXCEPTION: {str(e)}")
        return {"success": False, "error": str(e), "test_type": test_type}

async def main():
    """Run comprehensive memory processing path tests"""
    print("🚀 Comprehensive Memory Processing Path Tests")
    print("=" * 70)
    print("Testing all memory processing paths:")
    print("- Auto mode variations (default, explicit, schema_id, simple_schema_mode, property_overrides)")
    print("- Manual mode variations (minimal, complex)")
    print("- Error handling (invalid mode, malformed config)")
    print("=" * 70)
    
    tests = [
        ("Auto Mode Default (No graph_generation)", test_auto_mode_default),
        ("Auto Mode Explicit (Empty config)", test_auto_mode_explicit),
        ("Auto Mode with Schema ID", test_auto_mode_with_schema_id),
        ("Auto Mode with Simple Schema", test_auto_mode_simple_schema),
        ("Auto Mode with Property Overrides", test_auto_mode_property_overrides),
        ("Auto Mode Combined Config", test_auto_mode_combined_config),
        ("Manual Mode Minimal", test_manual_mode_minimal),
        ("Manual Mode Complex", test_manual_mode_complex),
        ("Invalid Mode (Error Test)", test_invalid_mode),
        ("Malformed Config (Error Test)", test_malformed_manual_config),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-' * 70}")
        try:
            result = await test_func()
            results.append(result)
        except Exception as e:
            print(f"   ❌ TEST FAILED: {str(e)}")
            results.append({"success": False, "error": str(e), "test_type": test_name})
    
    # Summary
    print(f"\n{'=' * 70}")
    print("🎯 COMPREHENSIVE TEST SUMMARY")
    print("=" * 70)
    
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    expected_failures = [r for r in successful if r.get("expected_failure")]
    actual_successes = [r for r in successful if not r.get("expected_failure")]
    
    print(f"✅ Total Successful: {len(successful)}/{len(results)}")
    print(f"   - Actual Successes: {len(actual_successes)}")
    print(f"   - Expected Failures: {len(expected_failures)}")
    print(f"❌ Unexpected Failures: {len(failed)}")
    
    if failed:
        print(f"\n❌ Unexpected Failures:")
        for result in failed:
            print(f"   - {result.get('test_type', 'Unknown')}: {result.get('error', 'Unknown error')}")
    
    if actual_successes:
        print(f"\n✅ Successful Memory Creations:")
        for result in actual_successes:
            memory_id = result.get('memory_id', 'N/A')
            print(f"   - {result.get('test_type', 'Unknown')}: {memory_id}")
    
    if expected_failures:
        print(f"\n✅ Expected Failures (Error Handling):")
        for result in expected_failures:
            print(f"   - {result.get('test_type', 'Unknown')}: Handled gracefully")
    
    print(f"\n📋 Check logs for detailed processing information")
    print(f"🔍 Search logs for memory IDs to verify different processing paths")
    
    return len(failed) == 0

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)




