#!/usr/bin/env python3
"""
Generate OpenAPI spec for schema routes

This script creates the OpenAPI documentation for the new schema endpoints
without importing problematic modules.
"""

import json
from typing import Dict, Any

def generate_schema_openapi() -> Dict[str, Any]:
    """Generate OpenAPI spec for schema routes"""
    
    schema_paths = {
        "/v1/schemas": {
            "post": {
                "tags": ["schemas"],
                "summary": "Create User Schema",
                "description": "Create a new user-defined graph schema",
                "operationId": "create_user_schema_v1",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/UserGraphSchema"
                            }
                        }
                    }
                },
                "responses": {
                    "201": {
                        "description": "Schema created successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/SchemaResponse"
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid schema data"
                    },
                    "401": {
                        "description": "Unauthorized"
                    },
                    "500": {
                        "description": "Internal server error"
                    }
                },
                "security": [
                    {"ApiKeyAuth": []},
                    {"SessionToken": []}
                ]
            },
            "get": {
                "tags": ["schemas"],
                "summary": "List User Schemas",
                "description": "Get all schemas for the authenticated user",
                "operationId": "list_user_schemas_v1",
                "parameters": [
                    {
                        "name": "status",
                        "in": "query",
                        "description": "Filter by schema status",
                        "required": False,
                        "schema": {
                            "type": "string",
                            "enum": ["active", "inactive", "draft"]
                        }
                    },
                    {
                        "name": "scope",
                        "in": "query", 
                        "description": "Filter by schema scope",
                        "required": False,
                        "schema": {
                            "type": "string",
                            "enum": ["personal", "workspace", "organization"]
                        }
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Schemas retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/SchemaListResponse"
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "Unauthorized"
                    },
                    "500": {
                        "description": "Internal server error"
                    }
                },
                "security": [
                    {"ApiKeyAuth": []},
                    {"SessionToken": []}
                ]
            }
        },
        "/v1/schemas/{schema_id}": {
            "get": {
                "tags": ["schemas"],
                "summary": "Get User Schema",
                "description": "Get a specific schema by ID",
                "operationId": "get_user_schema_v1",
                "parameters": [
                    {
                        "name": "schema_id",
                        "in": "path",
                        "description": "Schema ID",
                        "required": True,
                        "schema": {
                            "type": "string"
                        }
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Schema retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/SchemaResponse"
                                }
                            }
                        }
                    },
                    "404": {
                        "description": "Schema not found"
                    },
                    "401": {
                        "description": "Unauthorized"
                    },
                    "500": {
                        "description": "Internal server error"
                    }
                },
                "security": [
                    {"ApiKeyAuth": []},
                    {"SessionToken": []}
                ]
            },
            "put": {
                "tags": ["schemas"],
                "summary": "Update User Schema",
                "description": "Update an existing schema",
                "operationId": "update_user_schema_v1",
                "parameters": [
                    {
                        "name": "schema_id",
                        "in": "path",
                        "description": "Schema ID",
                        "required": True,
                        "schema": {
                            "type": "string"
                        }
                    }
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "$ref": "#/components/schemas/UserGraphSchema"
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Schema updated successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/SchemaResponse"
                                }
                            }
                        }
                    },
                    "404": {
                        "description": "Schema not found"
                    },
                    "401": {
                        "description": "Unauthorized"
                    },
                    "500": {
                        "description": "Internal server error"
                    }
                },
                "security": [
                    {"ApiKeyAuth": []},
                    {"SessionToken": []}
                ]
            },
            "delete": {
                "tags": ["schemas"],
                "summary": "Delete User Schema",
                "description": "Delete a schema by ID",
                "operationId": "delete_user_schema_v1",
                "parameters": [
                    {
                        "name": "schema_id",
                        "in": "path",
                        "description": "Schema ID",
                        "required": True,
                        "schema": {
                            "type": "string"
                        }
                    }
                ],
                "responses": {
                    "204": {
                        "description": "Schema deleted successfully"
                    },
                    "404": {
                        "description": "Schema not found"
                    },
                    "401": {
                        "description": "Unauthorized"
                    },
                    "500": {
                        "description": "Internal server error"
                    }
                },
                "security": [
                    {"ApiKeyAuth": []},
                    {"SessionToken": []}
                ]
            }
        },
        "/v1/schemas/{schema_id}/activate": {
            "post": {
                "tags": ["schemas"],
                "summary": "Activate User Schema",
                "description": "Activate a schema for use in graph generation",
                "operationId": "activate_user_schema_v1",
                "parameters": [
                    {
                        "name": "schema_id",
                        "in": "path",
                        "description": "Schema ID",
                        "required": True,
                        "schema": {
                            "type": "string"
                        }
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Schema activated successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/SchemaResponse"
                                }
                            }
                        }
                    },
                    "404": {
                        "description": "Schema not found"
                    },
                    "401": {
                        "description": "Unauthorized"
                    },
                    "500": {
                        "description": "Internal server error"
                    }
                },
                "security": [
                    {"ApiKeyAuth": []},
                    {"SessionToken": []}
                ]
            }
        }
    }
    
    schema_components = {
        "UserGraphSchema": {
            "type": "object",
            "required": ["name", "node_types", "relationship_types"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Schema name",
                    "minLength": 1,
                    "maxLength": 100
                },
                "description": {
                    "type": "string",
                    "description": "Schema description",
                    "maxLength": 500
                },
                "version": {
                    "type": "string",
                    "description": "Schema version",
                    "default": "1.0.0"
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive", "draft"],
                    "description": "Schema status",
                    "default": "active"
                },
                "scope": {
                    "type": "string",
                    "enum": ["personal", "workspace", "organization"],
                    "description": "Schema visibility scope",
                    "default": "personal"
                },
                "node_types": {
                    "type": "object",
                    "description": "Node type definitions",
                    "additionalProperties": {
                        "$ref": "#/components/schemas/UserNodeType"
                    }
                },
                "relationship_types": {
                    "type": "object",
                    "description": "Relationship type definitions",
                    "additionalProperties": {
                        "$ref": "#/components/schemas/UserRelationshipType"
                    }
                },
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Schema tags for categorization"
                }
            }
        },
        "UserNodeType": {
            "type": "object",
            "required": ["name", "label", "properties"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Node type name"
                },
                "label": {
                    "type": "string", 
                    "description": "Node label for display"
                },
                "properties": {
                    "type": "object",
                    "description": "Node property definitions",
                    "additionalProperties": {
                        "$ref": "#/components/schemas/PropertyDefinition"
                    }
                },
                "required_properties": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of required property names"
                }
            }
        },
        "UserRelationshipType": {
            "type": "object",
            "required": ["name", "label"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Relationship type name"
                },
                "label": {
                    "type": "string",
                    "description": "Relationship label for display"
                },
                "allowed_source_types": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Allowed source node types"
                },
                "allowed_target_types": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Allowed target node types"
                },
                "properties": {
                    "type": "object",
                    "description": "Relationship property definitions",
                    "additionalProperties": {
                        "$ref": "#/components/schemas/PropertyDefinition"
                    }
                }
            }
        },
        "PropertyDefinition": {
            "type": "object",
            "required": ["type"],
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["string", "integer", "float", "boolean", "datetime"],
                    "description": "Property data type"
                },
                "required": {
                    "type": "boolean",
                    "description": "Whether property is required",
                    "default": False
                },
                "description": {
                    "type": "string",
                    "description": "Property description"
                },
                "default": {
                    "description": "Default value for property"
                }
            }
        },
        "SchemaResponse": {
            "type": "object",
            "required": ["success", "data"],
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Operation success status"
                },
                "data": {
                    "allOf": [
                        {"$ref": "#/components/schemas/UserGraphSchema"},
                        {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Schema unique ID"
                                },
                                "created_at": {
                                    "type": "string",
                                    "format": "date-time",
                                    "description": "Schema creation timestamp"
                                },
                                "updated_at": {
                                    "type": "string",
                                    "format": "date-time", 
                                    "description": "Schema last update timestamp"
                                },
                                "user_id": {
                                    "type": "string",
                                    "description": "Schema owner user ID"
                                }
                            }
                        }
                    ]
                },
                "error": {
                    "type": "string",
                    "description": "Error message if operation failed"
                }
            }
        },
        "SchemaListResponse": {
            "type": "object",
            "required": ["success", "data"],
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Operation success status"
                },
                "data": {
                    "type": "array",
                    "items": {
                        "$ref": "#/components/schemas/SchemaResponse/properties/data"
                    },
                    "description": "List of user schemas"
                },
                "total": {
                    "type": "integer",
                    "description": "Total number of schemas"
                },
                "error": {
                    "type": "string",
                    "description": "Error message if operation failed"
                }
            }
        }
    }
    
    return {
        "paths": schema_paths,
        "components": {
            "schemas": schema_components
        }
    }

def merge_with_existing_openapi():
    """Merge schema routes with existing OpenAPI spec"""
    try:
        # Read existing OpenAPI spec
        with open('openapi.json', 'r') as f:
            existing_spec = json.load(f)
        
        # Generate schema spec
        schema_spec = generate_schema_openapi()
        
        # Merge paths
        if 'paths' not in existing_spec:
            existing_spec['paths'] = {}
        
        existing_spec['paths'].update(schema_spec['paths'])
        
        # Merge components/schemas
        if 'components' not in existing_spec:
            existing_spec['components'] = {}
        if 'schemas' not in existing_spec['components']:
            existing_spec['components']['schemas'] = {}
            
        existing_spec['components']['schemas'].update(schema_spec['components']['schemas'])
        
        # Add security schemes if not present
        if 'securitySchemes' not in existing_spec['components']:
            existing_spec['components']['securitySchemes'] = {}
            
        existing_spec['components']['securitySchemes'].update({
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key"
            },
            "SessionToken": {
                "type": "apiKey",
                "in": "header",
                "name": "X-Session-Token"
            }
        })
        
        # Write updated spec
        with open('openapi_with_schemas.json', 'w') as f:
            json.dump(existing_spec, f, indent=2)
            
        print("✅ OpenAPI spec updated with schema routes!")
        print(f"📝 New file: openapi_with_schemas.json")
        print(f"📋 Added {len(schema_spec['paths'])} schema endpoints")
        print(f"🔧 Added {len(schema_spec['components']['schemas'])} schema models")
        
        # List new endpoints
        print("\n🚀 New Schema Endpoints:")
        for path, methods in schema_spec['paths'].items():
            for method in methods.keys():
                operation_id = methods[method].get('operationId', 'unknown')
                summary = methods[method].get('summary', 'No description')
                print(f"   {method.upper()} {path} - {summary}")
        
        return True
        
    except FileNotFoundError:
        print("❌ openapi.json not found")
        return False
    except Exception as e:
        print(f"❌ Error updating OpenAPI spec: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Generating OpenAPI documentation for schema routes...")
    print("=" * 60)
    
    success = merge_with_existing_openapi()
    
    if success:
        print("\n🎉 SUCCESS!")
        print("✅ Schema routes are now documented in OpenAPI")
        print("✅ API clients can now discover and use schema endpoints")
        print("✅ Interactive API docs will show schema management")
        print("\n💡 Next steps:")
        print("   1. Replace openapi.json with openapi_with_schemas.json")
        print("   2. Update your API documentation")
        print("   3. Generate client SDKs with schema support")
    else:
        print("\n❌ Failed to update OpenAPI spec")
        print("Check the error messages above for details")







