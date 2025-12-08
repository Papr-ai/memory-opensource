# tests/test_memory_graph.py

import json
import pytest
from unittest.mock import MagicMock, patch
from memory.memory_graph import MemoryGraph  # Adjust the import path as needed
from services.logging_config import get_logger

# Create a logger instance for this module
logger = get_logger(__name__)  # Will use 'tests.test_memory_graph' as the logger name

# Fixtures to mock external dependencies

@pytest.fixture
def mock_pinecone():
    with patch('memory.memory_graph.Pinecone') as mock_pinecone:
        mock_instance = MagicMock()
        mock_pinecone.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_neo_conn():
    """
    Fixture to mock the Neo4j connection and session.
    """
    mock_session = MagicMock()
    mock_connection = MagicMock()
    mock_connection.session.return_value.__enter__.return_value = mock_session
    return mock_connection, mock_session

@pytest.fixture
async def memory_graph(mock_neo_conn, mock_pinecone):
    """
    Fixture to create an instance of MemoryGraph with mocked dependencies.
    """
    mock_connection, mock_session = mock_neo_conn
    mg = MemoryGraph()
    mg.neo_conn = mock_connection
    try:
        yield mg
    finally:
        await mg.cleanup()

def test_validate_metadata_success(memory_graph, capsys):
    """
    Test that valid metadata is processed correctly.
    """
    metadata = {
        'id': '12345',
        'user_id': 'user_1',
        'count': 10,
        'is_active': True,
        'tags': ['tag1', 'tag2'],
        'details': {
            'description': 'A sample memory item',
            'priority': 'high'
        }
    }

    # Mock the flatten_dict method if it's complex
    mock_flat_metadata = {
        'id': '12345',
        'user_id': 'user_1',
        'count': 10,
        'is_active': True,
        'tags': ['tag1', 'tag2'],
        'details_description': 'A sample memory item',
        'details_priority': 'high'
    }
    memory_graph.flatten_dict = lambda x: mock_flat_metadata

    result = memory_graph.validate_metadata(metadata)

    # Capture printed output
    captured = capsys.readouterr()
    assert "Original metadata:" in captured.out
    assert "Flattened metadata:" in captured.out
    assert result == mock_flat_metadata

def test_validate_metadata_invalid_key(memory_graph):
    """
    Test that metadata with a non-string key raises TypeError.
    """
    metadata = {
        123: 'invalid_key',
        'valid_key': 'valid_value'
    }

    with pytest.raises(TypeError) as exc_info:
        memory_graph.validate_metadata(metadata)

    assert "Invalid type for key" in str(exc_info.value)

def test_validate_metadata_invalid_value(memory_graph):
    """
    Test that metadata with an unsupported value type raises TypeError.
    """
    metadata = {
        'valid_key': 'valid_value',
        'invalid_value': {'unexpected': 'dict'}
    }

    with pytest.raises(TypeError) as exc_info:
        memory_graph.validate_metadata(metadata)

    assert "Invalid type for value" in str(exc_info.value)

def test_validate_metadata_empty(memory_graph, capsys):
    """
    Test that empty metadata is handled correctly.
    """
    metadata = {}

    # Mock the flatten_dict method
    mock_flat_metadata = {}
    memory_graph.flatten_dict = lambda x: mock_flat_metadata

    result = memory_graph.validate_metadata(metadata)

    # Capture printed output
    captured = capsys.readouterr()
    assert "Original metadata:" in captured.out
    assert "Flattened metadata:" in captured.out
    assert result == mock_flat_metadata

def test_validate_metadata_nested_flattening(memory_graph, capsys):
    """
    Test that nested metadata is flattened correctly.
    """
    metadata = {
        'a': 1,
        'b': {
            'c': 2,
            'd': {
                'e': 3
            }
        },
        'f': [4, 5, 6]
    }

    # Mock the flatten_dict method
    mock_flat_metadata = {
        'a': 1,
        'b_c': 2,
        'b_d_e': 3,
        'f': [4, 5, 6]
    }
    memory_graph.flatten_dict = lambda x: mock_flat_metadata

    result = memory_graph.validate_metadata(metadata)

    # Capture printed output
    captured = capsys.readouterr()
    assert "Original metadata:" in captured.out
    assert "Flattened metadata:" in captured.out
    assert result == mock_flat_metadata

def test_add_memory_item_to_neo4j_success(memory_graph, mocker):
    """
    Test adding a new memory item successfully.
    """
    # Sample memory item
    memory_item = {
        'id': 'bf80fb32-2d8c-414a-94e1-a3edffe46573',
        'type': 'messageList',
        'context': {'some': 'context'},
        'metadata': {
            'hierarchical structures': 'Slack Message',
            'sourceType': 'slack',
            'sourceUrl': 'https://paprbot.slack.com/archives/C1CB1HG13/p1727192750065499',
            'workspace_id': 'HXPpCEuF8N',
            'acl_object_ids': ['mhnkVbAdgG'],
            'type': 'messageList',
            'authed_user_id': 'U18PZHLFN',
            'members': ['U18PZHLFN', 'U1B7DCML7', 'U1CQY8EQ5', 'U2CHU4H6G'],
            'user_ids': ['U18PZHLFN'],
            'client_msg_ids': ['c95b55df-68f4-4747-a78e-49a2956a9a70'],
            'source_urls': ['https://paprbot.slack.com/archives/C1CB1HG13/p1727192750065499'],
            'imageGenerationCategory': 'None',
            'user_id': 'mhnkVbAdgG',
            'user_read_access': ['mhnkVbAdgG'],
            'user_write_access': ['mhnkVbAdgG'],
            'workspace_read_access': ['HXPpCEuF8N'],
            'workspace_write_access': ['HXPpCEuF8N'],
            'role_read_access': [],
            'role_write_access': [],
            'pageId': 'None'
        }
    }

    # Mock the check query to return no existing node
    mock_session = memory_graph.neo_conn.session().__enter__.return_value
    mock_session.run.return_value.single.return_value = None

    # Mock the run query to return a fake node
    fake_node = {'n': {'id': memory_item['id'], 'type': memory_item['type'], 'context': json.dumps(memory_item['context'])}}
    mock_session.run.return_value.single.return_value = fake_node

    # Call the method
    result = memory_graph.add_memory_item_to_neo4j(memory_item)

    # Assertions
    assert result is not None
    assert result['n']['id'] == memory_item['id']
    assert result['n']['type'] == memory_item['type']
    assert result['n']['context'] == json.dumps(memory_item['context'])

    # Verify that session.run was called with correct parameters
    expected_query = """
                MERGE (n:Memory {id: $id})
                SET n += $properties
                RETURN n
            """
    expected_properties = {
        "id": memory_item['id'],
        "type": memory_item['type'],
        "context": json.dumps(memory_item['context']),
        "user_id": memory_item['metadata'].get('user_id'),
        "pageId": memory_item['metadata'].get('pageId'),
        "hierarchical structures": memory_item['metadata'].get('hierarchical structures'),
        "type": memory_item['metadata'].get('type'),
        "title": memory_item['metadata'].get('title'),
        "topics": memory_item['metadata'].get('topics'),
        "conversationId": memory_item['metadata'].get('conversationId'),
        "prompt": memory_item['metadata'].get('prompt'),
        "imageURL": memory_item['metadata'].get('imageURL'),
        "sourceType": memory_item['metadata'].get('sourceType'),
        "sourceUrl": memory_item['metadata'].get('sourceUrl'),
        "workspace_id": memory_item['metadata'].get('workspace_id'),
        "user_read_access": memory_item['metadata'].get('user_read_access'),
        "user_write_access": memory_item['metadata'].get('user_write_access'),
        "workspace_read_access": memory_item['metadata'].get('workspace_read_access'),
        "workspace_write_access": memory_item['metadata'].get('workspace_write_access'),
        "role_read_access": memory_item['metadata'].get('role_read_access'),
        "role_write_access": memory_item['metadata'].get('role_write_access')
    }

    memory_graph.neo_conn.session().__enter__.return_value.run.assert_called_with(
        expected_query,
        id=memory_item['id'],
        properties=expected_properties
    )

def test_add_memory_item_to_neo4j_duplicate(memory_graph, mocker):
    """
    Test adding a memory item that already exists.
    """
    # Sample memory item
    memory_item = {
        'id': 'bf80fb32-2d8c-414a-94e1-a3edffe46573',
        'type': 'messageList',
        'context': {'some': 'context'},
        'metadata': {
            # ... same as above
        }
    }

    # Mock the check query to return an existing node
    mock_session = memory_graph.neo_conn.session().__enter__.return_value
    mock_session.run.return_value.single.return_value = {'n': {'id': memory_item['id']}}

    # Call the method
    result = memory_graph.add_memory_item_to_neo4j(memory_item)

    # Assertions
    assert result is None

    # Verify that session.run was called for the check query
    check_query = """
                MATCH (n:Memory {id: $id})
                RETURN n
            """
    memory_graph.neo_conn.session().__enter__.return_value.run.assert_any_call(
        check_query,
        id=memory_item['id']
    )

    # Ensure that the MERGE query was not called since the item exists
    merge_query = """
                MERGE (n:Memory {id: $id})
                SET n += $properties
                RETURN n
            """
    memory_graph.neo_conn.session().__enter__.return_value.run.assert_called_with(
        check_query,
        id=memory_item['id']
    )

def test_add_memory_item_to_qdrant_acl_fields(memory_graph, mocker):
    """
    Test that add_memory_item_to_qdrant includes ACL fields in the Qdrant payload.
    """
    # Prepare a memory_item with ACL fields in metadata
    memory_item = {
        'id': 'test-memory-id',
        'content': 'Test content for Qdrant',
        'metadata': {
            'external_user_id': 'external_user_123',
            'external_user_read_access': ['external_user_123', 'external_user_456'],
            'external_user_write_access': ['external_user_123'],
            'user_id': 'mhnkVbAdgG',
            'user_read_access': ['mhnkVbAdgG'],
            'user_write_access': ['mhnkVbAdgG'],
        }
    }
    related_memories = []
    # Patch embedding_model.get_qwen_embedding_4b to return dummy embeddings/chunks
    dummy_embeddings = [[0.1, 0.2, 0.3]]
    dummy_chunks = ['Test content for Qdrant']
    memory_graph.embedding_model = mocker.Mock()
    memory_graph.embedding_model.get_qwen_embedding_4b = mocker.AsyncMock(return_value=(dummy_embeddings, dummy_chunks))

    # Patch qdrant_client.upsert to capture the payload
    captured_points = {}
    async def fake_upsert(collection_name, points, wait):
        captured_points['points'] = points
        return None
    memory_graph.qdrant_client = mocker.Mock()
    memory_graph.qdrant_client.upsert = fake_upsert

    # Call the method
    import asyncio
    asyncio.run(memory_graph.add_memory_item_to_qdrant(memory_item, [memory_item]))

    # Assert the payload includes ACL fields
    assert 'points' in captured_points
    assert len(captured_points['points']) == 1
    payload = captured_points['points'][0].payload
    assert payload['external_user_id'] == 'external_user_123'
    assert payload['external_user_read_access'] == ['external_user_123', 'external_user_456']
    assert payload['external_user_write_access'] == ['external_user_123']
    assert payload['user_id'] == 'mhnkVbAdgG'
    assert payload['user_read_access'] == ['mhnkVbAdgG']
    assert payload['user_write_access'] == ['mhnkVbAdgG']


def test_make_qdrant_id():
    """
    Test the make_qdrant_id function with various input formats.
    """
    memory_graph = MemoryGraph()
    
    # Test that the function returns valid UUIDs
    import uuid
    
    # Test UUID-based chunk IDs
    result1 = memory_graph.make_qdrant_id('02a6d6b6-5a03-4acf-b12a-a09f3a42c07d_0')
    assert uuid.UUID(result1)  # Should be a valid UUID
    
    result2 = memory_graph.make_qdrant_id('02a6d6b6-5a03-4acf-b12a-a09f3a42c07d_1')
    assert uuid.UUID(result2)  # Should be a valid UUID
    
    result3 = memory_graph.make_qdrant_id('02a6d6b6-5a03-4acf-b12a-a09f3a42c07d_10')
    assert uuid.UUID(result3)  # Should be a valid UUID
    
    # Test simple chunk IDs
    result4 = memory_graph.make_qdrant_id('baseid_01')
    assert uuid.UUID(result4)  # Should be a valid UUID
    
    result5 = memory_graph.make_qdrant_id('baseid_1')
    assert uuid.UUID(result5)  # Should be a valid UUID
    
    result6 = memory_graph.make_qdrant_id('baseid_10')
    assert uuid.UUID(result6)  # Should be a valid UUID
    
    # Test IDs with multiple underscores
    result7 = memory_graph.make_qdrant_id('simple_id_with_underscores')
    assert uuid.UUID(result7)  # Should be a valid UUID
    
    # Test IDs without underscores
    result8 = memory_graph.make_qdrant_id('simpleid')
    assert uuid.UUID(result8)  # Should be a valid UUID
    
    result9 = memory_graph.make_qdrant_id('02a6d6b6-5a03-4acf-b12a-a09f3a42c07d')
    assert uuid.UUID(result9)  # Should be a valid UUID
    
    # Test edge cases
    result10 = memory_graph.make_qdrant_id('')
    assert uuid.UUID(result10)  # Should be a valid UUID
    
    result11 = memory_graph.make_qdrant_id('_')
    assert uuid.UUID(result11)  # Should be a valid UUID
    
    result12 = memory_graph.make_qdrant_id('__')
    assert uuid.UUID(result12)  # Should be a valid UUID
    
    # Test that the same input always produces the same output (deterministic)
    assert memory_graph.make_qdrant_id('test_id') == memory_graph.make_qdrant_id('test_id')
    assert memory_graph.make_qdrant_id('different_id') != memory_graph.make_qdrant_id('test_id')