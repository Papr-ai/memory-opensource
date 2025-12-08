import numpy as np
from models.embedding_model import EmbeddingModel
from services.logging_config import get_logger

# Create a logger instance for this module
logger = get_logger(__name__)  # Will use 'memory.memory_graph' as the logger name



class MemoryItemRelevance:
    def __init__(self, memory_item, query, context = {}):
        self.memory_item = memory_item
        self.query = query
        self.context = context
        # Replace 'SentenceTransformer' initialization with 'EmbeddingModel'
        self.embedding_model = EmbeddingModel()

        self.relevance = self.calculate_relevance()

    def calculate_relevance(self):
        base_relevance = self.get_base_relevance(self.memory_item, self.query)
        
        # You might want to normalize the context relevance, depending on its range and how much weight you want to give it
        context_relevance = self.get_context_relevance(self.memory_item, self.context)
        
        # This is a simple linear combination, but you could use a more complex formula if desired
        return 0.7 * base_relevance + 0.3 * context_relevance

    def get_base_relevance(self, memory_item, query):
        # Replace this with your actual relevance calculation code
        # For example, you might want to use cosine similarity between the memory item's embedding and the query's embedding
        memory_embedding = np.array(memory_item.embedding)
        query_embedding = self.get_query_embedding(self.query)
        return np.dot(memory_embedding, query_embedding) / (np.linalg.norm(memory_embedding) * np.linalg.norm(query_embedding))

    def get_context_relevance(self, memory_item, context):
        # This is a simple example that checks if the context's topic matches the memory item's topics
        # In a real application, you might use more complex logic
        if context['topic'] and context['topic'] in memory_item.metadata['topics']:
            return 1
        else:
            return 0

    def get_query_embedding(self, query):
        # Use the EmbeddingModel's get_sentence_embedding method
        #return self.embedding_model.get_sentence_embedding(query)
        return self.embedding_model.get_qwen_embedding_4b(query)