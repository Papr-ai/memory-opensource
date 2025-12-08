import json
import sys
from collections import Counter
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator

# Global counter for debug output
DEBUG_COUNTER = 0

class SourceLocation(BaseModel):
    in_pinecone: bool
    in_neo: bool
    in_bigbird: bool

class RetrievalSource(BaseModel):
    memory_id: str
    source_location: SourceLocation

class Response(BaseModel):
    query_id: int
    memory_id: str
    query: str
    response: str
    score_gpt: Optional[float] = None

class NeoNode(BaseModel):
    content: str
    user_id: str
    memory_id: str

class Entry(BaseModel):
    _debug_counter = 0  # Class variable to track instances
    
    query_text: str
    response_from_papr: List[Response] = Field(..., alias='retreived responses from papr')
    memory_ids_from_pinecone: List[str] = Field(..., alias='memory IDs from pinecone')
    memory_ids_from_bigbird: List[str] = Field(..., alias='memory IDs from bigbird')
    neo_node_content: List[NeoNode] = Field(..., alias='neo node content')
    retreival_sources: Dict[str, Any] = Field(default_factory=dict)  # Made this optional with default
    hit_position_with_ranking: Optional[int] = Field(None, alias='Hit position without Ranking')
    hit_position_without_ranking: Optional[int] = Field(None, alias='Hit position without Ranking')
    memory_id_with_ranking: Optional[str] = Field(None, alias='Memory ID with ranking')
    memory_id_without_ranking: Optional[str] = Field(None, alias='Memory ID without ranking')
    query_id: Optional[int] = Field(None, alias='query_ID')
    iteration: Optional[int] = None
    match_found_pinecone_score_0_7: Optional[bool] = Field(None, alias='match_found_pinecone_score_0_7')
    number_chunk_embeddings: Optional[int] = Field(None, alias='Number of chunk embeddings')
    number_queried_chunks_pinecone: Optional[int] = Field(None, alias='Number of queried chunks from pinecone')
    number_ids_matched_parse: Optional[int] = Field(None, alias='Number of IDs matched in parse')
    number_ids_matched_neo: Optional[int] = Field(None, alias='Number of IDs matched in neo')
    match_found: bool = False

    @field_validator('hit_position_with_ranking', 'hit_position_without_ranking', mode='before')
    def validate_hit_position(cls, v):
        if v is None or v == -1:
            return None
        return v

    @field_validator('memory_id_with_ranking', mode='before')
    def validate_memory_id(cls, v):
        if not v:  # Handles None, empty string, etc
            return None
        return v

    @field_validator('memory_ids_from_pinecone', 'memory_ids_from_bigbird', mode='before')
    def clean_memory_ids(cls, v):
        if isinstance(v, list):
            # Remove suffixes and create a unique set
            cleaned_ids = {id.split('_')[0] for id in v}
            return list(cleaned_ids)
        return v

    @field_validator('retreival_sources', mode='before')
    def validate_retrieval_sources(cls, v):
        if isinstance(v, dict):
            return v
        return {}  # Return empty dict if no value provided

    def is_memory_added(self) -> bool:
        in_pinecone = (
            self.match_found_pinecone_score_0_7
            and self.number_chunk_embeddings is not None
            and self.number_queried_chunks_pinecone is not None
            and (self.number_chunk_embeddings == self.number_queried_chunks_pinecone)
        )
        in_parse = (self.number_ids_matched_parse and self.number_ids_matched_parse > 0)
        in_neo = (self.number_ids_matched_neo and self.number_ids_matched_neo > 0)
        in_bigbird = (len(self.memory_ids_from_bigbird) > 0)
        return in_pinecone and in_parse and in_bigbird and in_neo

def analyze_retrieval_sources(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Initialize ALL counters
    total_queries = 0
    source_counts = Counter()
    count_hit1_no_rank = 0
    count_hit5_no_rank = 0
    count_hit1_rank = 0
    count_hit5_rank = 0
    mrr_sum = 0
    recall_sum = 0
    non_added_iterations = []

    # If the last item is a summary entry, skip it
    for idx, entry_data in enumerate(data[:-1], start=1):
        try:
            total_queries += 1

            # Check if memory was added to all retrieval sources
            in_pinecone = (
                entry_data.get('match_found_pinecone_score_0_7')
                and entry_data.get('Number of chunk embeddings') is not None
                and entry_data.get('Number of queried chunks from pinecone') is not None
                and (entry_data.get('Number of chunk embeddings') == entry_data.get('Number of queried chunks from pinecone'))
            )
            in_parse = entry_data.get('Number of IDs matched in parse', 0) > 0
            in_neo = entry_data.get('Number of IDs matched in neo', 0) > 0
            in_bigbird = len(entry_data.get('memory IDs from bigbird', [])) > 0
            
            # Check Golden answer sources
            golden_sources = entry_data.get('Golden answer in sources_found', {})
            golden_all_true = all(golden_sources.values()) if golden_sources else False
            
            # Memory is added if either all sources have content or golden answer shows all sources found
            memory_added = (in_pinecone and in_parse and in_bigbird and in_neo) or golden_all_true
            match_found = entry_data.get('match_found', False) or golden_all_true

            if not memory_added and not match_found:
                iteration = entry_data.get('iteration')
                query_id = entry_data.get('query_ID')
                iteration_str = iteration if iteration is not None else query_id
                if iteration_str is None:
                    iteration_str = f"idx_{idx}"
                print(f"Excluding iteration {iteration_str} from strict set (no memory added + no match).")
                non_added_iterations.append(iteration_str)
                continue

            # 1) Count Hit@1 / Hit@5 for NO ranking
            pos_no_rank = entry_data.get('Hit position without Ranking')
            if isinstance(pos_no_rank, int):
                if pos_no_rank == 0:
                    count_hit1_no_rank += 1
                if 0 <= pos_no_rank <= 4:
                    count_hit5_no_rank += 1

            # 2) Count Hit@1 / Hit@5 WITH ranking
            pos_rank = entry_data.get('Hit position with Ranking')
            if isinstance(pos_rank, int):
                if pos_rank == 0:
                    count_hit1_rank += 1
                if 0 <= pos_rank <= 4:
                    count_hit5_rank += 1

            # MRR and recall@20 can be averaged over all queries
            # If no information, they may be 0 or missing
            mrr = entry_data.get('MRR', 0.0)
            recall = entry_data.get('recall@20', 0.0)
            mrr_sum += mrr
            recall_sum += recall

            # Finally, track which source was used for the winning memory, if any
            winning_id_rank = entry_data.get("Memory ID  with Ranking")
            if not winning_id_rank:
                source_counts['No Match'] += 1
                continue

            base_id_rank = winning_id_rank.split('_')[0]
            retrieval_sources = entry_data.get('retreival sources', {})
            source_locations = retrieval_sources.get('memory_id_source_location', [])

            found_sources = []
            for loc in source_locations:
                loc_mem = loc['memory_id']
                sloc = loc['source_location']
                if loc_mem in (winning_id_rank, base_id_rank):
                    if sloc.get('in_pinecone'):
                        found_sources.append('Pinecone')
                    if sloc.get('in_bigbird'):
                        found_sources.append('Bigbird')
                    if sloc.get('in_neo'):
                        found_sources.append('Neo')

            if len(found_sources) > 1:
                source_counts['Multiple Sources'] += 1
            elif len(found_sources) == 1:
                source_counts[found_sources[0]] += 1
            else:
                source_counts['Unknown'] += 1

            if not memory_added and not match_found:
                print(f"Excluding iteration {iteration_str} from strict set (no memory added + no match).")
                non_added_iterations.append(iteration_str)
                continue

        except Exception as e:
            print(f"\nError processing entry: {str(e)}")
            continue

    # -------------------------
    # Print out final stats
    # -------------------------
    print("\nIterations where memory was NOT added or 'match_found' was false:")
    print(f"Total count: {len(non_added_iterations)}")
    print(f"Iteration (or Query) IDs: {non_added_iterations}")

    # Source distribution
    total_src = sum(source_counts.values())
    print("\nOverall Distribution:")
    print("| Retrieval Source              | Number of Answers | Percentage (%) |")
    print("|------------------------------|------------------|----------------|")
    for source, count in source_counts.most_common():
        percentage = (count / total_src * 100) if total_src > 0 else 0
        print(f"| {source:<28} | {count:>16} | {percentage:>12.2f}% |")
    print(f"| {'Total':<28} | {total_src:>16} | {100:>12.2f}% |")

    # Compute hits out of ALL queries (total_queries)
    print("\nRetrieval Metrics (all queries):")
    print("| Metric                        | Success Rate     | Percentage (%) |")
    print("|------------------------------|------------------|----------------|")

    if total_queries > 0:
        # Hits@1 no rank
        print(f"| {'Hit@1 (no ranking)':<28} | "
              f"{count_hit1_no_rank:>7}/{total_queries:<8} | "
              f"{(count_hit1_no_rank / total_queries * 100):>12.2f}% |")

        # Hits@5 no rank
        print(f"| {'Hit@5 (no ranking)':<28} | "
              f"{count_hit5_no_rank:>7}/{total_queries:<8} | "
              f"{(count_hit5_no_rank / total_queries * 100):>12.2f}% |")

        # Hits@1 rank
        print(f"| {'Hit@1 (with ranking)':<28} | "
              f"{count_hit1_rank:>7}/{total_queries:<8} | "
              f"{(count_hit1_rank / total_queries * 100):>12.2f}% |")

        # Hits@5 rank
        print(f"| {'Hit@5 (with ranking)':<28} | "
              f"{count_hit5_rank:>7}/{total_queries:<8} | "
              f"{(count_hit5_rank / total_queries * 100):>12.2f}% |")

        # MRR average over all queries
        print(f"| {'MRR':<28} | {'-':>16} | {(mrr_sum / total_queries * 100):>12.2f}% |")

        # Recall@20 average over all queries
        print(f"| {'Recall@20':<28} | {'-':>16} | {(recall_sum / total_queries * 100):>12.2f}% |")
    else:
        print("No queries found")

    print(f"\nTotal Number of Queries: {total_queries} (excluded {len(non_added_iterations)})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_retrieval_sources.py <json_file>")
        sys.exit(1)
    json_file = sys.argv[1]
    analyze_retrieval_sources(json_file) 