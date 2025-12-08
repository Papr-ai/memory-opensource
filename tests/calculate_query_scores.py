#!/usr/bin/env python3
"""
Calculate accuracy scores for all 10 support engineering queries.

This script reads the expected answers and calculates scores based on:
- The JSON report file from the test run
- Or allows manual input of results

Usage:
    poetry run python tests/calculate_query_scores.py
    poetry run python tests/calculate_query_scores.py --report tests/test_reports/support_engineering_queries_20251029_124919.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from difflib import SequenceMatcher
import re

# Import the expected answers and scoring function from the test file
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    Calculate accuracy score (0-10) by comparing returned content with expected answer.
    
    Scoring criteria:
    - Must-include keywords: 0-4 points
    - Keyword coverage: 0-3 points  
    - Content similarity: 0-3 points
    """
    if not returned_content or len(returned_content.strip()) == 0:
        return 0.0, {
            "error": "Empty content",
            "must_include_found": [],
            "must_include_missing": expected_answer.get("must_include", []),
            "keywords_found": [],
            "keywords_missing": expected_answer.get("keywords", []),
            "similarity_score": 0.0,
            "breakdown": {
                "must_include_points": 0.0,
                "keywords_points": 0.0,
                "similarity_points": 0.0
            }
        }
    
    returned_lower = returned_content.lower()
    details = {
        "must_include_found": [],
        "must_include_missing": [],
        "keywords_found": [],
        "keywords_missing": [],
        "similarity_score": 0.0
    }
    
    score = 0.0
    
    # 1. Must-include keywords (0-4 points)
    must_include_score = 0.0
    for keyword in expected_answer.get("must_include", []):
        keyword_lower = keyword.lower()
        found = False
        
        if keyword_lower in returned_lower:
            found = True
        elif keyword_lower == "low pressure":
            if "low pdp" in returned_lower or ("pdp" in returned_lower and "low" in returned_lower):
                found = True
        
        if found:
            details["must_include_found"].append(keyword)
            must_include_score += 1.0
        else:
            details["must_include_missing"].append(keyword)
    
    if len(expected_answer.get("must_include", [])) > 0:
        must_include_score = min(4.0, (must_include_score / len(expected_answer["must_include"])) * 4.0)
    score += must_include_score
    
    # 2. All keywords coverage (0-3 points)
    keywords_score = 0.0
    all_keywords = expected_answer.get("keywords", [])
    for keyword in all_keywords:
        keyword_lower = keyword.lower()
        found = False
        
        if keyword_lower in returned_lower:
            found = True
        elif keyword_lower == "low pressure":
            if "low pdp" in returned_lower or ("pdp" in returned_lower and "low" in returned_lower):
                found = True
        elif keyword_lower == "hot gas bypass valve":
            if "hot gas" in returned_lower and ("bypass" in returned_lower or "by pass" in returned_lower):
                found = True
        elif keyword_lower == "lower than limits":
            if "lower" in returned_lower and "limits" in returned_lower:
                found = True
        
        if found:
            details["keywords_found"].append(keyword)
            keywords_score += 1.0
        else:
            details["keywords_missing"].append(keyword)
    
    if len(all_keywords) > 0:
        keywords_score = min(3.0, (keywords_score / len(all_keywords)) * 3.0)
    score += keywords_score
    
    # 3. Content similarity (0-3 points)
    expected_text = expected_answer.get("expected_content", "").lower()
    similarity = SequenceMatcher(None, returned_lower[:500], expected_text[:500]).ratio()
    details["similarity_score"] = similarity
    similarity_points = similarity * 3.0
    score += similarity_points
    
    score = round(min(10.0, max(0.0, score)), 1)
    
    details["breakdown"] = {
        "must_include_points": round(must_include_score, 1),
        "keywords_points": round(keywords_score, 1),
        "similarity_points": round(similarity_points, 1)
    }
    
    return score, details


def extract_content_from_results(query_result: Dict[str, Any]) -> str:
    """Extract concatenated content from query results.
    
    Priority:
    1. Use all_memories (if available) - has full content for all results
    2. Use top_10_results (if available)
    3. Try to extract from raw_api_response (if parsing failed)
    """
    contents = []
    
    # First, try all_memories (new format with full content)
    all_memories = query_result.get("all_memories", [])
    if all_memories:
        for memory in all_memories[:20]:  # Top 20 for scoring (improved coverage)
            content = memory.get("content_full", "")
            if content:
                # Clean markdown images and noise
                content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
                content = re.sub(r'\*+', '', content)
                contents.append(content.strip())
        if contents:
            return "\n\n".join(contents)
    
    # Fallback to top_10_results
    for result in query_result.get("top_10_results", []):
        content = result.get("content_full", "") or result.get("content_preview", "")
        if content:
            # Clean markdown images and noise
            content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
            content = re.sub(r'\*+', '', content)
            contents.append(content.strip())
    
    # If still no content, try to extract from raw_api_response
    if not contents:
        raw_response = query_result.get("raw_api_response", {})
        if raw_response:
            # Try to extract memories from raw response
            memories = []
            data_obj = raw_response.get("data", {})
            if isinstance(data_obj, dict):
                memories = data_obj.get("memories", [])
                if not memories:
                    memories = data_obj.get("data", [])
            elif isinstance(data_obj, list):
                memories = data_obj
            
            if not memories:
                memories = raw_response.get("memories", [])
            if not memories:
                memories = raw_response.get("results", [])
            
            # Extract content from raw memories
            for memory in memories[:20]:  # Top 20 for scoring (improved coverage)
                if isinstance(memory, dict):
                    content = memory.get("content", "")
                    if content:
                        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
                        content = re.sub(r'\*+', '', content)
                        contents.append(content.strip())
    
    return "\n\n".join(contents)


def main():
    """Calculate scores for all 10 queries."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate accuracy scores for support engineering queries")
    parser.add_argument(
        "--report",
        type=str,
        default="tests/test_reports/support_engineering_queries_20251029_124919.json",
        help="Path to the JSON report file from test run"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: print to stdout and save as scores_<timestamp>.json)"
    )
    
    args = parser.parse_args()
    
    # Load report file
    report_path = Path(args.report)
    if not report_path.exists():
        print(f"❌ Report file not found: {report_path}")
        print("\nAvailable report files:")
        reports_dir = Path("tests/test_reports")
        if reports_dir.exists():
            for f in sorted(reports_dir.glob("support_engineering_queries_*.json")):
                print(f"  - {f}")
        sys.exit(1)
    
    with open(report_path, "r") as f:
        report = json.load(f)
    
    print("="*80)
    print("ACCURACY SCORE CALCULATION FOR SUPPORT ENGINEERING QUERIES")
    print("="*80)
    print(f"\nReport file: {report_path}")
    print(f"Schema ID: {report.get('schema_id', 'N/A')}")
    print(f"Timestamp: {report.get('timestamp', 'N/A')}")
    print(f"Total queries: {len(report.get('queries', []))}")
    print(f"⚠️  Scoring method: Top 20 results (improved coverage)\n")
    
    all_scores = {
        "report_file": str(report_path),
        "schema_id": report.get("schema_id"),
        "timestamp": report.get("timestamp"),
        "queries": []
    }
    
    scores_summary = []
    
    # Calculate scores for each query
    for query_data in report.get("queries", []):
        query = query_data.get("query", "")
        query_num = query_data.get("query_number", 0)
        
        if query not in EXPECTED_ANSWERS:
            print(f"⚠️  Query {query_num}: No expected answer defined for: '{query}'")
            continue
        
        expected_answer = EXPECTED_ANSWERS[query]
        
        # Extract content from results - check all possible sources
        returned_content = extract_content_from_results(query_data)
        
        # Check if we actually have content (even if total_results says 0)
        has_content = bool(returned_content.strip())
        has_error = bool(query_data.get("error"))
        total_results = query_data.get("total_results", 0)
        
        if has_error and not has_content and total_results == 0:
            # Truly no results
            print(f"\n{'='*80}")
            print(f"Query {query_num}: {query}")
            print(f"{'='*80}")
            print("❌ No results returned - Score: 0.0/10.0")
            if query_data.get("raw_api_response"):
                print("   ℹ️  Raw API response saved for debugging")
            score = 0.0
            details = {
                "error": query_data.get("error", "No results"),
                "must_include_found": [],
                "must_include_missing": expected_answer.get("must_include", []),
                "keywords_found": [],
                "keywords_missing": expected_answer.get("keywords", []),
                "similarity_score": 0.0,
                "breakdown": {
                    "must_include_points": 0.0,
                    "keywords_points": 0.0,
                    "similarity_points": 0.0
                }
            }
        else:
            # We have content (possibly extracted from raw_api_response)
            if total_results == 0 and has_content:
                print(f"\n{'='*80}")
                print(f"Query {query_num}: {query}")
                print(f"{'='*80}")
                print("⚠️  Results extracted from raw API response (parsing may have failed)")
            else:
                print(f"\n{'='*80}")
                print(f"Query {query_num}: {query}")
                print(f"{'='*80}")
            
            score, details = calculate_accuracy_score(returned_content, expected_answer, query)
            
            print(f"📊 Results returned: {query_data.get('total_results', 0)}")
            print(f"📝 Content length: {len(returned_content)} characters")
            print(f"\n✅ Accuracy Score: {score}/10.0")
            print(f"   • Must-include keywords: {len(details['must_include_found'])}/{len(expected_answer.get('must_include', []))} found")
            print(f"     Points: {details['breakdown']['must_include_points']}/4.0")
            print(f"   • Keyword coverage: {len(details['keywords_found'])}/{len(expected_answer.get('keywords', []))} found")
            print(f"     Points: {details['breakdown']['keywords_points']}/3.0")
            print(f"   • Content similarity: {details['similarity_score']:.3f}")
            print(f"     Points: {details['breakdown']['similarity_points']:.1f}/3.0")
            
            if details.get("must_include_missing"):
                print(f"\n   ❌ Missing must-include: {', '.join(details['must_include_missing'])}")
            if details.get("keywords_missing"):
                print(f"   ⚠️  Missing keywords: {', '.join(details['keywords_missing'][:5])}{' ...' if len(details['keywords_missing']) > 5 else ''}")
        
        scores_summary.append({
            "query_number": query_num,
            "query": query,
            "score": score,
            "total_results": query_data.get("total_results", 0),
            "has_results": query_data.get("total_results", 0) > 0
        })
        
        all_scores["queries"].append({
            "query_number": query_num,
            "query": query,
            "score": score,
            "details": details,
            "expected_answer": expected_answer,
            "returned_content_length": len(returned_content),
            "total_results": query_data.get("total_results", 0)
        })
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    queries_with_results = [s for s in scores_summary if s["has_results"]]
    queries_without_results = [s for s in scores_summary if not s["has_results"]]
    
    # Initialize variables
    avg_score = 0.0
    median_score = 0.0
    max_score = 0.0
    min_score = 0.0
    
    if queries_with_results:
        scores = [s["score"] for s in queries_with_results]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        
        # Calculate median
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        if n % 2 == 0:
            median_score = (sorted_scores[n//2 - 1] + sorted_scores[n//2]) / 2
        else:
            median_score = sorted_scores[n//2]
        
        print(f"\nQueries with results: {len(queries_with_results)}/{len(scores_summary)}")
        print(f"  Average score: {avg_score:.2f}/10.0")
        print(f"  Median score: {median_score:.2f}/10.0")
        print(f"  Best score: {max_score:.2f}/10.0 (Query {max([s for s in queries_with_results if s['score'] == max_score], key=lambda x: x['query_number'])['query_number']})")
        print(f"  Worst score: {min_score:.2f}/10.0 (Query {min([s for s in queries_with_results if s['score'] == min_score], key=lambda x: x['query_number'])['query_number']})")
    else:
        print("\n❌ No queries returned results - all scores are 0.0")
    
    if queries_without_results:
        print(f"\nQueries without results: {len(queries_without_results)}")
        for s in queries_without_results:
            print(f"  - Query {s['query_number']}: {s['query'][:60]}...")
    
    # Save to file
    output_path = args.output or Path("tests/test_reports") / f"accuracy_scores_{report.get('timestamp', 'latest')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_scores["summary"] = {
        "total_queries": len(scores_summary),
        "queries_with_results": len(queries_with_results),
        "queries_without_results": len(queries_without_results),
        "average_score": round(avg_score, 2) if queries_with_results else 0.0,
        "median_score": round(median_score, 2) if queries_with_results else 0.0,
        "max_score": round(max_score, 2) if queries_with_results else 0.0,
        "min_score": round(min_score, 2) if queries_with_results else 0.0,
        "scoring_method": "top_20_results"  # Indicate we're using top 20
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed scores saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()

