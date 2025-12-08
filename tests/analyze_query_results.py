#!/usr/bin/env python3
"""
Deep analysis of query results to understand why scores are low and how to improve.

This script analyzes:
1. Which memories were used for scoring (top 5 vs top 20)
2. Neo4j node types found
3. Content relevance
4. Whether vector embeddings would help
5. Missing keywords and why
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter

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


def analyze_query(query_data: Dict[str, Any], query_num: int, results_query_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Analyze a single query's results."""
    query = query_data.get("query", "")
    expected = EXPECTED_ANSWERS.get(query, {})
    
    analysis = {
        "query_number": query_num,
        "query": query,
        "score": query_data.get("score", 0.0),
        "total_results": query_data.get("total_results", 0),
        "insights": {}
    }
    
    # Get all memories - prefer from results_data, fallback to query_data
    all_memories = []
    if results_query_data:
        all_memories = results_query_data.get("all_memories", results_query_data.get("top_10_results", []))
        # If we only have top_10, use that and mark it
        if not results_query_data.get("all_memories"):
            all_memories = results_query_data.get("top_10_results", [])
    else:
        all_memories = query_data.get("all_memories", query_data.get("top_10_results", []))
    
    top_10_results = query_data.get("top_10_results", [])
    
    if not all_memories:
        analysis["insights"]["error"] = "No memories available for analysis"
        return analysis
    
    # Analyze top 5 vs top 20
    top_5 = all_memories[:5]
    top_20 = all_memories[:20]
    
    # Check if keywords appear in top 5 vs top 20
    def extract_text(memories):
        return " ".join([m.get("content_full", "") for m in memories]).lower()
    
    top_5_text = extract_text(top_5)
    top_20_text = extract_text(top_20)
    
    keywords_found_top5 = []
    keywords_found_top20 = []
    keywords_missing_top5 = []
    keywords_missing_top20 = []
    
    if expected:
        for keyword in expected.get("keywords", []):
            keyword_lower = keyword.lower()
            if keyword_lower in top_5_text or ("low pressure" == keyword_lower and ("low pdp" in top_5_text or ("pdp" in top_5_text and "low" in top_5_text))):
                keywords_found_top5.append(keyword)
            else:
                keywords_missing_top5.append(keyword)
            
            if keyword_lower in top_20_text or ("low pressure" == keyword_lower and ("low pdp" in top_20_text or ("pdp" in top_20_text and "low" in top_20_text))):
                keywords_found_top20.append(keyword)
            else:
                keywords_missing_top20.append(keyword)
    
    analysis["insights"]["keyword_coverage"] = {
        "top_5": {
            "found": len(keywords_found_top5),
            "total": len(expected.get("keywords", [])),
            "percentage": round(len(keywords_found_top5) / len(expected.get("keywords", [])) * 100, 1) if expected.get("keywords") else 0,
            "found_keywords": keywords_found_top5,
            "missing_keywords": keywords_missing_top5
        },
        "top_20": {
            "found": len(keywords_found_top20),
            "total": len(expected.get("keywords", [])),
            "percentage": round(len(keywords_found_top20) / len(expected.get("keywords", [])) * 100, 1) if expected.get("keywords") else 0,
            "found_keywords": keywords_found_top20,
            "missing_keywords": keywords_missing_top20
        },
        "improvement_if_top20": len(keywords_found_top20) - len(keywords_found_top5)
    }
    
    # Analyze Neo4j nodes
    all_node_labels = []
    node_label_counts = Counter()
    memories_with_nodes = 0
    memories_without_nodes = 0
    
    for memory in top_20:
        node_labels = memory.get("node_labels", [])
        if node_labels:
            all_node_labels.extend(node_labels)
            node_label_counts.update(node_labels)
            memories_with_nodes += 1
        else:
            memories_without_nodes += 1
    
    analysis["insights"]["neo4j_nodes"] = {
        "unique_node_types": list(set(all_node_labels)),
        "node_type_counts": dict(node_label_counts),
        "memories_with_nodes": memories_with_nodes,
        "memories_without_nodes": memories_without_nodes,
        "total_node_relationships": len(all_node_labels)
    }
    
    # Check if relevant nodes match expected content
    relevant_node_keywords = []
    if expected:
        # Check if any node labels contain expected keywords
        for keyword in expected.get("keywords", []):
            keyword_lower = keyword.lower()
            for node_label in all_node_labels:
                if keyword_lower in node_label.lower():
                    relevant_node_keywords.append((keyword, node_label))
    
    analysis["insights"]["relevant_nodes"] = {
        "matching_keywords": relevant_node_keywords,
        "has_relevant_nodes": len(relevant_node_keywords) > 0
    }
    
    # Analyze memory ranks where keywords appear
    keyword_rank_distribution = {}
    for keyword in expected.get("keywords", []):
        keyword_lower = keyword.lower()
        ranks = []
        for idx, memory in enumerate(top_20, 1):
            content = memory.get("content_full", "").lower()
            if keyword_lower in content or ("low pressure" == keyword_lower and ("low pdp" in content or ("pdp" in content and "low" in content))):
                ranks.append(idx)
        if ranks:
            keyword_rank_distribution[keyword] = {
                "appears_at_ranks": ranks,
                "lowest_rank": min(ranks),
                "in_top_5": any(r <= 5 for r in ranks),
                "in_top_10": any(r <= 10 for r in ranks)
            }
    
    analysis["insights"]["keyword_location"] = keyword_rank_distribution
    
    # Calculate potential score improvement with top 20
    must_include_found_top20 = 0
    for keyword in expected.get("must_include", []):
        keyword_lower = keyword.lower()
        if keyword_lower in top_20_text or ("low pressure" == keyword_lower and ("low pdp" in top_20_text or ("pdp" in top_20_text and "low" in top_20_text))):
            must_include_found_top20 += 1
    
    current_must_include_points = query_data.get("details", {}).get("breakdown", {}).get("must_include_points", 0.0)
    potential_must_include_points = min(4.0, (must_include_found_top20 / len(expected.get("must_include", []))) * 4.0) if expected.get("must_include") else 0.0
    
    current_keywords_points = query_data.get("details", {}).get("breakdown", {}).get("keywords_points", 0.0)
    potential_keywords_points = min(3.0, (len(keywords_found_top20) / len(expected.get("keywords", []))) * 3.0) if expected.get("keywords") else 0.0
    
    analysis["insights"]["score_improvement"] = {
        "current_score": query_data.get("score", 0.0),
        "potential_with_top20": round(
            potential_must_include_points + 
            potential_keywords_points + 
            query_data.get("details", {}).get("breakdown", {}).get("similarity_points", 0.0),
            1
        ),
        "improvement": round(
            (potential_must_include_points + potential_keywords_points) - 
            (current_must_include_points + current_keywords_points),
            1
        )
    }
    
    # Recommendations
    recommendations = []
    
    if analysis["insights"]["score_improvement"]["improvement"] > 1.0:
        recommendations.append(f"💡 Using top 20 instead of top 5 could improve score by {analysis['insights']['score_improvement']['improvement']:.1f} points")
    
    if memories_without_nodes > memories_with_nodes:
        recommendations.append(f"⚠️  {memories_without_nodes} memories have no Neo4j nodes - graph relationships may be incomplete")
    
    if not analysis["insights"]["relevant_nodes"]["has_relevant_nodes"]:
        recommendations.append("⚠️  No Neo4j node labels match expected keywords - schema may not match domain")
    
    if len(keywords_missing_top20) > 0:
        recommendations.append(f"🔍 {len(keywords_missing_top20)} keywords not found even in top 20 results - may need better query understanding or more data")
    
    if analysis["insights"]["keyword_coverage"]["top_20"]["percentage"] < 50:
        recommendations.append("💡 Consider using vector embedding similarity for better semantic matching")
        recommendations.append("💡 Expected answer may be spread across multiple memories - consider combining or reranking")
    
    analysis["insights"]["recommendations"] = recommendations
    
    return analysis


def main():
    """Run deep analysis on accuracy scores."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deep analysis of query accuracy scores")
    parser.add_argument(
        "--scores",
        type=str,
        default="tests/test_reports/accuracy_scores_20251029_141901.json",
        help="Path to accuracy scores JSON file"
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Path to original query results JSON file (auto-detected if not provided)"
    )
    
    args = parser.parse_args()
    
    scores_path = Path(args.scores)
    if not scores_path.exists():
        print(f"❌ Scores file not found: {scores_path}")
        sys.exit(1)
    
    with open(scores_path, "r") as f:
        scores_data = json.load(f)
    
    # Get original results file
    results_path = args.results or scores_data.get("report_file")
    if not results_path:
        # Try to infer from scores filename
        timestamp = scores_data.get("timestamp", "")
        results_path = f"tests/test_reports/support_engineering_queries_{timestamp}.json"
    
    results_path = Path(results_path)
    if not results_path.exists():
        print(f"⚠️  Original results file not found: {results_path}")
        print(f"   Will analyze using only scores data (limited insights)")
        results_data = None
    else:
        with open(results_path, "r") as f:
            results_data = json.load(f)
    
    print("="*80)
    print("DEEP QUERY ANALYSIS - Understanding Score Performance")
    print("="*80)
    print(f"\nAnalyzing: {scores_path}")
    print(f"Schema ID: {scores_data.get('schema_id', 'N/A')}")
    print(f"Average Score: {scores_data.get('summary', {}).get('average_score', 0.0)}/10.0\n")
    
    all_analyses = []
    
    # Create lookup for results data
    results_lookup = {}
    if results_data:
        for rq in results_data.get("queries", []):
            query_num = rq.get("query_number", 0)
            results_lookup[query_num] = rq
    
    # Analyze each query
    for query_data in scores_data.get("queries", []):
        query_num = query_data.get("query_number", 0)
        results_query_data = results_lookup.get(query_num)
        analysis = analyze_query(query_data, query_num, results_query_data)
        all_analyses.append(analysis)
        
        # Print detailed analysis
        print("="*80)
        print(f"Query {query_num}: {analysis['query']}")
        print(f"Score: {analysis['score']}/10.0 | Total Results: {analysis['total_results']}")
        print("="*80)
        
        # Check if we have insights
        if "error" in analysis["insights"]:
            print(f"\n⚠️  {analysis['insights']['error']}\n")
            continue
        
        # Keyword coverage
        kw_insights = analysis["insights"].get("keyword_coverage", {})
        if kw_insights:
            top5 = kw_insights.get("top_5", {})
            top20 = kw_insights.get("top_20", {})
            print(f"\n📊 KEYWORD COVERAGE:")
            print(f"   Top 5:  {top5.get('found', 0)}/{top5.get('total', 0)} keywords ({top5.get('percentage', 0)}%)")
            print(f"   Top 20: {top20.get('found', 0)}/{top20.get('total', 0)} keywords ({top20.get('percentage', 0)}%)")
            if kw_insights.get("improvement_if_top20", 0) > 0:
                print(f"   💡 Using top 20 would find {kw_insights['improvement_if_top20']} additional keywords")
        
        # Neo4j nodes
        node_insights = analysis["insights"].get("neo4j_nodes", {})
        if node_insights:
            print(f"\n🔗 NEO4J NODES:")
            mem_with = node_insights.get('memories_with_nodes', 0)
            mem_without = node_insights.get('memories_without_nodes', 0)
            print(f"   Memories with nodes: {mem_with}/{mem_with + mem_without}")
            print(f"   Unique node types: {len(node_insights.get('unique_node_types', []))}")
            unique_types = node_insights.get('unique_node_types', [])
            if unique_types:
                print(f"   Node types: {', '.join(unique_types[:10])}")
            else:
                print(f"   ⚠️  No Neo4j nodes found!")
        
        # Relevant nodes
        rel_insights = analysis["insights"].get("relevant_nodes", {})
        if rel_insights.get("has_relevant_nodes"):
            print(f"\n✅ RELEVANT NODES FOUND:")
            for keyword, node_label in rel_insights.get("matching_keywords", []):
                print(f"   • '{keyword}' matches node: {node_label}")
        else:
            print(f"\n❌ NO RELEVANT NODES: Node labels don't match expected keywords")
        
        # Keyword locations
        kw_loc = analysis["insights"].get("keyword_location", {})
        if kw_loc:
            print(f"\n📍 KEYWORD LOCATIONS (in top 20):")
            for keyword, info in list(kw_loc.items())[:5]:  # Show top 5 keywords
                status = "✅" if info.get("in_top_5") else ("✓" if info.get("in_top_10") else "📍")
                print(f"   {status} '{keyword}': appears at ranks {info.get('appears_at_ranks', [])}, lowest: {info.get('lowest_rank', 'N/A')}")
        
        # Score improvement potential
        score_improve = analysis["insights"].get("score_improvement", {})
        if score_improve.get("improvement", 0) > 0:
            print(f"\n📈 SCORE IMPROVEMENT POTENTIAL:")
            print(f"   Current: {score_improve.get('current_score', 0)}/10.0")
            print(f"   With top 20: {score_improve.get('potential_with_top20', 0)}/10.0")
            print(f"   Improvement: +{score_improve.get('improvement', 0):.1f} points")
        
        # Recommendations
        recommendations = analysis["insights"].get("recommendations", [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   {rec}")
        
        print()
    
    # Overall summary
    print("="*80)
    print("OVERALL INSIGHTS")
    print("="*80)
    
    total_queries = len(all_analyses)
    queries_with_node_issues = sum(1 for a in all_analyses 
                                   if a["insights"].get("neo4j_nodes", {}).get("memories_without_nodes", 0) > 
                                      a["insights"].get("neo4j_nodes", {}).get("memories_with_nodes", 0))
    queries_with_top20_benefit = sum(1 for a in all_analyses 
                                     if a["insights"].get("score_improvement", {}).get("improvement", 0) > 1.0)
    queries_with_no_relevant_nodes = sum(1 for a in all_analyses 
                                         if not a["insights"].get("relevant_nodes", {}).get("has_relevant_nodes", False))
    
    print(f"\n📊 Summary Statistics:")
    print(f"   • Queries with Neo4j node issues: {queries_with_node_issues}/{total_queries}")
    print(f"   • Queries that would benefit from top 20: {queries_with_top20_benefit}/{total_queries}")
    print(f"   • Queries with no relevant node matches: {queries_with_no_relevant_nodes}/{total_queries}")
    
    print(f"\n💡 System-wide Recommendations:")
    print(f"   1. Consider using top 20 memories for accuracy calculation (vs top 5)")
    print(f"   2. Use vector embedding similarity for semantic matching")
    print(f"   3. Verify Neo4j graph relationships are being created correctly")
    print(f"   4. Check if custom schema node types match expected domain entities")
    
    # Save detailed analysis
    output_path = Path("tests/test_reports") / f"deep_analysis_{scores_path.stem.replace('accuracy_scores_', '')}.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": scores_data.get("summary", {}),
            "queries": all_analyses
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed analysis saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()

