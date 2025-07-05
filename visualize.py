"""
visualize.py

Script to analyze and visualize agent responses for research paper presentation.
Creates plots showing reasoning patterns, response lengths, and task routing.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import re

def load_results(filename="test_results.json"):
    """
    Load test results from JSON file.
    """
    with open(filename, 'r') as f:
        return json.load(f)

def analyze_response_lengths(results):
    """
    Analyze response lengths and create a histogram.
    """
    lengths = [len(result['response']) for result in results if not result['response'].startswith('ERROR')]
    
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Response Length (characters)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Agent Response Lengths')
    plt.grid(True, alpha=0.3)
    plt.savefig('response_lengths.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'mean_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths)
    }

def analyze_reasoning_patterns(results):
    """
    Analyze reasoning patterns in responses.
    """
    reasoning_keywords = ['because', 'since', 'therefore', 'thus', 'reason', 'explain', 'step']
    reasoning_counts = []
    
    for result in results:
        if not result['response'].startswith('ERROR'):
            response_lower = result['response'].lower()
            count = sum(1 for keyword in reasoning_keywords if keyword in response_lower)
            reasoning_counts.append(count)
    
    plt.figure(figsize=(10, 6))
    plt.hist(reasoning_counts, bins=range(max(reasoning_counts) + 2), alpha=0.7, color='lightgreen', edgecolor='black')
    plt.xlabel('Number of Reasoning Keywords')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reasoning Keywords in Responses')
    plt.grid(True, alpha=0.3)
    plt.savefig('reasoning_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'avg_reasoning_keywords': np.mean(reasoning_counts),
        'responses_with_reasoning': sum(1 for count in reasoning_counts if count > 0)
    }

def analyze_task_routing(results):
    """
    Analyze which queries triggered task routing.
    """
    # Simple analysis based on query content
    routing_triggers = {
        'order_status': ['order', 'delayed', 'status', 'pending'],
        'escalation': ['angry', 'complaint', 'damaged'],
        'general': ['policy', 'hours', 'help', 'track', 'cancel']
    }
    
    routing_counts = {'order_status': 0, 'escalation': 0, 'general': 0}
    
    for result in results:
        query_lower = result['query'].lower()
        for category, keywords in routing_triggers.items():
            if any(keyword in query_lower for keyword in keywords):
                routing_counts[category] += 1
                break
    
    # Create pie chart
    plt.figure(figsize=(8, 8))
    categories = list(routing_counts.keys())
    counts = list(routing_counts.values())
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    
    plt.pie(counts, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Task Routing Distribution')
    plt.savefig('task_routing.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return routing_counts

def create_response_quality_analysis(results):
    """
    Create a simple quality analysis based on response characteristics.
    """
    quality_metrics = []
    
    for result in results:
        if not result['response'].startswith('ERROR'):
            response = result['response']
            
            # Simple quality indicators
            has_reasoning = any(word in response.lower() for word in ['because', 'reason', 'explain'])
            has_action = any(word in response.lower() for word in ['check', 'escalate', 'action'])
            response_length = len(response)
            
            # Simple quality score (0-100)
            quality_score = 0
            if has_reasoning: quality_score += 30
            if has_action: quality_score += 20
            if response_length > 100: quality_score += 25
            if response_length > 200: quality_score += 25
            
            quality_metrics.append(quality_score)
    
    plt.figure(figsize=(10, 6))
    plt.hist(quality_metrics, bins=10, alpha=0.7, color='gold', edgecolor='black')
    plt.xlabel('Quality Score (0-100)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Response Quality Scores')
    plt.grid(True, alpha=0.3)
    plt.savefig('quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'avg_quality_score': np.mean(quality_metrics),
        'high_quality_responses': sum(1 for score in quality_metrics if score >= 70)
    }

def generate_research_summary(results):
    """
    Generate a comprehensive summary for research paper.
    """
    print("\n" + "="*60)
    print("RESEARCH PAPER ANALYSIS SUMMARY")
    print("="*60)
    
    # Basic stats
    total_queries = len(results)
    successful_queries = len([r for r in results if not r['response'].startswith('ERROR')])
    success_rate = (successful_queries / total_queries) * 100
    
    print(f"Total Queries Tested: {total_queries}")
    print(f"Successful Responses: {successful_queries}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    # Length analysis
    length_stats = analyze_response_lengths(results)
    print(f"\nResponse Length Analysis:")
    print(f"  Average Length: {length_stats['mean_length']:.0f} characters")
    print(f"  Median Length: {length_stats['median_length']:.0f} characters")
    
    # Reasoning analysis
    reasoning_stats = analyze_reasoning_patterns(results)
    print(f"\nReasoning Analysis:")
    print(f"  Average Reasoning Keywords: {reasoning_stats['avg_reasoning_keywords']:.1f}")
    print(f"  Responses with Reasoning: {reasoning_stats['responses_with_reasoning']}/{successful_queries}")
    
    # Task routing analysis
    routing_stats = analyze_task_routing(results)
    print(f"\nTask Routing Analysis:")
    for category, count in routing_stats.items():
        print(f"  {category.replace('_', ' ').title()}: {count} queries")
    
    # Quality analysis
    quality_stats = create_response_quality_analysis(results)
    print(f"\nQuality Analysis:")
    print(f"  Average Quality Score: {quality_stats['avg_quality_score']:.1f}/100")
    print(f"  High Quality Responses: {quality_stats['high_quality_responses']}/{successful_queries}")
    
    print(f"\nVisualization files created:")
    print(f"  - response_lengths.png")
    print(f"  - reasoning_patterns.png")
    print(f"  - task_routing.png")
    print(f"  - quality_analysis.png")

if __name__ == "__main__":
    print("Loading test results...")
    try:
        results = load_results()
        generate_research_summary(results)
    except FileNotFoundError:
        print("Error: test_results.json not found. Run test_agent.py first!")
    except Exception as e:
        print(f"Error during analysis: {str(e)}") 