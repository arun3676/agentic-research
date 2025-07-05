"""
test_agent.py

Script to test the customer support agent on multiple queries and save results for analysis.
This creates a dataset of agent responses with reasoning for research purposes.
"""

import json
import os
from datetime import datetime
from agent import CustomerSupportAgent

# Sample queries for testing
SAMPLE_QUERIES = [
    "Why is my order delayed?",
    "I'm angry about the service quality",
    "What's the return policy?",
    "My order status shows pending",
    "I want to file a complaint",
    "How do I track my shipment?",
    "The product arrived damaged",
    "Can I cancel my order?",
    "What are your business hours?",
    "I need help with my account"
]

def run_batch_test():
    """
    Run the agent on all sample queries and save results.
    Returns a list of dictionaries with query, response, and metadata.
    """
    print("Initializing agent...")
    agent = CustomerSupportAgent()
    
    results = []
    
    print(f"Testing agent on {len(SAMPLE_QUERIES)} queries...")
    
    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"Processing query {i}/{len(SAMPLE_QUERIES)}: {query}")
        
        try:
            # Get agent response
            response = agent.run(query)
            
            # Store result
            result = {
                "query": query,
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "query_id": i
            }
            results.append(result)
            
            print(f"✓ Query {i} completed")
            
        except Exception as e:
            print(f"✗ Error on query {i}: {str(e)}")
            result = {
                "query": query,
                "response": f"ERROR: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "query_id": i
            }
            results.append(result)
    
    return results

def save_results(results, filename="test_results.json"):
    """
    Save test results to a JSON file for later analysis.
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

def print_summary(results):
    """
    Print a summary of the test results.
    """
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    successful = len([r for r in results if not r["response"].startswith("ERROR")])
    failed = len(results) - successful
    
    print(f"Total queries: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    print("\nSample responses:")
    for i, result in enumerate(results[:3]):  # Show first 3
        print(f"\nQuery {i+1}: {result['query']}")
        print(f"Response: {result['response'][:200]}...")

if __name__ == "__main__":
    print("Starting batch test of customer support agent...")
    
    # Run the tests
    results = run_batch_test()
    
    # Save results
    save_results(results)
    
    # Print summary
    print_summary(results)
    
    print("\nTest completed! Check test_results.json for full results.") 