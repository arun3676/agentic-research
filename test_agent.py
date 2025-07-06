"""
test_agent.py

Script to test the customer support agent on multiple queries and save results for analysis.
This creates a dataset of agent responses with reasoning for research purposes.
"""

import json
import os
import random
from datetime import datetime
from agent import CustomerSupportAgent

# Comprehensive test queries across different categories
SAMPLE_QUERIES = [
    # Technical Support (15 queries)
    "My robot dog broke and won't respond to commands",
    "The smart home system keeps disconnecting from WiFi",
    "My drone crashed and the camera is damaged",
    "The AI assistant keeps giving wrong answers",
    "My virtual reality headset shows a black screen",
    "The smart refrigerator won't connect to the app",
    "My robot vacuum is stuck under the couch",
    "The smart thermostat won't adjust temperature",
    "My fitness tracker stopped syncing with my phone",
    "The smart speaker won't recognize my voice commands",
    "My gaming console keeps freezing during gameplay",
    "The smart doorbell camera shows no video feed",
    "My wireless earbuds won't pair with my device",
    "The smart coffee maker won't brew properly",
    "My tablet screen is completely unresponsive",
    
    # Billing Inquiries (12 queries)
    "I was charged twice for the same subscription",
    "Why did my monthly fee increase without notice?",
    "I want to cancel my premium membership",
    "There's an unauthorized charge on my account",
    "How do I update my payment method?",
    "I need a refund for a defective product",
    "My invoice shows incorrect tax calculations",
    "Can I get a discount for being a long-term customer?",
    "I was charged for a service I never used",
    "How do I dispute a billing error?",
    "I need to change my billing cycle",
    "Why am I being charged for shipping when it was free?",
    
    # Complaint Handling (13 queries)
    "I'm extremely frustrated with your customer service",
    "The delivery person was rude and unprofessional",
    "My order arrived three weeks late",
    "The product quality is much worse than advertised",
    "I've been on hold for over an hour",
    "Your website keeps crashing when I try to order",
    "The customer service representative was unhelpful",
    "My complaint from last week still hasn't been addressed",
    "The product broke after only one week of use",
    "I'm disappointed with the overall experience",
    "The packaging was damaged and items were missing",
    "Your return process is too complicated",
    "I feel like my concerns are being ignored",
    
    # General Inquiries (10 queries)
    "What are your business hours?",
    "Do you ship internationally?",
    "What's your return policy for electronics?",
    "How do I create an account on your website?",
    "What payment methods do you accept?",
    "Do you offer student discounts?",
    "What's the warranty period for your products?",
    "How do I track my order status?",
    "Do you have a loyalty program?",
    "What's your privacy policy regarding customer data?"
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