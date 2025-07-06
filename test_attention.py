"""
test_attention.py

Test script to demonstrate attention analysis functionality in the CustomerSupportAgent.
This script shows how to analyze attention patterns in agent responses using DistilBERT.
"""

from agent import CustomerSupportAgent

def test_attention_analysis():
    """Test the attention analysis functionality."""
    
    print("🧠 Testing Attention Analysis in CustomerSupportAgent")
    print("=" * 60)
    
    # Initialize agent
    agent = CustomerSupportAgent()
    
    # Test query
    test_query = "My robot dog broke and won't respond to commands"
    
    print(f"📝 Test Query: {test_query}")
    print("-" * 40)
    
    # Run agent with attention analysis
    print("🤖 Running agent with attention analysis...")
    response = agent.run(test_query, analyze_attention=True)
    
    print(f"\n📄 Agent Response:")
    print(response)
    print("-" * 40)
    
    # Manual attention analysis on the response
    print("\n🔍 Manual Attention Analysis:")
    attention_result = agent.analyze_response_attention(response, save_visualization=True)
    
    if 'error' not in attention_result:
        print("✅ Attention analysis successful!")
        
        # Show top attended tokens
        print(f"\n📊 Top 10 Most Attended Tokens:")
        for i, (token, weight) in enumerate(attention_result['top_attended_tokens'][:10], 1):
            print(f"{i:2d}. '{token}': {weight:.4f}")
        
        # Show analysis summary
        summary = attention_result['analysis_summary']
        print(f"\n📈 Analysis Summary:")
        print(f"   Total tokens: {summary['total_tokens']}")
        print(f"   Max attention: {summary['max_attention']:.4f}")
        print(f"   Average attention: {summary['avg_attention']:.4f}")
        print(f"   Attention std dev: {summary['attention_std']:.4f}")
        
        print(f"\n🎨 Attention heatmap saved as 'attention_heatmap.png'")
        
    else:
        print(f"❌ Attention analysis failed: {attention_result['error']}")
        print("💡 Make sure to install required dependencies:")
        print("   pip install torch transformers matplotlib seaborn")

def test_different_layers():
    """Test attention analysis on different transformer layers."""
    
    print("\n🔬 Testing Different Transformer Layers")
    print("=" * 60)
    
    agent = CustomerSupportAgent()
    sample_text = "The customer is experiencing technical difficulties with their smart device"
    
    print(f"📝 Sample Text: {sample_text}")
    
    # Test different layers
    for layer in [0, 3, 6]:
        print(f"\n🔍 Layer {layer} Analysis:")
        attention_data = agent.analyze_attention(sample_text, layer=layer, head=0)
        
        if 'error' not in attention_data:
            tokens = attention_data['tokens']
            weights = attention_data['attention_weights']
            
            # Find most attended token pair
            max_attention = 0
            max_pair = ("", "")
            
            for i, token_i in enumerate(tokens):
                for j, token_j in enumerate(tokens):
                    if weights[i][j] > max_attention:
                        max_attention = weights[i][j]
                        max_pair = (token_i, token_j)
            
            print(f"   Max attention: '{max_pair[0]}' → '{max_pair[1]}' ({max_attention:.4f})")
            print(f"   Total tokens: {len(tokens)}")
        else:
            print(f"   Error: {attention_data['error']}")

if __name__ == "__main__":
    # Run attention analysis test
    test_attention_analysis()
    
    # Test different layers
    test_different_layers()
    
    print(f"\n✅ Attention analysis testing completed!")
    print("📁 Check 'attention_heatmap.png' for visualization") 