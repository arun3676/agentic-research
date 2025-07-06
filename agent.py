"""
AI Customer Support Agent with Chain-of-Thought Reasoning
Research-ready implementation for analysis and evaluation
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
import re

# Attention analysis imports
try:
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from transformers import DistilBertTokenizer, DistilBertModel
    ATTENTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Attention analysis dependencies not available: {e}")
    print("Install with: pip install torch transformers matplotlib seaborn")
    ATTENTION_AVAILABLE = False

# --- Ensure OPENAI_API_KEY is set from Streamlit secrets if available ---
try:
    import streamlit as st
    if hasattr(st, 'secrets') and st.secrets:
        api_key = st.secrets.get("openai", {}).get("api_key")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
except Exception:
    pass

class CustomerSupportAgent:
    """
    AI Customer Support Agent with Chain-of-Thought reasoning for research analysis.
    
    Features:
    - Chain-of-Thought prompting for transparent reasoning
    - Task routing and classification
    - Structured response format for analysis
    - Comprehensive logging for research evaluation
    """
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7):
        """
        Initialize the customer support agent.
        
        Args:
            model_name: OpenAI model to use (default: gpt-4)
            temperature: Model temperature for response creativity (default: 0.7)
        """
        # Get API key using the helper
        api_key = self._get_api_key()
        # Use ChatOpenAI for chat models, explicitly passing model_name and api_key
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key
        )
        
        # Initialize chains
        self._setup_chains()
        
        # Research tracking
        self.query_count = 0
        self.responses = []
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from multiple sources with proper error handling."""
        # Try environment variable first
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            return api_key
        
        # Try Streamlit secrets
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and st.secrets:
                api_key = st.secrets.get("openai", {}).get("api_key")
                if api_key:
                    return api_key
        except Exception:
            pass
        
        # Try direct secrets access
        try:
            import streamlit as st
            api_key = st.secrets["openai"]["api_key"]
            if api_key:
                return api_key
        except Exception:
            pass
        
        return None
    
    def _setup_chains(self):
        """Setup LangChain components for the agent."""
        
        # Main reasoning prompt with Chain-of-Thought
        reasoning_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
You are an expert customer support agent with advanced reasoning capabilities. 
Your task is to help customers with their inquiries using clear, step-by-step reasoning.

Customer Query: {query}

Please provide a comprehensive response following this exact format:

Reasoning: [Your step-by-step analysis of the customer's issue, including:
- What type of problem this appears to be
- What information you need to consider
- How you would approach solving this
- Any potential complications or considerations]

Answer: [Your direct, helpful response to the customer's query]

Task Category: [Classify this as one of: technical_support, billing_inquiry, product_information, complaint_handling, general_inquiry]

Action Required: [What specific action should be taken: provide_information, escalate_issue, create_ticket, schedule_callback, no_action_needed]

Confidence Level: [High/Medium/Low based on how certain you are about your response]

Remember to always start with "Reasoning:" and provide clear, logical steps in your analysis.
"""
        )
        
        self.reasoning_chain = LLMChain(
            llm=self.llm,
            prompt=reasoning_prompt,
            verbose=False
        )
    
    def classify_query(self, query: str) -> str:
        """
        Classify the type of customer query for routing analysis.
        
        Args:
            query: Customer's query text
            
        Returns:
            Classification category
        """
        classification_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
Classify this customer query into one of these categories:
- technical_support: Technical issues, bugs, system problems
- billing_inquiry: Payment, charges, subscription questions
- product_information: Product details, features, specifications
- complaint_handling: Complaints, dissatisfaction, refund requests
- general_inquiry: General questions, account help, other

Query: {query}

Category:"""
        )
        
        classification_chain = LLMChain(llm=self.llm, prompt=classification_prompt)
        result = classification_chain.run(query=query)
        return result.strip().lower()
    
    def extract_components(self, response: str) -> Dict:
        """
        Extract structured components from the agent's response.
        
        Args:
            response: Full response from the agent
            
        Returns:
            Dictionary with extracted components
        """
        components = {
            'reasoning': '',
            'answer': '',
            'task_category': '',
            'action_required': '',
            'confidence_level': ''
        }
        
        # Extract reasoning
        reasoning_match = re.search(r'Reasoning:(.*?)(?=Answer:|$)', response, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            components['reasoning'] = reasoning_match.group(1).strip()
        
        # Extract answer
        answer_match = re.search(r'Answer:(.*?)(?=Task Category:|$)', response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            components['answer'] = answer_match.group(1).strip()
        
        # Extract task category
        category_match = re.search(r'Task Category:(.*?)(?=Action Required:|$)', response, re.DOTALL | re.IGNORECASE)
        if category_match:
            components['task_category'] = category_match.group(1).strip()
        
        # Extract action required
        action_match = re.search(r'Action Required:(.*?)(?=Confidence Level:|$)', response, re.DOTALL | re.IGNORECASE)
        if action_match:
            components['action_required'] = action_match.group(1).strip()
        
        # Extract confidence level
        confidence_match = re.search(r'Confidence Level:(.*?)$', response, re.DOTALL | re.IGNORECASE)
        if confidence_match:
            components['confidence_level'] = confidence_match.group(1).strip()
        
        return components
    
    def analyze_attention(self, text: str, layer: int = 6, head: int = 0) -> Dict:
        """
        Analyze attention weights using DistilBERT for a given text.
        
        Args:
            text: Text to analyze
            layer: Which transformer layer to analyze (default: 6)
            head: Which attention head to analyze (default: 0)
            
        Returns:
            Dictionary containing attention weights, tokens, and analysis
        """
        if not ATTENTION_AVAILABLE:
            return {"error": "Attention analysis dependencies not available"}
        
        try:
            # Load DistilBERT model and tokenizer
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            model = DistilBertModel.from_pretrained('distilbert-base-uncased')
            model.eval()
            
            # Tokenize the text
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Get attention weights
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
                attention_weights = outputs.attentions[layer][0, head].numpy()
            
            # Remove special tokens for cleaner analysis
            if len(tokens) > 2:
                tokens = tokens[1:-1]  # Remove [CLS] and [SEP]
                attention_weights = attention_weights[1:-1, 1:-1]
            
            return {
                'tokens': tokens,
                'attention_weights': attention_weights.tolist(),
                'layer': layer,
                'head': head,
                'text_length': len(tokens)
            }
            
        except Exception as e:
            return {"error": f"Attention analysis failed: {str(e)}"}
    
    def visualize_attention(self, attention_data: Dict, filename: str = "attention_heatmap.png") -> bool:
        """
        Create and save an attention heatmap visualization.
        
        Args:
            attention_data: Output from analyze_attention method
            filename: Output filename for the heatmap
            
        Returns:
            True if successful, False otherwise
        """
        if not ATTENTION_AVAILABLE:
            print("Warning: Visualization dependencies not available")
            return False
        
        if 'error' in attention_data:
            print(f"Error in attention data: {attention_data['error']}")
            return False
        
        try:
            tokens = attention_data['tokens']
            attention_weights = np.array(attention_data['attention_weights'])
            
            # Create the heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                attention_weights,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap='Blues',
                annot=False,
                cbar_kws={'label': 'Attention Weight'}
            )
            
            plt.title(f'DistilBERT Attention Weights (Layer {attention_data["layer"]}, Head {attention_data["head"]})')
            plt.xlabel('Key Tokens')
            plt.ylabel('Query Tokens')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Attention heatmap saved as {filename}")
            return True
            
        except Exception as e:
            print(f"Error creating attention visualization: {str(e)}")
            return False
    
    def analyze_response_attention(self, response: str, save_visualization: bool = True) -> Dict:
        """
        Analyze attention patterns in an agent response.
        
        Args:
            response: Agent response text to analyze
            save_visualization: Whether to save attention heatmap
            
        Returns:
            Dictionary with attention analysis results
        """
        if not ATTENTION_AVAILABLE:
            return {"error": "Attention analysis not available"}
        
        try:
            # Analyze attention for the response
            attention_data = self.analyze_attention(response)
            
            if 'error' in attention_data:
                return attention_data
            
            # Find most attended tokens
            attention_weights = np.array(attention_data['attention_weights'])
            tokens = attention_data['tokens']
            
            # Calculate average attention for each token
            avg_attention = np.mean(attention_weights, axis=1)
            token_attention_pairs = list(zip(tokens, avg_attention))
            token_attention_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Get top attended tokens
            top_tokens = token_attention_pairs[:10]
            
            # Save visualization if requested
            if save_visualization:
                self.visualize_attention(attention_data)
            
            return {
                'attention_data': attention_data,
                'top_attended_tokens': top_tokens,
                'analysis_summary': {
                    'total_tokens': len(tokens),
                    'max_attention': float(np.max(attention_weights)),
                    'avg_attention': float(np.mean(attention_weights)),
                    'attention_std': float(np.std(attention_weights))
                }
            }
            
        except Exception as e:
            return {"error": f"Response attention analysis failed: {str(e)}"}
    
    def run(self, query: str, analyze_attention: bool = False) -> str:
        """
        Process a customer query and return a reasoned response.
        
        Args:
            query: Customer's query text
            analyze_attention: Whether to analyze attention patterns in the response
            
        Returns:
            Structured response with reasoning
        """
        try:
            # Generate response with reasoning
            response = self.reasoning_chain.run(query=query)
            
            # Ensure response has reasoning section
            if not response.strip().startswith('Reasoning:'):
                response = f"Reasoning: Analyzing customer query step by step...\n\nAnswer: {response}"
            
            # Analyze attention patterns if requested
            if analyze_attention and ATTENTION_AVAILABLE:
                print("🔍 Analyzing attention patterns in response...")
                attention_analysis = self.analyze_response_attention(response)
                if 'error' not in attention_analysis:
                    print("✅ Attention analysis completed")
                    print(f"📊 Top attended tokens: {[token for token, _ in attention_analysis['top_attended_tokens'][:5]]}")
                else:
                    print(f"⚠️ Attention analysis failed: {attention_analysis['error']}")
            
            # Track for research
            self.query_count += 1
            self._log_response(query, response)
            
            return response
            
        except Exception as e:
            error_response = f"ERROR: Failed to process query - {str(e)}"
            self._log_response(query, error_response)
            return error_response
    
    def _log_response(self, query: str, response: str):
        """Log response for research analysis."""
        log_entry = {
            'query_id': f"Q{self.query_count:04d}",
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'components': self.extract_components(response)
        }
        self.responses.append(log_entry)
    
    def save_responses(self, filename: str = "test_results.json"):
        """Save all responses to JSON file for analysis."""
        with open(filename, 'w') as f:
            json.dump(self.responses, f, indent=2)
    
    def get_analytics(self) -> Dict:
        """Get analytics data for research evaluation."""
        if not self.responses:
            return {}
        
        total_queries = len(self.responses)
        successful_queries = len([r for r in self.responses if not r['response'].startswith('ERROR')])
        
        # Category distribution
        categories = {}
        for r in self.responses:
            category = r['components'].get('task_category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
        
        # Response length analysis
        response_lengths = [len(r['response']) for r in self.responses if not r['response'].startswith('ERROR')]
        
        return {
            'total_queries': total_queries,
            'successful_queries': successful_queries,
            'success_rate': (successful_queries / total_queries) * 100 if total_queries > 0 else 0,
            'category_distribution': categories,
            'avg_response_length': sum(response_lengths) / len(response_lengths) if response_lengths else 0,
            'min_response_length': min(response_lengths) if response_lengths else 0,
            'max_response_length': max(response_lengths) if response_lengths else 0
        }

# Example usage for testing
if __name__ == "__main__":
    agent = CustomerSupportAgent()
    
    # Test queries
    test_queries = [
        "I can't log into my account",
        "How much does the premium plan cost?",
        "The app keeps crashing when I try to upload files",
        "I want to cancel my subscription",
        "What features are included in the basic plan?"
    ]
    
    print("🤖 AI Customer Support Agent - Research Test")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 30)
        # Run with attention analysis for the first query
        analyze_attn = (i == 1)  # Only analyze attention for first query
        response = agent.run(query, analyze_attention=analyze_attn)
        print(response)
        print("=" * 50)
    
    # Save results
    agent.save_responses()
    
    # Show analytics
    analytics = agent.get_analytics()
    print(f"\n📊 Analytics Summary:")
    print(f"Total Queries: {analytics['total_queries']}")
    print(f"Success Rate: {analytics['success_rate']:.1f}%")
    print(f"Average Response Length: {analytics['avg_response_length']:.0f} characters")
    print(f"Category Distribution: {analytics['category_distribution']}")
    
    # Demonstrate attention analysis on a sample response
    if ATTENTION_AVAILABLE and analytics['total_queries'] > 0:
        print(f"\n🔍 Attention Analysis Demo:")
        sample_response = "Reasoning: This appears to be a technical login issue. I need to consider authentication problems, account status, and potential solutions. The user is likely experiencing access problems that require troubleshooting steps."
        
        print("Analyzing attention patterns in sample response...")
        attention_result = agent.analyze_response_attention(sample_response)
        
        if 'error' not in attention_result:
            print("✅ Attention analysis successful!")
            print(f"📊 Top 5 attended tokens:")
            for token, weight in attention_result['top_attended_tokens'][:5]:
                print(f"   '{token}': {weight:.4f}")
            print(f"📈 Analysis summary: {attention_result['analysis_summary']}")
        else:
            print(f"❌ Attention analysis failed: {attention_result['error']}")
    else:
        print(f"\n⚠️ Attention analysis not available. Install dependencies with:")
        print("pip install torch transformers matplotlib seaborn") 