"""
review_app.py

Enhanced Streamlit web app to review agent test results for research paper presentation.
- Loads test_results.json and visualizations
- Provides comprehensive analytics and insights
- Allows data export and comparison
- Shows detailed reasoning analysis
"""

import streamlit as st
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import importlib.util

# Load results
def load_results(filename="test_results.json"):
    if not os.path.exists(filename):
        st.error(f"File not found: {filename}. Run test_agent.py first.")
        st.stop()
    with open(filename, 'r') as f:
        return json.load(f)

def analyze_reasoning_patterns(results):
    """Analyze reasoning patterns in responses"""
    reasoning_keywords = ['because', 'since', 'therefore', 'thus', 'reason', 'explain', 'step']
    reasoning_data = []
    
    for result in results:
        if not result['response'].startswith('ERROR'):
            response_lower = result['response'].lower()
            count = sum(1 for keyword in reasoning_keywords if keyword in response_lower)
            reasoning_data.append({
                'query': result['query'],
                'reasoning_count': count,
                'response_length': len(result['response']),
                'has_reasoning': count > 0
            })
    
    return pd.DataFrame(reasoning_data)

def analyze_task_routing(results):
    """Analyze task routing patterns"""
    routing_triggers = {
        'order_status': ['order', 'delayed', 'status', 'pending'],
        'escalation': ['angry', 'complaint', 'damaged'],
        'general': ['policy', 'hours', 'help', 'track', 'cancel']
    }
    
    routing_data = []
    for result in results:
        query_lower = result['query'].lower()
        category = 'general'
        for cat, keywords in routing_triggers.items():
            if any(keyword in query_lower for keyword in keywords):
                category = cat
                break
        
        routing_data.append({
            'query': result['query'],
            'category': category,
            'response_length': len(result['response']),
            'success': not result['response'].startswith('ERROR')
        })
    
    return pd.DataFrame(routing_data)

def calculate_quality_scores(results):
    """Calculate quality scores for responses"""
    quality_data = []
    
    for result in results:
        if not result['response'].startswith('ERROR'):
            response = result['response']
            
            # Quality indicators
            has_reasoning = any(word in response.lower() for word in ['because', 'reason', 'explain'])
            has_action = any(word in response.lower() for word in ['check', 'escalate', 'action'])
            response_length = len(response)
            
            # Quality score (0-100)
            quality_score = 0
            if has_reasoning: quality_score += 30
            if has_action: quality_score += 20
            if response_length > 100: quality_score += 25
            if response_length > 200: quality_score += 25
            
            quality_data.append({
                'query': result['query'],
                'quality_score': quality_score,
                'has_reasoning': has_reasoning,
                'has_action': has_action,
                'response_length': response_length
            })
    
    return pd.DataFrame(quality_data)

# Helper to split answer and reasoning
def split_answer_reasoning(response: str):
    if 'Reasoning:' in response:
        parts = response.split('Reasoning:', 1)
        answer = parts[0].strip()
        reasoning = parts[1].strip()
    else:
        answer = response.strip()
        reasoning = '(No explicit reasoning found)'
    return answer, reasoning

# Helper to run live agent query
def run_live_agent(query: str):
    # Dynamically import agent.py
    spec = importlib.util.spec_from_file_location("agent", os.path.join(os.getcwd(), "agent.py"))
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)
    agent = agent_module.CustomerSupportAgent()
    return agent.run(query)

def main():
    st.set_page_config(page_title="AI Agent Research Review", layout="wide")
    
    # Header
    st.title("🤖 AI Agent Research: Customer Support Analysis")
    st.markdown("**Comprehensive review interface for research paper presentation**")
    
    # --- Live Query Box ---
    st.subheader("Ask the Agent a New Question")
    with st.form(key="live_query_form"):
        user_query = st.text_input("Type your question to the agent:")
        submit = st.form_submit_button("Ask Agent")
    if submit and user_query:
        with st.spinner("Agent is thinking..."):
            live_response = run_live_agent(user_query)
            answer, reasoning = split_answer_reasoning(live_response)
            st.success("**Agent Answer:**\n" + answer)
            st.info("**Reasoning:**\n" + reasoning)

    # Load data
    results = load_results()
    
    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📋 Results Overview", "🔍 Detailed Analysis", "📈 Visualizations", "📊 Research Metrics", "💾 Export Data"]
    )
    
    if page == "📋 Results Overview":
        show_results_overview(results)
    elif page == "🔍 Detailed Analysis":
        show_detailed_analysis(results)
    elif page == "📈 Visualizations":
        show_visualizations(results)
    elif page == "📊 Research Metrics":
        show_research_metrics(results)
    elif page == "💾 Export Data":
        show_export_options(results)

def show_results_overview(results):
    """Show overview of all results"""
    st.header("📋 Results Overview")
    
    # Search and filter
    col1, col2 = st.columns([2, 1])
    with col1:
        search_query = st.text_input("🔍 Search queries or responses:")
    with col2:
        show_only_errors = st.checkbox("Show only errors", value=False)
    
    # Filter results
    filtered = []
    for r in results:
        if show_only_errors and not r["response"].startswith("ERROR"):
            continue
        if search_query:
            if search_query.lower() in r["query"].lower() or search_query.lower() in r["response"].lower():
                filtered.append(r)
        else:
            filtered.append(r)
    
    st.write(f"**Showing {len(filtered)} of {len(results)} results**")
    
    # Display results
    for i, r in enumerate(filtered):
        answer, reasoning = split_answer_reasoning(r['response'])
        with st.expander(f"**Query {i+1}:** {r['query']}", expanded=False):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("**Answer:**")
                st.write(answer)
                st.markdown("**Reasoning:**")
                st.write(reasoning)
            with col2:
                st.markdown("**Metadata:**")
                st.write(f"**Timestamp:** {r['timestamp']}")
                st.write(f"**Response Length:** {len(r['response'])} chars")
                st.write(f"**Query ID:** {r['query_id']}")

def show_detailed_analysis(results):
    """Show detailed analysis of results"""
    st.header("🔍 Detailed Analysis")
    
    # Create analysis dataframes
    reasoning_df = analyze_reasoning_patterns(results)
    routing_df = analyze_task_routing(results)
    quality_df = calculate_quality_scores(results)
    
    # Tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["🧠 Reasoning Analysis", "🔄 Task Routing", "⭐ Quality Assessment"])
    
    with tab1:
        st.subheader("Reasoning Pattern Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Reasoning Keywords Distribution:**")
            fig = px.histogram(reasoning_df, x='reasoning_count', 
                             title='Distribution of Reasoning Keywords',
                             labels={'reasoning_count': 'Number of Reasoning Keywords', 'count': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Response Length vs Reasoning:**")
            fig = px.scatter(reasoning_df, x='response_length', y='reasoning_count',
                           title='Response Length vs Reasoning Keywords',
                           labels={'response_length': 'Response Length (chars)', 'reasoning_count': 'Reasoning Keywords'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.write("**Detailed Reasoning Analysis:**")
        st.dataframe(reasoning_df)
    
    with tab2:
        st.subheader("Task Routing Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            category_counts = routing_df['category'].value_counts()
            fig = px.pie(values=category_counts.values, names=category_counts.index,
                        title='Task Routing Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            avg_length_by_category = routing_df.groupby('category')['response_length'].mean()
            fig = px.bar(x=avg_length_by_category.index, y=avg_length_by_category.values,
                        title='Average Response Length by Category',
                        labels={'x': 'Category', 'y': 'Average Length (chars)'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.write("**Task Routing Details:**")
        st.dataframe(routing_df)
    
    with tab3:
        st.subheader("Quality Assessment")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(quality_df, x='quality_score',
                             title='Quality Score Distribution',
                             labels={'quality_score': 'Quality Score (0-100)', 'count': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            quality_breakdown = {
                'Has Reasoning': quality_df['has_reasoning'].sum(),
                'Has Action': quality_df['has_action'].sum(),
                'Long Response (>200 chars)': (quality_df['response_length'] > 200).sum()
            }
            fig = px.bar(x=list(quality_breakdown.keys()), y=list(quality_breakdown.values()),
                        title='Quality Indicators Breakdown')
            st.plotly_chart(fig, use_container_width=True)
        
        st.write("**Quality Assessment Details:**")
        st.dataframe(quality_df)

def show_visualizations(results):
    """Show generated visualizations"""
    st.header("📈 Generated Visualizations")
    
    # Check for generated plots
    plot_files = ["response_lengths.png", "reasoning_patterns.png", "task_routing.png", "quality_analysis.png"]
    available_plots = [f for f in plot_files if os.path.exists(f)]
    
    if not available_plots:
        st.warning("No visualization files found. Run `python visualize.py` first.")
        return
    
    # Display plots in columns
    cols = st.columns(2)
    for i, plot_file in enumerate(available_plots):
        with cols[i % 2]:
            st.image(plot_file, caption=plot_file.replace('.png', '').replace('_', ' ').title(), use_column_width=True)

def show_research_metrics(results):
    """Show comprehensive research metrics"""
    st.header("📊 Research Metrics Summary")
    
    # Calculate metrics
    total_queries = len(results)
    successful_queries = len([r for r in results if not r['response'].startswith('ERROR')])
    success_rate = (successful_queries / total_queries) * 100
    
    # Response length metrics
    response_lengths = [len(r['response']) for r in results if not r['response'].startswith('ERROR')]
    avg_length = np.mean(response_lengths)
    median_length = np.median(response_lengths)
    
    # Reasoning metrics
    reasoning_df = analyze_reasoning_patterns(results)
    avg_reasoning = reasoning_df['reasoning_count'].mean()
    responses_with_reasoning = reasoning_df['has_reasoning'].sum()
    
    # Quality metrics
    quality_df = calculate_quality_scores(results)
    avg_quality = quality_df['quality_score'].mean()
    high_quality_responses = (quality_df['quality_score'] >= 70).sum()
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Queries", total_queries)
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col2:
        st.metric("Avg Response Length", f"{avg_length:.0f} chars")
        st.metric("Median Length", f"{median_length:.0f} chars")
    
    with col3:
        st.metric("Avg Reasoning Keywords", f"{avg_reasoning:.1f}")
        st.metric("Responses with Reasoning", f"{responses_with_reasoning}/{successful_queries}")
    
    with col4:
        st.metric("Avg Quality Score", f"{avg_quality:.1f}/100")
        st.metric("High Quality Responses", f"{high_quality_responses}/{successful_queries}")
    
    # Task routing summary
    st.subheader("Task Routing Summary")
    routing_df = analyze_task_routing(results)
    routing_summary = routing_df['category'].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Routing Distribution:**")
        for category, count in routing_summary.items():
            st.write(f"- {category.replace('_', ' ').title()}: {count} queries")
    
    with col2:
        fig = px.pie(values=routing_summary.values, names=routing_summary.index,
                    title='Task Routing Distribution')
        st.plotly_chart(fig, use_container_width=True)

def show_export_options(results):
    """Show data export options"""
    st.header("💾 Export Data")
    
    # Create exportable dataframes
    reasoning_df = analyze_reasoning_patterns(results)
    routing_df = analyze_task_routing(results)
    quality_df = calculate_quality_scores(results)
    
    # Export options
    st.subheader("Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Raw Results:**")
        if st.button("Download Raw Results (JSON)"):
            st.download_button(
                label="Click to download",
                data=json.dumps(results, indent=2),
                file_name="agent_results.json",
                mime="application/json"
            )
    
    with col2:
        st.write("**Analysis Data:**")
        if st.button("Download Analysis Data (CSV)"):
            # Combine all analysis data
            combined_df = reasoning_df.merge(routing_df, on='query', suffixes=('_reasoning', '_routing'))
            combined_df = combined_df.merge(quality_df, on='query', suffixes=('', '_quality'))
            
            csv = combined_df.to_csv(index=False)
            st.download_button(
                label="Click to download",
                data=csv,
                file_name="agent_analysis.csv",
                mime="text/csv"
            )
    
    # Preview data
    st.subheader("Data Preview")
    tab1, tab2, tab3 = st.tabs(["Reasoning Data", "Routing Data", "Quality Data"])
    
    with tab1:
        st.dataframe(reasoning_df)
    with tab2:
        st.dataframe(routing_df)
    with tab3:
        st.dataframe(quality_df)

if __name__ == "__main__":
    main() 