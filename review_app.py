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
    try:
        # Dynamically import agent.py
        spec = importlib.util.spec_from_file_location("agent", os.path.join(os.getcwd(), "agent.py"))
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        agent = agent_module.CustomerSupportAgent()
        return agent.run(query)
    except ValueError as e:
        if "API key" in str(e):
            return "ERROR: OpenAI API key not configured. Please set your API key in Streamlit secrets or environment variables."
        else:
            return f"ERROR: {str(e)}"
    except Exception as e:
        return f"ERROR: Failed to initialize agent - {str(e)}"

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
    """Show generated visualizations from visualize.py"""
    st.header("📈 Generated Visualizations")
    
    # Define plot files with their descriptions
    plot_configs = {
        "response_lengths.png": {
            "title": "Response Length Distribution",
            "description": "Histogram showing the distribution of agent response lengths in characters. Helps understand response complexity and consistency."
        },
        "reasoning_patterns.png": {
            "title": "Reasoning Keywords Analysis", 
            "description": "Distribution of reasoning keywords (because, since, therefore, etc.) found in agent responses. Indicates reasoning depth."
        },
        "task_routing.png": {
            "title": "Task Routing Distribution",
            "description": "Pie chart showing how queries are categorized and routed (order status, escalation, general inquiries)."
        },
        "quality_analysis.png": {
            "title": "Response Quality Scores",
            "description": "Histogram of quality scores (0-100) based on reasoning presence, action items, and response length."
        }
    }
    
    # Check for generated plots
    available_plots = [f for f in plot_configs.keys() if os.path.exists(f)]
    missing_plots = [f for f in plot_configs.keys() if not os.path.exists(f)]
    
    if not available_plots:
        st.warning("⚠️ No visualization files found!")
        st.info("💡 To generate visualizations, run: `python visualize.py`")
        st.write("**Expected files:**")
        for plot_file in plot_configs.keys():
            st.write(f"- {plot_file}")
        return
    
    # Show status of available vs missing plots
    if missing_plots:
        st.info(f"📊 Showing {len(available_plots)} of {len(plot_configs)} visualizations")
        if st.checkbox("Show missing files"):
            st.write("**Missing visualization files:**")
            for plot_file in missing_plots:
                st.write(f"- {plot_file}")
    
    # Display plots in two columns with enhanced information
    st.subheader("📊 Research Visualizations")
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    for i, plot_file in enumerate(available_plots):
        # Choose which column to use
        current_col = col1 if i % 2 == 0 else col2
        
        with current_col:
            # Get plot configuration
            config = plot_configs[plot_file]
            
            # Create expandable section for each plot
            with st.expander(f"📈 {config['title']}", expanded=True):
                # Display the image
                st.image(plot_file, caption=config['title'], use_column_width=True)
                
                # Add description
                st.write(f"**Description:** {config['description']}")
                
                # Add file info
                file_size = os.path.getsize(plot_file) / 1024  # KB
                st.write(f"**File:** {plot_file} ({file_size:.1f} KB)")
                
                # Add timestamp if available
                try:
                    file_time = datetime.fromtimestamp(os.path.getmtime(plot_file))
                    st.write(f"**Generated:** {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
                except:
                    pass
    
    # Add summary information
    st.subheader("📋 Visualization Summary")
    
    # Calculate some basic stats from the results for context
    if results:
        total_queries = len(results)
        successful_queries = len([r for r in results if not r['response'].startswith('ERROR')])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries", total_queries)
        with col2:
            st.metric("Successful Responses", successful_queries)
        with col3:
            st.metric("Visualizations Available", len(available_plots))
    
    # Add instructions for generating new visualizations
    st.subheader("🔄 Regenerate Visualizations")
    st.write("To update these visualizations with new data:")
    st.code("python visualize.py", language="bash")
    st.write("This will analyze the current `test_results.json` and create updated plots.")

def show_research_metrics(results):
    """Show comprehensive research metrics using agent analytics"""
    st.header("📊 Research Metrics Summary")
    
    # Check if we have results to analyze
    if not results:
        st.warning("⚠️ No results data available for analysis.")
        st.info("💡 Run `python test_agent.py` to generate test results first.")
        return
    
    # Try to get analytics from agent (if available)
    try:
        # Dynamically import agent to get analytics
        spec = importlib.util.spec_from_file_location("agent", os.path.join(os.getcwd(), "agent.py"))
        agent_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agent_module)
        
        # Create agent instance and load responses
        agent = agent_module.CustomerSupportAgent()
        agent.responses = results  # Load the results into agent
        analytics = agent.get_analytics()
        
        if analytics:
            st.success("✅ Using agent analytics for comprehensive metrics")
            display_agent_analytics(analytics)
        else:
            st.warning("⚠️ Agent analytics not available, using basic calculations")
            display_basic_analytics(results)
            
    except Exception as e:
        st.warning(f"⚠️ Could not load agent analytics: {str(e)}")
        st.info("💡 Using basic analytics instead")
        display_basic_analytics(results)

def display_agent_analytics(analytics):
    """Display metrics using agent's get_analytics method"""
    
    # Main metrics in 2x2 layout
    st.subheader("📈 Core Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Total Queries", 
            value=analytics.get('total_queries', 0),
            help="Total number of queries processed by the agent"
        )
        st.metric(
            label="Success Rate", 
            value=f"{analytics.get('success_rate', 0):.1f}%",
            help="Percentage of queries that were processed successfully"
        )
    
    with col2:
        st.metric(
            label="Avg Response Length", 
            value=f"{analytics.get('avg_response_length', 0):.0f} chars",
            help="Average number of characters in agent responses"
        )
        st.metric(
            label="Successful Queries", 
            value=analytics.get('successful_queries', 0),
            help="Number of queries that were processed without errors"
        )
    
    # Category distribution with pie chart
    st.subheader("📊 Category Distribution Analysis")
    
    category_distribution = analytics.get('category_distribution', {})
    
    if category_distribution:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Query Categories:**")
            total_categories = sum(category_distribution.values())
            
            for category, count in category_distribution.items():
                percentage = (count / total_categories * 100) if total_categories > 0 else 0
                st.write(f"• **{category.replace('_', ' ').title()}**: {count} queries ({percentage:.1f}%)")
        
        with col2:
            # Create pie chart for category distribution
            if len(category_distribution) > 0:
                categories = list(category_distribution.keys())
                counts = list(category_distribution.values())
                
                # Create a more readable pie chart
                fig = px.pie(
                    values=counts,
                    names=categories,
                    title="Query Category Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                
                # Customize the pie chart
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hole=0.3  # Make it a donut chart
                )
                
                fig.update_layout(
                    title_x=0.5,
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.05
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No category data available")
    else:
        st.info("📝 No category distribution data available")
    
    # Response length statistics
    st.subheader("📏 Response Length Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Minimum Length", 
            value=f"{analytics.get('min_response_length', 0):.0f} chars",
            help="Shortest response length"
        )
    
    with col2:
        st.metric(
            label="Maximum Length", 
            value=f"{analytics.get('max_response_length', 0):.0f} chars",
            help="Longest response length"
        )
    
    with col3:
        st.metric(
            label="Average Length", 
            value=f"{analytics.get('avg_response_length', 0):.0f} chars",
            help="Average response length across all queries"
        )

def display_basic_analytics(results):
    """Display basic analytics when agent analytics are not available"""
    
    # Calculate basic metrics
    total_queries = len(results)
    successful_queries = len([r for r in results if not r['response'].startswith('ERROR')])
    success_rate = (successful_queries / total_queries) * 100 if total_queries > 0 else 0
    
    # Response length metrics
    response_lengths = [len(r['response']) for r in results if not r['response'].startswith('ERROR')]
    avg_length = np.mean(response_lengths) if response_lengths else 0
    median_length = np.median(response_lengths) if response_lengths else 0
    
    # Display metrics in 2x2 layout
    st.subheader("📈 Basic Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Queries", total_queries)
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col2:
        st.metric("Avg Response Length", f"{avg_length:.0f} chars")
        st.metric("Successful Queries", successful_queries)
    
    # Simple category analysis
    st.subheader("📊 Basic Category Analysis")
    
    # Simple keyword-based categorization
    categories = {
        'technical_support': 0,
        'billing_inquiry': 0,
        'complaint_handling': 0,
        'general_inquiry': 0
    }
    
    for result in results:
        query_lower = result['query'].lower()
        if any(word in query_lower for word in ['broken', 'crash', 'error', 'technical']):
            categories['technical_support'] += 1
        elif any(word in query_lower for word in ['charge', 'billing', 'payment', 'refund']):
            categories['billing_inquiry'] += 1
        elif any(word in query_lower for word in ['complaint', 'angry', 'frustrated', 'damaged']):
            categories['complaint_handling'] += 1
        else:
            categories['general_inquiry'] += 1
    
    # Display category distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Query Categories:**")
        for category, count in categories.items():
            if count > 0:
                percentage = (count / total_queries * 100) if total_queries > 0 else 0
                st.write(f"• **{category.replace('_', ' ').title()}**: {count} queries ({percentage:.1f}%)")
    
    with col2:
        # Create pie chart for basic categories
        non_zero_categories = {k: v for k, v in categories.items() if v > 0}
        
        if non_zero_categories:
            fig = px.pie(
                values=list(non_zero_categories.values()),
                names=list(non_zero_categories.keys()),
                title="Basic Category Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hole=0.3
            )
            
            fig.update_layout(title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No category data available")

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