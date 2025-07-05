# AI Agent Research Project: Transparent Customer Support Agent

A research-ready implementation of a LangChain-based customer support agent using OpenAI's GPT-4 API. This project demonstrates transparent reasoning, task routing, and agent orchestration for AI/ML engineering research and resume enhancement.

## 🎯 Project Overview

This project implements a customer support agent that:
- **Uses Chain-of-Thought (CoT) prompting** for transparent reasoning
- **Implements task routing** (order status checks, escalation)
- **Maintains conversation memory** for context-aware responses
- **Provides comprehensive analysis** for research paper presentation

## 📁 Project Structure

```
/agent_project/
├── agent.py              # Main agent logic with CoT prompting
├── test_agent.py         # Batch testing script
├── visualize.py          # Analysis and visualization tools
├── review_app.py         # Streamlit web interface
├── requirements.txt      # Python dependencies
├── .streamlit/config.toml # Streamlit configuration
├── DEPLOYMENT.md         # Deployment guide
├── README.md            # This file
└── test_results.json    # Generated test results (after running tests)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key
**Option A: Environment Variable (Recommended)**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Option B: Streamlit Secrets (For Deployment)**
Create `.streamlit/secrets.toml`:
```toml
[openai]
api_key = "your-api-key-here"
```

### 3. Test the Agent
```bash
# Quick single test
python agent.py

# Batch testing on multiple queries
python test_agent.py

# Generate analysis and visualizations
python visualize.py

# Launch web interface
streamlit run review_app.py
```

### 4. Deploy to Production
See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment guide to Streamlit Community Cloud.

## 🔧 Core Components

### agent.py
**Main agent implementation with:**
- **CustomerSupportAgent class**: Modular agent with memory and task routing
- **Chain-of-Thought prompting**: Forces the LLM to explain its reasoning
- **Task router**: Simulates database checks and escalation logic
- **Conversation memory**: Maintains context across interactions

**Key Features:**
- Transparent reasoning output
- Mock task routing (order status, escalation)
- Modular design for easy extension

### test_agent.py
**Batch testing framework that:**
- Tests the agent on 10 diverse customer queries
- Saves results with timestamps and metadata
- Provides success/failure statistics
- Creates JSON output for analysis

**Sample queries include:**
- Order status inquiries
- Complaints and escalations
- Policy questions
- General support requests

### visualize.py
**Research analysis tools that create:**
- Response length distribution plots
- Reasoning pattern analysis
- Task routing visualization
- Quality score assessment
- Comprehensive research summary

### review_app.py
**Streamlit web interface that provides:**
- Interactive results review and analysis
- Live agent query testing
- Data visualization and metrics
- Export functionality for research data
- Professional presentation interface

## 📊 Research Insights

### Chain-of-Thought Analysis
The agent uses CoT prompting to:
- Force explicit reasoning steps
- Improve response transparency
- Enable reasoning pattern analysis
- Demonstrate prompt engineering skills

### Task Routing Demonstration
Shows agent orchestration through:
- Query classification
- Action planning
- Database simulation
- Escalation logic

### Quality Metrics
The analysis includes:
- Response length distribution
- Reasoning keyword frequency
- Task routing effectiveness
- Overall quality scoring

## 🎓 Research Paper Applications

### Technical Skills Demonstrated
1. **LLM Integration**: OpenAI API usage with LangChain
2. **Prompt Engineering**: Chain-of-thought implementation
3. **Agent Orchestration**: Task routing and planning
4. **Data Analysis**: Response analysis and visualization
5. **Research Methodology**: Systematic testing and evaluation

### Key Research Questions Addressed
- How does CoT prompting affect response quality?
- What patterns emerge in agent reasoning?
- How effective is task routing in customer support?
- What metrics best evaluate agent performance?

## 📈 Visualization Outputs

Running `visualize.py` generates:
- `response_lengths.png`: Distribution of response lengths
- `reasoning_patterns.png`: Reasoning keyword frequency
- `task_routing.png`: Task routing distribution pie chart
- `quality_analysis.png`: Quality score distribution

## 🔍 Customization Ideas

### For Research Enhancement
1. **Add attention analysis** using Hugging Face transformers
2. **Implement more complex task routing** with real databases
3. **Add sentiment analysis** for customer satisfaction
4. **Create A/B testing** for different prompt strategies

### For Resume Enhancement
1. **✅ Web interface** using Streamlit (implemented)
2. **Implement real-time chat** with WebSockets
3. **Add authentication** and user management
4. **✅ Deployment pipeline** with Streamlit Cloud (implemented)

## 🛠️ Technical Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for API calls

## 📝 Usage Examples

### Basic Agent Usage
```python
from agent import CustomerSupportAgent

agent = CustomerSupportAgent()
response = agent.run("Why is my order delayed?")
print(response)
```

### Batch Testing
```python
python test_agent.py
# Generates test_results.json with all responses
```

### Analysis and Visualization
```python
python visualize.py
# Creates plots and generates research summary

# Web Interface
streamlit run review_app.py
# Launches interactive web interface for analysis
```

## 🎯 Next Steps for Research

1. **✅ Run the complete pipeline** to generate data
2. **✅ Analyze the visualizations** for insights
3. **Extend with attention analysis** using transformers
4. **✅ Document findings** for your research paper
5. **✅ Prepare presentation** with generated plots
6. **🚀 Deploy to production** using Streamlit Cloud

## 📚 Research Paper Outline

### Suggested Structure
1. **Introduction**: AI agents in customer support
2. **Methodology**: CoT prompting and task routing
3. **Implementation**: LangChain and OpenAI integration
4. **Results**: Analysis of response patterns
5. **Discussion**: Implications for transparency
6. **Conclusion**: Future directions

### Key Metrics to Highlight
- Success rate of agent responses
- Reasoning transparency scores
- Task routing effectiveness
- Response quality distribution

---

**Ready to boost your AI/ML engineering resume? Start with `python test_agent.py` and see your agent in action!** 