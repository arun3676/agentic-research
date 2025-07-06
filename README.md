# AI Agent Research Project: Transparent Customer Support Agent

A research-ready implementation of a LangChain-based customer support agent using OpenAI's GPT-4 API. This project demonstrates transparent reasoning, task routing, and agent orchestration for AI/ML engineering research and code review.

## 🚀 Project Overview

This project implements a customer support agent that:
- Uses Chain-of-Thought (CoT) prompting for transparent reasoning
- Implements task routing (order status checks, escalation)
- Maintains conversation memory for context-aware responses
- Provides comprehensive analysis for research paper presentation

## 📁 Project Structure

```
Agentic research paper/
├── agent.py                # Main agent logic with CoT prompting
├── test_agent.py           # Batch testing script
├── visualize.py            # Analysis and visualization tools
├── review_app.py           # Streamlit web interface
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── test_results.json       # Generated test results (after running tests)
├── response_lengths.png    # (optional) Visualization output
├── reasoning_patterns.png  # (optional) Visualization output
├── task_routing.png        # (optional) Visualization output
├── quality_analysis.png    # (optional) Visualization output
```

## ⚡ Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set OpenAI API Key**
   - Environment variable (recommended):
     ```bash
     export OPENAI_API_KEY="your-api-key-here"
     ```
   - Or, for Streamlit deployment, add to `.streamlit/secrets.toml`:
     ```toml
     [openai]
     api_key = "your-api-key-here"
     ```
3. **Test the Agent**
   ```bash
   python test_agent.py
   ```
4. **Generate Visualizations (optional)**
   ```bash
   python visualize.py
   ```
5. **Launch Web Interface**
   ```bash
   streamlit run review_app.py
   ```

## 🧑‍💻 How to Upload to Grok (or other code review tools)

1. Ensure your project folder contains only the files listed above.
2. (Optional) Run `python test_agent.py` and `python visualize.py` to generate `test_results.json` and PNGs for richer context.
3. Zip the entire `Agentic research paper` folder.
4. Upload the ZIP to Grok or your code review platform.

## 📝 Key Features
- Transparent reasoning output (Chain-of-Thought)
- Task routing and classification
- Batch testing and research analytics
- Streamlit web interface for interactive review
- Optional visualizations for research presentation

## 📊 Research & Analysis
- **Chain-of-Thought Analysis:** Forces explicit reasoning steps for transparency
- **Task Routing:** Demonstrates agent orchestration and planning
- **Quality Metrics:** Response length, reasoning keyword frequency, routing effectiveness, and quality scoring

## 🛠️ Technical Requirements
- Python 3.8+
- OpenAI API key
- Internet connection for API calls

## 🧩 Usage Examples

**Basic Agent Usage**
```python
from agent import CustomerSupportAgent
agent = CustomerSupportAgent()
response = agent.run("Why is my order delayed?")
print(response)
```

**Batch Testing**
```bash
python test_agent.py
```

**Analysis and Visualization**
```bash
python visualize.py
```

**Web Interface**
```bash
streamlit run review_app.py
```

## 📚 For More Information
- See `DEPLOYMENT.md` for deployment instructions.
- See `DEPLOYMENT_SUMMARY.md` for a summary of deployment steps.

---

**Ready for research, review, and presentation. Zip and upload this folder to Grok or your preferred code review tool!** 