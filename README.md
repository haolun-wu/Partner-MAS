# Partner-MAS: Multi-Agent System for Co-Investor Selection

A sophisticated multi-agent system designed to evaluate and select optimal co-investor partnerships in venture capital deals. The system uses AI-powered specialized agents to analyze potential co-investors from multiple perspectives and make data-driven selection decisions.

## 🎯 Overview

Partner-MAS employs a three-tier agent architecture to simulate the decision-making process of venture capital firms when selecting co-investors:

1. **Planner Agent** - Designs the multi-agent system architecture for each specific investment case
2. **Specialized Agents** - Evaluate candidates from distinct perspectives (e.g., industry expertise, geographic alignment, financial metrics)
3. **Supervisor Agent** - Makes final selection decisions by synthesizing all agent evaluations

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Planner Agent  │───▶│ Specialized      │───▶│ Supervisor      │
│                 │    │ Agents (3-5)     │    │ Agent           │
│ • System Design │    │ • Market Analysis│    │ • Final         │
│ • Agent Config  │    │ • Financial Eval │    │   Selection     │
│ • Strategy      │    │ • Strategic Fit  │    │ • Rationale     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Key Features

- **Adaptive Agent Design**: Planner dynamically creates specialized agents based on investment context
- **Multi-Perspective Evaluation**: Each agent evaluates candidates from a unique angle (industry, geography, financials, etc.)
- **Pydantic Integration**: Structured outputs with validation for reliable data processing
- **Flexible Decision Modes**: Multiple supervisor decision strategies (importance ranking, weighted scoring, majority vote)
- **Performance Tracking**: Comprehensive logging with token usage, timing, and match rate analysis
- **Parallel Processing**: Efficient batch processing of multiple investment cases

## 📋 Requirements

- Python 3.8+
- OpenAI API access (GPT-4o, GPT-4o-mini, GPT-4.1-mini, or GPT-5 series)
- Required packages: `pandas`, `pydantic`, `openai`

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Partner-MAS
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas pydantic openai
   ```

3. **Set up API credentials:**
   Create a `secret.json` file in the project root:
   ```json
   [
     {
       "API provider": "OpenAI",
       "API key": "your-openai-api-key-here"
     }
   ]
   ```

4. **Prepare your data:**
   - Place CSV files with investment data in a `data/` directory
   - Each CSV should contain columns for VC firms, companies, and investment details
   - Required columns: `vcfirmid`, `leadornot`, `real`, `companyid`, etc.

## 🎮 Usage

### Basic Usage

Run the system on all CSV files in your data directory:

```bash
python main.py
```

### Configuration

Modify `config.py` to customize:

```python
# Model configurations
planner_config = {
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    "prompt_strategy": "business"  # or "generic"
}

specialized_config = {
    "model": "gpt-4.1-mini", 
    "temperature": 0.0
}

supervisor_config = {
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    "mode": "importance",  # "importance", "weight", or "majority_vote"
    "importance_strategy": "business"  # or "generic"
}
```

### Example Workflow

1. **Load Investment Case**: System reads CSV with lead investor, target company, and candidate co-investors
2. **Plan Agents**: Planner designs 3-5 specialized agents based on investment context
3. **Evaluate Candidates**: Each specialized agent ranks candidates from their perspective
4. **Final Selection**: Supervisor synthesizes all evaluations to select optimal co-investors
5. **Performance Analysis**: System calculates match rates against actual co-investor data

## 📊 Output

The system generates:

- **Agent Evaluations**: Detailed rankings and rationales from each specialized agent
- **Final Shortlist**: Selected co-investors with supervisor rationale
- **Performance Metrics**: Match rate against actual historical co-investors
- **Comprehensive Logs**: JSON logs with complete session details, token usage, and timing

Example output:
```
🎯 CASE SUMMARY:
   📋 Candidates evaluated: 45
   🎯 Final shortlist: 15 firms
   📈 Match rate: 73.33%
   ✅ Processing: SUCCESS
```

## 📁 Project Structure

```
Partner-MAS/
├── main.py              # Main execution script
├── mas.py               # Multi-Agent System orchestrator
├── agent.py             # Agent classes (Planner, Specialized, Supervisor)
├── config.py            # Configuration settings
├── utilities.py         # OpenAI client and data processing utilities
├── prompts.py           # Prompt management system
├── pydantic_schemas.py  # Data validation schemas
├── project_types.py     # Type definitions
├── prompts/
│   ├── prompts.md       # Agent prompts library
│   └── schemas.json     # JSON schemas for structured outputs
├── data/                # CSV investment data files (create this)
└── logs/                # Generated session logs (auto-created)
```

## 🔧 Advanced Features

### Parallel Processing
Process multiple cases efficiently:
```python
from parallel_test import ParallelProcessor
processor = ParallelProcessor(num_workers=4)
# Process multiple CSV files in parallel
```

### Custom Agent Strategies
- **Business Strategy**: Domain-specific prompts with VC industry knowledge
- **Generic Strategy**: General-purpose evaluation without domain hints

### Decision Modes
- **Importance Mode**: Rank agents by importance, then make weighted decisions
- **Weight Mode**: Assign numerical weights to agent recommendations  
- **Majority Vote**: Democratic consensus across all agents

## 📈 Performance Analysis

The system tracks several key metrics:

- **Match Rate**: Percentage of actual co-investors correctly identified
- **Token Usage**: Detailed breakdown of API costs and usage
- **Processing Time**: Performance metrics for optimization
- **Agent Effectiveness**: Individual agent performance analysis

## 🛠️ Customization

### Adding New Agent Types
1. Modify the planner prompts in `prompts/prompts.md`
2. Update Pydantic schemas in `pydantic_schemas.py`
3. Add new evaluation criteria in specialized agent prompts

### Custom Data Sources
Implement custom data loaders in `utilities.py`:
```python
class CustomDataExtractor(ProfileExtractor):
    def load_from_source(self, source):
        # Your custom data loading logic
        pass
```

## 🔍 Testing

Run in test mode without API calls:
```python
# Set test mode in secret.json
{
  "API provider": "OpenAI", 
  "API key": "test-key-for-local-testing-only"
}
```

## 📝 Logging

Comprehensive logs are saved in JSON format containing:
- Complete agent conversations and decisions
- Performance metrics and token usage
- Configuration snapshots
- Match rate analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🔗 Citation

If you use this work in your research, please cite:
```bibtex
@software{partner_mas_2025,
  title={Partner-MAS: Multi-Agent System for Co-Investor Selection},
  author={[Your Name]},
  year={2025},
  url={[Repository URL]}
}
```

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Contact: [your-email@example.com]

---

**Note**: This system is designed for research and educational purposes. Ensure compliance with data privacy regulations when processing real investment data.
