# Partner-MAS

## Overview

The Collaboration Framework (Business-MAS-v2) is a sophisticated multi-agent system designed for co-investor selection that emphasizes adaptive agent design, strategic planning, and hierarchical decision-making. Unlike iterative debate systems, this framework focuses on dynamic agent creation, specialized evaluation, and supervisor-guided final selection through multiple decision modes.

## Core Design Principles

### 1. **Adaptive Agent Design**
- Dynamic agent creation based on investment context
- Planner agent designs specialized agents for each case
- Context-aware agent configuration and specialization

### 2. **Strategic Planning & Guidance**
- High-level strategic guidance from planner
- Case-specific agent design and configuration
- Strategic alignment throughout the evaluation process

### 3. **Hierarchical Decision Making**
- Three-tier architecture: Planner → Specialized Agents → Supervisor
- Multiple supervisor decision modes (importance, weight, majority vote)
- Structured decision-making with clear authority levels

### 4. **Structured Validation & Output**
- Pydantic integration for robust data validation
- Structured JSON outputs with comprehensive schemas
- Retry logic and fallback mechanisms for reliability

## System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Collaboration Framework                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                Planner Agent                                │ │
│  │              (System Architect)                            │ │
│  │                                                             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │   Analyze   │  │   Design    │  │   Generate  │        │ │
│  │  │   Context   │  │   Agents    │  │   Strategy  │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Specialized Agents                             │ │
│  │                                                             │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │ │
│  │  │   Agent 1   │    │   Agent 2   │    │   Agent 3   │    │ │
│  │  │ (Dynamic)   │    │ (Dynamic)   │    │ (Dynamic)   │    │ │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    │ │
│  │         │                   │                   │          │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │ Evaluate &  │  │ Evaluate &  │  │ Evaluate &  │        │ │
│  │  │   Rank      │  │   Rank      │  │   Rank      │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                Supervisor Agent                             │ │
│  │              (Graham Paxon)                                │ │
│  │                                                             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │ Importance  │  │   Weight    │  │  Majority   │        │ │
│  │  │   Mode      │  │   Mode      │  │   Vote      │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  │         │                   │                   │          │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │   Final     │  │   Final     │  │   Final     │        │ │
│  │  │ Selection   │  │ Selection   │  │ Selection   │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. **Planner Agent Component**

#### **Core Planner Class**
```python
class PlannerAgent:
    def __init__(self, name, role, ability, model, temperature, 
                 prompt_strategy, reasoning_effort, verbosity, api_secret_path):
        # Configuration and client initialization
```

#### **Planner Capabilities**
- **Context Analysis**: Analyze lead investor, target company, and candidate profiles
- **Agent Design**: Create specialized agents tailored to investment context
- **Strategic Guidance**: Provide high-level strategic direction for evaluation
- **Dynamic Configuration**: Adapt agent design based on case requirements

#### **Prompt Strategies**
- **Business Strategy**: Domain knowledge hints and business context
- **Generic Strategy**: General evaluation without domain-specific guidance

#### **Output Structure**
```json
{
  "strategic_guidance": "High-level strategic direction",
  "agents": [
    {
      "name": "Agent Name",
      "role": "Agent Role",
      "ability": "Agent Capabilities",
      "profile": "Specific Focus Areas"
    }
  ]
}
```

### 2. **Specialized Agent Component**

#### **Core Specialized Agent Class**
```python
class SpecializedAgent:
    def __init__(self, name, role, ability, profile, model, temperature, 
                 reasoning_effort, verbosity, api_secret_path):
        # Agent configuration and client initialization
```

#### **Specialized Agent Capabilities**
- **Focused Evaluation**: Evaluate candidates from specific perspective
- **Dynamic Ranking**: Rank candidates based on specialized criteria
- **Alignment Scoring**: Provide alignment scores (1-10) for each candidate
- **Rationale Generation**: Generate detailed rationale for rankings

#### **Evaluation Framework**
- **Evaluation Focus**: Key features and criteria for analysis
- **Overall Rationale**: Methodology and reasoning approach
- **Ranked Candidates**: Ordered list with alignment scores and rationales

#### **Dynamic Top-K Calculation**
```python
dynamic_top_k = max(1, math.ceil(total_candidates / 3))
dynamic_top_k = min(dynamic_top_k, total_candidates)
```

### 3. **Supervisor Agent Component**

#### **Core Supervisor Class**
```python
class SupervisorAgent:
    def __init__(self, name, role, ability, profile, model, temperature, 
                 mode, importance_strategy, reasoning_effort, verbosity, api_secret_path):
        # Supervisor configuration and decision mode
```

#### **Supervisor Decision Modes**

##### **Importance Mode**
- **Step 1**: Rank agents by importance for specific investment
- **Step 2**: Use importance ranking to resolve conflicts and make final selection
- **Business vs Generic**: Domain knowledge hints vs general evaluation

##### **Weight Mode**
- **Step 1**: Assign numerical weights to agents (sum = 1.0)
- **Step 2**: Use weighted scoring for final candidate selection
- **Mathematical Approach**: Quantitative decision-making

##### **Majority Vote Mode**
- **Consensus Building**: Identify candidates with broad agent support
- **Democratic Decision**: Select based on agent agreement
- **Collaborative Approach**: Collective decision-making

#### **Supervisor Capabilities**
- **Strategic Integration**: Synthesize all agent insights
- **Conflict Resolution**: Resolve disagreements between agents
- **Final Authority**: Ultimate decision-making power
- **Comprehensive Rationale**: Detailed explanation of final decisions

### 4. **Multi-Agent System (MAS) Orchestrator**

#### **Core MAS Class**
```python
class MAS:
    def __init__(self, specialized_agents, candidate_profiles, csv_filename, 
                 planner_output, api_secret_path):
        # System initialization and coordination
```

#### **MAS Responsibilities**
- **Workflow Orchestration**: Coordinate the complete evaluation pipeline
- **State Management**: Track evaluations, decisions, and outcomes
- **Metrics Collection**: Token usage, timing, performance tracking
- **Logging & Persistence**: Comprehensive session logging

#### **Evaluation Pipeline**
```
1. Profile Loading → 2. Planner Execution → 3. Agent Creation → 
4. Specialized Evaluation → 5. Supervisor Decision → 6. Final Selection
```

## Data Flow Architecture

### 1. **Input Processing Flow**
```
CSV File → Profile Extraction → Lead/Target/Candidate Profiles → MAS Initialization
```

### 2. **Planner Execution Flow**
```
┌─────────────────┐
│   Context       │
│   Analysis      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Agent Design  │
│   & Strategy    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Agent         │
│   Configs       │
└─────────────────┘
```

### 3. **Specialized Agent Evaluation Flow**
```
┌─────────────────┐
│   Agent         │
│   Creation      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Candidate     │
│   Evaluation    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Ranking &     │
│   Scoring       │
└─────────────────┘
```

### 4. **Supervisor Decision Flow**
```
┌─────────────────┐
│   Agent         │
│   Evaluations   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Decision      │
│   Mode          │
│   Selection     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Final         │
│   Selection     │
└─────────────────┘
```

## Configuration Architecture

### 1. **System Configuration**
```python
config = {
    "data_dir": "data",
    "logs_dir": "dynamic_log_directory",
    "planner_agent": {...},
    "specialized_agent": {...},
    "supervisor_agent": {...},
    "pydantic": {...}
}
```

### 2. **Agent Configuration**
- **Planner Agent**: Model, temperature, prompt strategy, reasoning effort
- **Specialized Agent**: Model, temperature, reasoning effort, verbosity
- **Supervisor Agent**: Model, temperature, mode, importance strategy

### 3. **Pydantic Integration**
```python
pydantic_config = {
    "use_pydantic_validation": True,
    "enable_structured_outputs": True,
    "retry_on_validation_failure": True,
    "max_validation_retries": 3
}
```

## Evaluation Framework

### 1. **Multi-Dimensional Evaluation**
- **Alignment Scoring (1-10)**: How well candidate matches ideal criteria
- **Ranking**: Ordered list based on alignment scores
- **Rationale**: Detailed explanation for each ranking decision

### 2. **Dynamic Shortlist Sizing**
- **Adaptive Top-K**: Based on total candidate count
- **Proportional Selection**: Maintains selection ratio
- **Context-Aware**: Adjusts to case requirements

### 3. **Supervisor Integration**
- **Strategic Guidance**: Planner's high-level direction
- **Agent Importance**: Relative importance of each agent's perspective
- **Final Authority**: Supervisor's ultimate decision-making power

## Logging & Persistence Architecture

### 1. **Session Logging Structure**
```json
{
    "case_info": {
        "timestamp": "20250106_125501",
        "csv_file": "case_714",
        "pydantic_integration": true
    },
    "performance_summary": {
        "total_case_processing_time_seconds": 45.2,
        "total_prompt_tokens": 12500,
        "total_completion_tokens": 3200,
        "total_tokens": 15700
    },
    "run_details": {
        "planner_output": {...},
        "specialized_agent_evaluations": [...],
        "supervisor_evaluation": {...},
        "final_summary": {...}
    },
    "config_snapshot": {...}
}
```

### 2. **Component-Level Logging**
- **Planner Output**: Agent configurations and strategic guidance
- **Specialized Evaluations**: Individual agent evaluations and rankings
- **Supervisor Evaluation**: Decision process and final selection
- **Final Summary**: Match statistics and performance metrics

### 3. **Performance Metrics**
- **Token Usage**: Prompt and completion tokens across all agents
- **Processing Time**: Total case processing time
- **Match Statistics**: Real match detection and success rates

## Error Handling & Resilience

### 1. **Graceful Degradation**
- **API Failures**: Fallback to mock responses for testing
- **Validation Errors**: Retry logic with Pydantic validation
- **Agent Failures**: Continue with remaining agents

### 2. **Validation Framework**
- **Pydantic Models**: Robust data validation and parsing
- **Schema Validation**: Structured output validation
- **Error Recovery**: Fallback mechanisms for failed operations

### 3. **Recovery Mechanisms**
- **Mock Responses**: Testing mode when API unavailable
- **Partial Results**: Handle incomplete evaluations
- **Fallback Selection**: Alternative selection methods

## Performance Optimization

### 1. **Structured Outputs**
- **Pydantic Integration**: Efficient data validation and parsing
- **Schema-Based**: Consistent output formats
- **Retry Logic**: Robust error handling and recovery

### 2. **Token Optimization**
- **Efficient Prompts**: Optimized prompt templates
- **Context Management**: Effective context passing
- **Response Parsing**: Minimal token overhead

### 3. **Parallel Processing**
- **Agent Parallelization**: Simultaneous agent evaluations
- **Pipeline Optimization**: Efficient workflow management
- **Resource Management**: Optimal resource utilization

## Scalability Considerations

### 1. **Dynamic Agent Scaling**
- **Context-Aware**: Agents designed for specific cases
- **Flexible Configuration**: Adaptable agent parameters
- **Load Distribution**: Even workload across agents

### 2. **Data Scaling**
- **Large Candidate Sets**: Efficient handling of many candidates
- **Memory Management**: Optimal memory usage
- **Storage Optimization**: Efficient logging and persistence

### 3. **System Scaling**
- **Horizontal Scaling**: Multiple system instances
- **Vertical Scaling**: Enhanced system capabilities
- **Cloud Integration**: Cloud-based deployment options

## Integration Points

### 1. **External Systems**
- **LLM APIs**: OpenAI, Anthropic, etc.
- **Data Sources**: CSV files, databases
- **Monitoring**: Performance and usage tracking

### 2. **Internal Components**
- **Utilities**: Profile extraction and data processing
- **Prompts**: Dynamic prompt management
- **Schemas**: Pydantic model definitions

### 3. **Output Systems**
- **Logging**: Comprehensive session logs
- **Results**: Final selections and rankings
- **Analytics**: Performance and match statistics

## Security & Privacy

### 1. **Data Protection**
- **Sensitive Data**: Secure handling of investment data
- **API Keys**: Secure API key management
- **Logging**: Privacy-preserving logging

### 2. **Access Control**
- **Authentication**: System access control
- **Authorization**: Role-based permissions
- **Audit Trail**: Complete activity logging

## Advanced Features

### 1. **Pydantic Integration**
- **Structured Validation**: Robust data validation
- **Type Safety**: Strong typing and validation
- **Error Handling**: Comprehensive error management

### 2. **Multiple Decision Modes**
- **Importance Mode**: Agent importance ranking
- **Weight Mode**: Numerical weight assignment
- **Majority Vote Mode**: Consensus-based selection

### 3. **Dynamic Configuration**
- **Context-Aware**: Case-specific agent design
- **Adaptive Parameters**: Dynamic configuration adjustment
- **Flexible Strategies**: Multiple evaluation approaches

## Future Enhancements

### 1. **Advanced Features**
- **Learning Agents**: Adaptive agent capabilities
- **Real-time Collaboration**: Live agent interaction
- **Predictive Modeling**: Outcome prediction capabilities

### 2. **Integration Enhancements**
- **API Integration**: RESTful API endpoints
- **Web Interface**: User-friendly web interface
- **Mobile Support**: Mobile application support

### 3. **Analytics & Insights**
- **Performance Analytics**: Detailed performance metrics
- **Decision Insights**: Decision pattern analysis
- **Optimization**: System performance optimization

---

This architecture design provides a comprehensive framework for understanding and implementing the collaboration framework, emphasizing its unique approach to adaptive agent design, strategic planning, and hierarchical decision-making through multiple supervisor modes.
# Collaboration Framework Architecture Design (Business-MAS-v2)

## Overview

The Collaboration Framework (Business-MAS-v2) is a sophisticated multi-agent system designed for co-investor selection that emphasizes adaptive agent design, strategic planning, and hierarchical decision-making. Unlike iterative debate systems, this framework focuses on dynamic agent creation, specialized evaluation, and supervisor-guided final selection through multiple decision modes.

## Core Design Principles

### 1. **Adaptive Agent Design**
- Dynamic agent creation based on investment context
- Planner agent designs specialized agents for each case
- Context-aware agent configuration and specialization

### 2. **Strategic Planning & Guidance**
- High-level strategic guidance from planner
- Case-specific agent design and configuration
- Strategic alignment throughout the evaluation process

### 3. **Hierarchical Decision Making**
- Three-tier architecture: Planner → Specialized Agents → Supervisor
- Multiple supervisor decision modes (importance, weight, majority vote)
- Structured decision-making with clear authority levels

### 4. **Structured Validation & Output**
- Pydantic integration for robust data validation
- Structured JSON outputs with comprehensive schemas
- Retry logic and fallback mechanisms for reliability

## System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Collaboration Framework                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                Planner Agent                                │ │
│  │              (System Architect)                            │ │
│  │                                                             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │   Analyze   │  │   Design    │  │   Generate  │        │ │
│  │  │   Context   │  │   Agents    │  │   Strategy  │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Specialized Agents                             │ │
│  │                                                             │ │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │ │
│  │  │   Agent 1   │    │   Agent 2   │    │   Agent 3   │    │ │
│  │  │ (Dynamic)   │    │ (Dynamic)   │    │ (Dynamic)   │    │ │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    │ │
│  │         │                   │                   │          │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │ Evaluate &  │  │ Evaluate &  │  │ Evaluate &  │        │ │
│  │  │   Rank      │  │   Rank      │  │   Rank      │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                Supervisor Agent                             │ │
│  │              (Graham Paxon)                                │ │
│  │                                                             │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │ Importance  │  │   Weight    │  │  Majority   │        │ │
│  │  │   Mode      │  │   Mode      │  │   Vote      │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  │         │                   │                   │          │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │   Final     │  │   Final     │  │   Final     │        │ │
│  │  │ Selection   │  │ Selection   │  │ Selection   │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. **Planner Agent Component**

#### **Core Planner Class**
```python
class PlannerAgent:
    def __init__(self, name, role, ability, model, temperature, 
                 prompt_strategy, reasoning_effort, verbosity, api_secret_path):
        # Configuration and client initialization
```

#### **Planner Capabilities**
- **Context Analysis**: Analyze lead investor, target company, and candidate profiles
- **Agent Design**: Create specialized agents tailored to investment context
- **Strategic Guidance**: Provide high-level strategic direction for evaluation
- **Dynamic Configuration**: Adapt agent design based on case requirements

#### **Prompt Strategies**
- **Business Strategy**: Domain knowledge hints and business context
- **Generic Strategy**: General evaluation without domain-specific guidance

#### **Output Structure**
```json
{
  "strategic_guidance": "High-level strategic direction",
  "agents": [
    {
      "name": "Agent Name",
      "role": "Agent Role",
      "ability": "Agent Capabilities",
      "profile": "Specific Focus Areas"
    }
  ]
}
```

### 2. **Specialized Agent Component**

#### **Core Specialized Agent Class**
```python
class SpecializedAgent:
    def __init__(self, name, role, ability, profile, model, temperature, 
                 reasoning_effort, verbosity, api_secret_path):
        # Agent configuration and client initialization
```

#### **Specialized Agent Capabilities**
- **Focused Evaluation**: Evaluate candidates from specific perspective
- **Dynamic Ranking**: Rank candidates based on specialized criteria
- **Alignment Scoring**: Provide alignment scores (1-10) for each candidate
- **Rationale Generation**: Generate detailed rationale for rankings

#### **Evaluation Framework**
- **Evaluation Focus**: Key features and criteria for analysis
- **Overall Rationale**: Methodology and reasoning approach
- **Ranked Candidates**: Ordered list with alignment scores and rationales

#### **Dynamic Top-K Calculation**
```python
dynamic_top_k = max(1, math.ceil(total_candidates / 3))
dynamic_top_k = min(dynamic_top_k, total_candidates)
```

### 3. **Supervisor Agent Component**

#### **Core Supervisor Class**
```python
class SupervisorAgent:
    def __init__(self, name, role, ability, profile, model, temperature, 
                 mode, importance_strategy, reasoning_effort, verbosity, api_secret_path):
        # Supervisor configuration and decision mode
```

#### **Supervisor Decision Modes**

##### **Importance Mode**
- **Step 1**: Rank agents by importance for specific investment
- **Step 2**: Use importance ranking to resolve conflicts and make final selection
- **Business vs Generic**: Domain knowledge hints vs general evaluation

##### **Weight Mode**
- **Step 1**: Assign numerical weights to agents (sum = 1.0)
- **Step 2**: Use weighted scoring for final candidate selection
- **Mathematical Approach**: Quantitative decision-making

##### **Majority Vote Mode**
- **Consensus Building**: Identify candidates with broad agent support
- **Democratic Decision**: Select based on agent agreement
- **Collaborative Approach**: Collective decision-making

#### **Supervisor Capabilities**
- **Strategic Integration**: Synthesize all agent insights
- **Conflict Resolution**: Resolve disagreements between agents
- **Final Authority**: Ultimate decision-making power
- **Comprehensive Rationale**: Detailed explanation of final decisions

### 4. **Multi-Agent System (MAS) Orchestrator**

#### **Core MAS Class**
```python
class MAS:
    def __init__(self, specialized_agents, candidate_profiles, csv_filename, 
                 planner_output, api_secret_path):
        # System initialization and coordination
```

#### **MAS Responsibilities**
- **Workflow Orchestration**: Coordinate the complete evaluation pipeline
- **State Management**: Track evaluations, decisions, and outcomes
- **Metrics Collection**: Token usage, timing, performance tracking
- **Logging & Persistence**: Comprehensive session logging

#### **Evaluation Pipeline**
```
1. Profile Loading → 2. Planner Execution → 3. Agent Creation → 
4. Specialized Evaluation → 5. Supervisor Decision → 6. Final Selection
```

## Data Flow Architecture

### 1. **Input Processing Flow**
```
CSV File → Profile Extraction → Lead/Target/Candidate Profiles → MAS Initialization
```

### 2. **Planner Execution Flow**
```
┌─────────────────┐
│   Context       │
│   Analysis      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Agent Design  │
│   & Strategy    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Agent         │
│   Configs       │
└─────────────────┘
```

### 3. **Specialized Agent Evaluation Flow**
```
┌─────────────────┐
│   Agent         │
│   Creation      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Candidate     │
│   Evaluation    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Ranking &     │
│   Scoring       │
└─────────────────┘
```

### 4. **Supervisor Decision Flow**
```
┌─────────────────┐
│   Agent         │
│   Evaluations   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Decision      │
│   Mode          │
│   Selection     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Final         │
│   Selection     │
└─────────────────┘
```

## Configuration Architecture

### 1. **System Configuration**
```python
config = {
    "data_dir": "data",
    "logs_dir": "dynamic_log_directory",
    "planner_agent": {...},
    "specialized_agent": {...},
    "supervisor_agent": {...},
    "pydantic": {...}
}
```

### 2. **Agent Configuration**
- **Planner Agent**: Model, temperature, prompt strategy, reasoning effort
- **Specialized Agent**: Model, temperature, reasoning effort, verbosity
- **Supervisor Agent**: Model, temperature, mode, importance strategy

### 3. **Pydantic Integration**
```python
pydantic_config = {
    "use_pydantic_validation": True,
    "enable_structured_outputs": True,
    "retry_on_validation_failure": True,
    "max_validation_retries": 3
}
```

## Evaluation Framework

### 1. **Multi-Dimensional Evaluation**
- **Alignment Scoring (1-10)**: How well candidate matches ideal criteria
- **Ranking**: Ordered list based on alignment scores
- **Rationale**: Detailed explanation for each ranking decision

### 2. **Dynamic Shortlist Sizing**
- **Adaptive Top-K**: Based on total candidate count
- **Proportional Selection**: Maintains selection ratio
- **Context-Aware**: Adjusts to case requirements

### 3. **Supervisor Integration**
- **Strategic Guidance**: Planner's high-level direction
- **Agent Importance**: Relative importance of each agent's perspective
- **Final Authority**: Supervisor's ultimate decision-making power

## Logging & Persistence Architecture

### 1. **Session Logging Structure**
```json
{
    "case_info": {
        "timestamp": "20250106_125501",
        "csv_file": "case_714",
        "pydantic_integration": true
    },
    "performance_summary": {
        "total_case_processing_time_seconds": 45.2,
        "total_prompt_tokens": 12500,
        "total_completion_tokens": 3200,
        "total_tokens": 15700
    },
    "run_details": {
        "planner_output": {...},
        "specialized_agent_evaluations": [...],
        "supervisor_evaluation": {...},
        "final_summary": {...}
    },
    "config_snapshot": {...}
}
```

### 2. **Component-Level Logging**
- **Planner Output**: Agent configurations and strategic guidance
- **Specialized Evaluations**: Individual agent evaluations and rankings
- **Supervisor Evaluation**: Decision process and final selection
- **Final Summary**: Match statistics and performance metrics

### 3. **Performance Metrics**
- **Token Usage**: Prompt and completion tokens across all agents
- **Processing Time**: Total case processing time
- **Match Statistics**: Real match detection and success rates

## Error Handling & Resilience

### 1. **Graceful Degradation**
- **API Failures**: Fallback to mock responses for testing
- **Validation Errors**: Retry logic with Pydantic validation
- **Agent Failures**: Continue with remaining agents

### 2. **Validation Framework**
- **Pydantic Models**: Robust data validation and parsing
- **Schema Validation**: Structured output validation
- **Error Recovery**: Fallback mechanisms for failed operations

### 3. **Recovery Mechanisms**
- **Mock Responses**: Testing mode when API unavailable
- **Partial Results**: Handle incomplete evaluations
- **Fallback Selection**: Alternative selection methods

## Performance Optimization

### 1. **Structured Outputs**
- **Pydantic Integration**: Efficient data validation and parsing
- **Schema-Based**: Consistent output formats
- **Retry Logic**: Robust error handling and recovery

### 2. **Token Optimization**
- **Efficient Prompts**: Optimized prompt templates
- **Context Management**: Effective context passing
- **Response Parsing**: Minimal token overhead

### 3. **Parallel Processing**
- **Agent Parallelization**: Simultaneous agent evaluations
- **Pipeline Optimization**: Efficient workflow management
- **Resource Management**: Optimal resource utilization

## Scalability Considerations

### 1. **Dynamic Agent Scaling**
- **Context-Aware**: Agents designed for specific cases
- **Flexible Configuration**: Adaptable agent parameters
- **Load Distribution**: Even workload across agents

### 2. **Data Scaling**
- **Large Candidate Sets**: Efficient handling of many candidates
- **Memory Management**: Optimal memory usage
- **Storage Optimization**: Efficient logging and persistence

### 3. **System Scaling**
- **Horizontal Scaling**: Multiple system instances
- **Vertical Scaling**: Enhanced system capabilities
- **Cloud Integration**: Cloud-based deployment options

## Integration Points

### 1. **External Systems**
- **LLM APIs**: OpenAI, Anthropic, etc.
- **Data Sources**: CSV files, databases
- **Monitoring**: Performance and usage tracking

### 2. **Internal Components**
- **Utilities**: Profile extraction and data processing
- **Prompts**: Dynamic prompt management
- **Schemas**: Pydantic model definitions

### 3. **Output Systems**
- **Logging**: Comprehensive session logs
- **Results**: Final selections and rankings
- **Analytics**: Performance and match statistics

## Security & Privacy

### 1. **Data Protection**
- **Sensitive Data**: Secure handling of investment data
- **API Keys**: Secure API key management
- **Logging**: Privacy-preserving logging

### 2. **Access Control**
- **Authentication**: System access control
- **Authorization**: Role-based permissions
- **Audit Trail**: Complete activity logging

## Advanced Features

### 1. **Pydantic Integration**
- **Structured Validation**: Robust data validation
- **Type Safety**: Strong typing and validation
- **Error Handling**: Comprehensive error management

### 2. **Multiple Decision Modes**
- **Importance Mode**: Agent importance ranking
- **Weight Mode**: Numerical weight assignment
- **Majority Vote Mode**: Consensus-based selection

### 3. **Dynamic Configuration**
- **Context-Aware**: Case-specific agent design
- **Adaptive Parameters**: Dynamic configuration adjustment
- **Flexible Strategies**: Multiple evaluation approaches

## Future Enhancements

### 1. **Advanced Features**
- **Learning Agents**: Adaptive agent capabilities
- **Real-time Collaboration**: Live agent interaction
- **Predictive Modeling**: Outcome prediction capabilities

### 2. **Integration Enhancements**
- **API Integration**: RESTful API endpoints
- **Web Interface**: User-friendly web interface
- **Mobile Support**: Mobile application support

### 3. **Analytics & Insights**
- **Performance Analytics**: Detailed performance metrics
- **Decision Insights**: Decision pattern analysis
- **Optimization**: System performance optimization

---

This architecture design provides a comprehensive framework for understanding and implementing the collaboration framework, emphasizing its unique approach to adaptive agent design, strategic planning, and hierarchical decision-making through multiple supervisor modes.
