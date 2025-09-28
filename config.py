# Configuration for Multi-Agent System for Co-Investor Selection

# --- Define components ---
data_dir_name = "data"

planner_config = {
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    # "reasoning_effort": "medium",
    # "verbosity": "low",
    # Options: "business" (with domain hints), "generic" (without domain hints)
    "prompt_strategy": "business"
}

specialized_config = {
    "model": "gpt-4.1-mini",
    "temperature": 0.0,
    # "reasoning_effort": "medium",
    # "verbosity": "low"
}

supervisor_config = {
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    # "reasoning_effort": "medium",
    # "verbosity": "low",
    # Options: "importance", "weight", "majority_vote"
    "mode": "importance",
    # Options: "business" (with domain hints), "generic" (without domain hints)
    "importance_strategy": "business"
}

# --- Create the log directory ---

# Safely get components for the log directory name
planner_effort = planner_config.get('reasoning_effort', 'na')
specialized_effort = specialized_config.get('reasoning_effort', 'na')
supervisor_effort = supervisor_config.get('reasoning_effort', 'na')

importance_strategy = supervisor_config.get('importance_strategy', 'na')

log_dir_name = (
    f"{data_dir_name}-"
    f"{planner_config['model']}-{planner_effort}-"
    f"{planner_config['prompt_strategy']}-"
    f"{specialized_config['model']}-{specialized_effort}-"
    f"{supervisor_config['model']}-{supervisor_effort}-"
    f"{supervisor_config['mode']}-"
    f"{importance_strategy}"
)

# --- Assemble the config dictionary ---
config = {
    "data_dir": data_dir_name,
    "logs_dir": log_dir_name,
    "planner_agent": planner_config,
    "specialized_agent": specialized_config,
    "supervisor_agent": supervisor_config
}

# --- Enhanced settings for Pydantic integration ---
pydantic_config = {
    "use_pydantic_validation": True,
    "enable_structured_outputs": True,
    "retry_on_validation_failure": True,
    "max_validation_retries": 3
}

# Add Pydantic settings to config
config["pydantic"] = pydantic_config

# --- Export for easy importing ---
__all__ = ['config', 'planner_config', 'specialized_config', 'supervisor_config', 'pydantic_config']