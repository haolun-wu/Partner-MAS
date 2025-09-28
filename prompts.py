import sys
import os
import json

class PromptExtractor:
    """A class to extract prompts from markdown files and schemas from JSON files."""
    
    def __init__(self):
        """Initialize with the path to the prompts and schemas files."""
        self.file_path = "./prompts/prompts.md"
        self.schema_path = "./prompts/schemas.json"
    
    def extract_prompts(self) -> dict:
        """
        Extract prompts from the markdown file.
        
        Returns:
            dict: Dictionary mapping prompt names to their content
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split by separator
            sections = content.split('=' * 10)
            
            prompts = {}
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                
                lines = section.split('\n')
                name = None
                prompt_started = False
                prompt_lines = []
                
                for line in lines:
                    line = line.strip()
                    
                    # Extract name from "# **Name: xxx**" or "# **Name**" format
                    if line.startswith('# **Name'):
                        if ':' in line:
                            # Format: # **Name: kks_system_prompt**
                            name = line.split(':', 1)[1].strip().rstrip('*').strip()
                        else:
                            # Format: # **Name**
                            name = line.replace('# **', '').replace('**', '').strip()
                    
                    # Check if we've reached the prompt section
                    elif line == '## Prompt':
                        prompt_started = True
                        continue
                    
                    # Collect prompt content
                    elif prompt_started and line:
                        prompt_lines.append(line)
                
                # Store the extracted prompt if we have both name and content
                if name and prompt_lines:
                    prompts[name] = '\n'.join(prompt_lines)
            
            return prompts
            
        except FileNotFoundError:
            print(f"Error: File {self.file_path} not found.")
            return {}
        except Exception as e:
            print(f"Error reading file: {e}")
            return {}
    
    def extract_schemas(self) -> dict:
        """
        Extract schemas from the JSON file.
        
        Returns:
            dict: Dictionary mapping schema names to their schema objects
        """
        try:
            with open(self.schema_path, 'r', encoding='utf-8') as file:
                schemas = json.load(file)
            
            # The JSON file already contains the desired format: {name: schema_dict}
            return schemas
            
        except FileNotFoundError:
            print(f"Error: File {self.schema_path} not found.")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {self.schema_path}: {e}")
            return {}
        except Exception as e:
            print(f"Error reading schema file: {e}")
            return {}


def handle_prompt_extraction(file_path: str = "./prompts/prompts.md") -> dict:
    """
    Convenience function to extract prompts from a markdown file.

    Args:
        file_path: Path to the prompts markdown file

    Returns:
        dict: Dictionary mapping prompt names to their content
    """
    extractor = PromptExtractor()
    extractor.file_path = file_path
    return extractor.extract_prompts()


def handle_schema_extraction(schema_path: str = "./prompts/schemas.json") -> dict:
    """
    Convenience function to extract schemas from a JSON file.

    Args:
        schema_path: Path to the schemas JSON file

    Returns:
        dict: Dictionary mapping schema names to their schema objects
    """
    extractor = PromptExtractor()
    extractor.schema_path = schema_path
    return extractor.extract_schemas()


# Example usage
if __name__ == "__main__":
    # Extract prompts
    prompts = handle_prompt_extraction()
    print("Extracted prompts:")
    for name, prompt in prompts.items():
        print(f"\n=== {name} ===")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)

    # Extract schemas
    schemas = handle_schema_extraction()
    print("\n\nExtracted schemas:")
    for name, schema in schemas.items():
        print(f"\n=== {name} ===")
        print(f"Type: {schema.get('type', 'unknown')}")
        if 'properties' in schema:
            print(f"Properties: {list(schema['properties'].keys())}")
        elif 'json_schema' in schema and 'properties' in schema['json_schema'].get('schema', {}):
            print(f"Properties: {list(schema['json_schema']['schema']['properties'].keys())}")

    print(f"\n\nTotal prompts extracted: {len(prompts)}")
    print(f"Total schemas extracted: {len(schemas)}")


# =================== CONVENIENCE FUNCTIONS FOR MULTI-AGENT SYSTEM ===================

def get_all_prompts() -> dict:
    """
    Get all available prompts for the multi-agent system.

    Returns:
        dict: All extracted prompts with their names as keys
    """
    return handle_prompt_extraction()

def get_prompt(prompt_name: str) -> str:
    """
    Get a specific prompt by name.

    Args:
        prompt_name: Name of the prompt to retrieve

    Returns:
        str: The prompt content, or empty string if not found
    """
    prompts = get_all_prompts()
    return prompts.get(prompt_name, "")

def get_all_schemas() -> dict:
    """
    Get all available schemas for the multi-agent system.

    Returns:
        dict: All extracted schemas with their names as keys
    """
    return handle_schema_extraction()

def get_schema(schema_name: str) -> dict:
    """
    Get a specific schema by name.

    Args:
        schema_name: Name of the schema to retrieve

    Returns:
        dict: The schema object, or empty dict if not found
    """
    schemas = get_all_schemas()
    return schemas.get(schema_name, {})

# Agent-specific prompt getters
class Prompts:
    """Main class for accessing multi-agent system prompts"""

    @staticmethod
    def planner_business() -> str:
        """Get the business-guided planner agent prompt with domain knowledge hints"""
        return get_prompt("PLANNER_AGENT_PROMPT_BUSINESS")

    @staticmethod
    def planner_generic() -> str:
        """Get the generic planner agent prompt without domain hints"""
        return get_prompt("PLANNER_AGENT_PROMPT_GENERIC")

    @staticmethod
    def specialized_agent() -> str:
        """Get the specialized agent evaluation prompt"""
        return get_prompt("SPECIALIZED_AGENT_PROMPT")

    @staticmethod
    def supervisor_rank_importance_business() -> str:
        """Get the business-guided supervisor importance ranking prompt (Mode: Importance - Step 1)"""
        return get_prompt("SUPERVISOR_RANK_IMPORTANCE_BUSINESS_PROMPT")

    @staticmethod
    def supervisor_rank_importance_generic() -> str:
        """Get the generic supervisor importance ranking prompt (Mode: Importance - Step 1)"""
        return get_prompt("SUPERVISOR_RANK_IMPORTANCE_GENERIC_PROMPT")

    @staticmethod
    def supervisor_select_by_importance() -> str:
        """Get the supervisor selection by importance prompt (Mode: Importance - Step 2)"""
        return get_prompt("SUPERVISOR_SELECT_BY_IMPORTANCE_PROMPT")

    @staticmethod
    def supervisor_determine_weights() -> str:
        """Get the supervisor weight determination prompt (Mode: Weight - Step 1)"""
        return get_prompt("SUPERVISOR_DETERMINE_WEIGHTS_PROMPT")

    @staticmethod
    def supervisor_select_by_weight() -> str:
        """Get the supervisor selection by weight prompt (Mode: Weight - Step 2)"""
        return get_prompt("SUPERVISOR_SELECT_BY_WEIGHT_PROMPT")

    @staticmethod
    def supervisor_majority_vote() -> str:
        """Get the supervisor majority vote selection prompt"""
        return get_prompt("SUPERVISOR_SELECT_BY_MAJORITY_VOTE_PROMPT")

# Schema-specific getters (Legacy JSON schemas)
class Schemas:
    """Main class for accessing multi-agent system JSON schemas (Legacy)"""

    @staticmethod
    def planner_response() -> dict:
        """Get the planner response schema"""
        return get_schema("planner_response_schema")

    @staticmethod
    def specialized_agent_response() -> dict:
        """Get the specialized agent response schema"""
        return get_schema("specialized_agent_response_schema")

    @staticmethod
    def supervisor_selection() -> dict:
        """Get the supervisor selection schema"""
        return get_schema("supervisor_selection_schema")

    @staticmethod
    def agent_importance_ranking() -> dict:
        """Get the agent importance ranking schema"""
        return get_schema("agent_importance_ranking_schema")

    @staticmethod
    def agent_weights() -> dict:
        """Get the agent weights schema"""
        return get_schema("agent_weights_schema")


# =================== PYDANTIC SCHEMA INTEGRATION ===================

try:
    from pydantic_schemas import SchemaRegistry, PlannerResponse, SpecializedAgentResponse, SupervisorSelectionResponse, SupervisorImportanceResponse, SupervisorWeightsResponse
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("Warning: pydantic_schemas not available. Pydantic functionality will be limited.")


class PydanticSchemas:
    """Main class for accessing Pydantic schemas for structured outputs"""

    if PYDANTIC_AVAILABLE:
        # Pydantic model classes
        PLANNER_RESPONSE = PlannerResponse
        SPECIALIZED_AGENT_RESPONSE = SpecializedAgentResponse
        SUPERVISOR_SELECTION_RESPONSE = SupervisorSelectionResponse
        SUPERVISOR_IMPORTANCE_RESPONSE = SupervisorImportanceResponse
        SUPERVISOR_WEIGHTS_RESPONSE = SupervisorWeightsResponse

        @classmethod
        def planner_response(cls):
            """Get the Pydantic planner response model"""
            return cls.PLANNER_RESPONSE

        @classmethod
        def specialized_agent_response(cls):
            """Get the Pydantic specialized agent response model"""
            return cls.SPECIALIZED_AGENT_RESPONSE

        @classmethod
        def supervisor_selection_response(cls):
            """Get the Pydantic supervisor selection response model"""
            return cls.SUPERVISOR_SELECTION_RESPONSE

        @classmethod
        def supervisor_importance_response(cls):
            """Get the Pydantic supervisor importance response model"""
            return cls.SUPERVISOR_IMPORTANCE_RESPONSE

        @classmethod
        def supervisor_weights_response(cls):
            """Get the Pydantic supervisor weights response model"""
            return cls.SUPERVISOR_WEIGHTS_RESPONSE

        @classmethod
        def get_openai_schema(cls, pydantic_model):
            """Convert Pydantic model to OpenAI-compatible schema"""
            return SchemaRegistry.convert_to_openai_schema(pydantic_model)

        @classmethod
        def get_all_models(cls):
            """Get all available Pydantic models"""
            return {
                'planner_response': cls.PLANNER_RESPONSE,
                'specialized_agent_response': cls.SPECIALIZED_AGENT_RESPONSE,
                'supervisor_selection_response': cls.SUPERVISOR_SELECTION_RESPONSE,
                'supervisor_importance_response': cls.SUPERVISOR_IMPORTANCE_RESPONSE,
                'supervisor_weights_response': cls.SUPERVISOR_WEIGHTS_RESPONSE
            }

    else:
        @classmethod
        def _not_available(cls, *_args, **_kwargs):
            raise ImportError("Pydantic schemas not available. Install pydantic_schemas module.")

        planner_response = _not_available
        specialized_agent_response = _not_available
        supervisor_selection_response = _not_available
        supervisor_importance_response = _not_available
        supervisor_weights_response = _not_available
        get_openai_schema = _not_available
        get_all_models = _not_available


# =================== UNIFIED SCHEMA INTERFACE ===================

class UnifiedSchemas:
    """Unified interface for both JSON and Pydantic schemas"""

    @staticmethod
    def get_schema_format(schema_type: str, prefer_pydantic: bool = True):
        """
        Get schema in preferred format (Pydantic or JSON).

        Args:
            schema_type: Type of schema ('planner', 'specialized_agent', 'supervisor_selection', etc.)
            prefer_pydantic: Whether to prefer Pydantic models over JSON schemas

        Returns:
            Pydantic model class or dict: Schema in requested format
        """
        if prefer_pydantic and PYDANTIC_AVAILABLE:
            pydantic_mapping = {
                'planner': PydanticSchemas.planner_response(),
                'planner_response': PydanticSchemas.planner_response(),
                'specialized_agent': PydanticSchemas.specialized_agent_response(),
                'specialized_agent_response': PydanticSchemas.specialized_agent_response(),
                'supervisor_selection': PydanticSchemas.supervisor_selection_response(),
                'supervisor_selection_response': PydanticSchemas.supervisor_selection_response(),
                'supervisor_importance': PydanticSchemas.supervisor_importance_response(),
                'supervisor_importance_response': PydanticSchemas.supervisor_importance_response(),
                'supervisor_weights': PydanticSchemas.supervisor_weights_response(),
                'supervisor_weights_response': PydanticSchemas.supervisor_weights_response()
            }
            return pydantic_mapping.get(schema_type)
        else:
            # Fallback to JSON schemas
            json_mapping = {
                'planner': Schemas.planner_response(),
                'planner_response': Schemas.planner_response(),
                'specialized_agent': Schemas.specialized_agent_response(),
                'specialized_agent_response': Schemas.specialized_agent_response(),
                'supervisor_selection': Schemas.supervisor_selection(),
                'supervisor_selection_response': Schemas.supervisor_selection(),
                'supervisor_importance': Schemas.agent_importance_ranking(),
                'supervisor_importance_response': Schemas.agent_importance_ranking(),
                'supervisor_weights': Schemas.agent_weights(),
                'supervisor_weights_response': Schemas.agent_weights()
            }
            return json_mapping.get(schema_type)

    @staticmethod
    def create_openai_format(schema_type: str, prefer_pydantic: bool = True):
        """
        Create OpenAI-compatible response format.

        Args:
            schema_type: Type of schema
            prefer_pydantic: Whether to prefer Pydantic models

        Returns:
            dict: OpenAI response format
        """
        schema = UnifiedSchemas.get_schema_format(schema_type, prefer_pydantic)

        if PYDANTIC_AVAILABLE and prefer_pydantic and hasattr(schema, 'model_json_schema'):
            # It's a Pydantic model
            return PydanticSchemas.get_openai_schema(schema)
        else:
            # It's a JSON schema
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": f"{schema_type}_schema",
                    "strict": True,
                    "schema": schema
                }
            }


# =================== CONVENIENCE FUNCTIONS FOR AGENTS ===================

def get_agent_schema(agent_type: str, response_type: str = "response", use_pydantic: bool = True):
    """
    Convenience function to get the appropriate schema for an agent.

    Args:
        agent_type: Type of agent ('planner', 'specialized', 'supervisor')
        response_type: Type of response ('response', 'importance', 'weights', 'selection')
        use_pydantic: Whether to use Pydantic models (recommended)

    Returns:
        Schema in appropriate format
    """
    schema_key = f"{agent_type}_{response_type}" if response_type != "response" else agent_type
    return UnifiedSchemas.get_schema_format(schema_key, use_pydantic)


def create_structured_prompt_with_schema(base_prompt: str, schema_type: str, use_pydantic: bool = True) -> tuple:
    """
    Create a structured prompt with appropriate schema for OpenAI.

    Args:
        base_prompt: The base prompt text
        schema_type: Type of schema to use
        use_pydantic: Whether to use Pydantic models

    Returns:
        tuple: (prompt, openai_schema_format)
    """
    schema_format = UnifiedSchemas.create_openai_format(schema_type, use_pydantic)

    # Add schema instruction to prompt if not already present
    if "JSON" not in base_prompt and "json" not in base_prompt:
        enhanced_prompt = base_prompt + "\n\nPlease provide your response as a valid JSON object following the specified schema."
    else:
        enhanced_prompt = base_prompt

    return enhanced_prompt, schema_format