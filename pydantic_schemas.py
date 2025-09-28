"""
Pydantic schemas for Multi-Agent System structured outputs.
Compatible with OpenAI's structured output requirements and Response API.
"""

from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


# =================== ENUMS FOR VALIDATION ===================

# class ConfidenceLevel(int, Enum):
#     """Confidence levels for agent evaluations (1-5)"""
#     VERY_LOW = 1
#     LOW = 2
#     MEDIUM = 3
#     HIGH = 4
#     VERY_HIGH = 5


class VerbosityLevel(str, Enum):
    """Verbosity levels for GPT-5 models"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReasoningEffort(str, Enum):
    """Reasoning effort levels for reasoning models"""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# =================== AGENT CONFIGURATION SCHEMAS ===================

class AgentConfig(BaseModel):
    """Configuration for a specialized agent"""
    model_config = ConfigDict(extra='forbid')

    name: str = Field(..., description="Name of the specialized agent")
    role: str = Field(..., description="Role of the specialized agent")
    ability: str = Field(..., description="Abilities and expertise of the agent")
    profile: str = Field(..., description="Specific focus areas and evaluation criteria")


class PlannerResponse(BaseModel):
    """Schema for Planner Agent response"""
    model_config = ConfigDict(extra='forbid')

    strategic_guidance: str = Field(
        ...,
        description="High-level strategic guidance for co-investor selection",
        min_length=50,
        max_length=1000
    )
    agents: List[AgentConfig] = Field(
        ...,
        description="List of specialized agent configurations",
        min_items=2,
        max_items=8
    )

    @field_validator('agents')
    @classmethod
    def validate_unique_names(cls, v):
        """Ensure all agent names are unique"""
        names = [agent.name for agent in v]
        if len(names) != len(set(names)):
            raise ValueError("Agent names must be unique")
        return v



# =================== EVALUATION SCHEMAS ===================

class RankedCandidate(BaseModel):
    """Schema for a ranked candidate in agent evaluation"""
    model_config = ConfigDict(extra='forbid')
    firm_id: str = Field(..., description="ID of the candidate firm")
    rank: int = Field(..., ge=1, description="Rank assigned to the candidate (starting from 1)")
    alignment_score: int = Field(..., ge=1, le=10, description="An integer score (1-10) indicating how perfectly the candidate matches the ideal criteria defined in your 'evaluation_focus'. 1 is a poor match, 10 is a perfect match. Please give an objective evaluation score!")
    rationale: str = Field(
        ...,
        description="A clear, concise explanation for the rank and score, referencing specific data points from the candidate's profile.",
        min_length=20,
        max_length=500
    )


class SpecializedAgentResponse(BaseModel):
    """Schema for Specialized Agent evaluation response"""
    model_config = ConfigDict(extra='forbid')

    evaluation_focus: str = Field(
        ...,
        description="Key features and criteria used for evaluation",
        min_length=20,
        max_length=300
    )
    overall_rationale: str = Field(
        ...,
        description="General explanation of ranking methodology and strategy",
        min_length=50,
        max_length=800
    )
    ranked_candidates: List[RankedCandidate] = Field(
        ...,
        description="List of ranked candidates in order of preference",
        min_items=1,
        max_items=20
    )

    @field_validator('ranked_candidates')
    @classmethod
    def validate_ranking_sequence(cls, v):
        """Ensure rankings are sequential starting from 1"""
        expected_ranks = list(range(1, len(v) + 1))
        actual_ranks = [candidate.rank for candidate in v]
        if actual_ranks != expected_ranks:
            raise ValueError("Rankings must be sequential starting from 1")
        return v



# =================== SUPERVISOR SCHEMAS ===================

class AgentImportanceRanking(BaseModel):
    """Schema for agent importance ranking"""
    model_config = ConfigDict(extra='forbid')
    agent_name: str = Field(..., description="Name of the agent")
    rank: int = Field(..., ge=1, description="Importance rank (1 = most important)")
    rationale: str = Field(
        ...,
        description="Explanation for the importance ranking",
        min_length=20,
        max_length=400
    )


class SupervisorImportanceResponse(BaseModel):
    """Schema for Supervisor agent importance ranking response"""
    model_config = ConfigDict(extra='forbid')
    agent_importance_ranking: List[AgentImportanceRanking] = Field(
        ...,
        description="Ranked list of agents by importance",
        min_items=1,
        max_items=10
    )
    overall_ranking_rationale: str = Field(
        ...,
        description="Summary of strategic thinking behind ranking order",
        min_length=30,
        max_length=500
    )

    @field_validator('agent_importance_ranking')
    @classmethod
    def validate_importance_sequence(cls, v):
        """Ensure importance rankings are sequential starting from 1"""
        expected_ranks = list(range(1, len(v) + 1))
        actual_ranks = [agent.rank for agent in v]
        if actual_ranks != expected_ranks:
            raise ValueError("Importance rankings must be sequential starting from 1")
        return v


class AgentWeight(BaseModel):
    """Schema for agent weight assignment"""
    model_config = ConfigDict(extra='forbid')
    agent_name: str = Field(..., description="Name of the agent")
    weight: float = Field(..., ge=0.0, le=1.0, description="Numerical weight (must sum to 1.0)")
    rationale: str = Field(
        ...,
        description="Explanation for the assigned weight",
        min_length=20,
        max_length=400
    )


class SupervisorWeightsResponse(BaseModel):
    """Schema for Supervisor agent weights response"""
    model_config = ConfigDict(extra='forbid')
    agent_weights: List[AgentWeight] = Field(
        ...,
        description="List of agents with assigned weights",
        min_items=1,
        max_items=10
    )
    overall_weighting_rationale: str = Field(
        ...,
        description="Summary of strategic thinking behind weight distribution",
        min_length=30,
        max_length=500
    )

    @field_validator('agent_weights')
    @classmethod
    def validate_weights_sum_to_one(cls, v):
        """Ensure all weights sum to approximately 1.0"""
        total_weight = sum(agent.weight for agent in v)
        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Agent weights must sum to 1.0, got {total_weight}")
        return v


class SupervisorSelectionResponse(BaseModel):
    """Schema for Supervisor final selection response"""
    model_config = ConfigDict(extra='forbid')
    selected_candidates: List[str] = Field(
        ...,
        description="List of selected firm IDs in order of preference",
        min_items=1,
        max_items=20
    )
    rationale: str = Field(
        ...,
        description="Detailed explanation of selection criteria and reasoning",
        min_length=100,
        max_length=2000
    )

    @field_validator('selected_candidates')
    @classmethod
    def validate_unique_candidates(cls, v):
        """Ensure all selected candidates are unique"""
        if len(v) != len(set(v)):
            raise ValueError("Selected candidates must be unique")
        return v



# =================== UTILITY SCHEMAS ===================

class ValidationError(BaseModel):
    """Schema for validation errors"""
    model_config = ConfigDict(extra='forbid')
    field: str = Field(..., description="Field that failed validation")
    error: str = Field(..., description="Error message")
    value: Optional[Union[str, int, float, bool]] = Field(None, description="Invalid value")


class SchemaValidationResult(BaseModel):
    """Schema for validation results"""
    model_config = ConfigDict(extra='forbid')
    is_valid: bool = Field(..., description="Whether validation passed")
    errors: List[ValidationError] = Field(default=[], description="List of validation errors")
    parsed_data: Optional[dict] = Field(None, description="Successfully parsed data if valid")


# =================== SCHEMA REGISTRY ===================

class SchemaRegistry:
    """Registry for all Pydantic schemas used in the multi-agent system"""

    # Agent configuration schemas
    PLANNER_RESPONSE = PlannerResponse
    AGENT_CONFIG = AgentConfig

    # Evaluation schemas
    SPECIALIZED_AGENT_RESPONSE = SpecializedAgentResponse
    RANKED_CANDIDATE = RankedCandidate

    # Supervisor schemas
    SUPERVISOR_IMPORTANCE_RESPONSE = SupervisorImportanceResponse
    SUPERVISOR_WEIGHTS_RESPONSE = SupervisorWeightsResponse
    SUPERVISOR_SELECTION_RESPONSE = SupervisorSelectionResponse
    AGENT_IMPORTANCE_RANKING = AgentImportanceRanking
    AGENT_WEIGHT = AgentWeight

    # Utility schemas
    VALIDATION_ERROR = ValidationError
    SCHEMA_VALIDATION_RESULT = SchemaValidationResult

    @classmethod
    def get_schema(cls, schema_name: str) -> BaseModel:
        """Get a schema by name"""
        return getattr(cls, schema_name.upper(), None)

    @classmethod
    def get_all_schemas(cls) -> dict:
        """Get all available schemas"""
        schemas = {}
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and attr_name not in ['get_schema', 'get_all_schemas']:
                attr_value = getattr(cls, attr_name)
                if isinstance(attr_value, type) and issubclass(attr_value, BaseModel):
                    schemas[attr_name.lower()] = attr_value
        return schemas

    @classmethod
    def convert_to_openai_schema(cls, pydantic_model: BaseModel) -> dict:
        """Convert Pydantic model to OpenAI-compatible JSON schema"""
        schema = pydantic_model.model_json_schema()

        # Remove unsupported fields for OpenAI compatibility
        unsupported_fields = ['title', 'examples']
        for field in unsupported_fields:
            schema.pop(field, None)

        # Recursively remove unsupported fields from properties
        def clean_schema(obj):
            if isinstance(obj, dict):
                for key in list(obj.keys()):
                    if key in unsupported_fields:
                        obj.pop(key)
                    else:
                        clean_schema(obj[key])
            elif isinstance(obj, list):
                for item in obj:
                    clean_schema(item)

        clean_schema(schema)

        return {
            "type": "json_schema",
            "json_schema": {
                "name": pydantic_model.__name__.lower(),
                "strict": True,
                "schema": schema
            }
        }


# =================== HELPER FUNCTIONS ===================

def validate_response_with_pydantic(response_text: str, schema_class: BaseModel) -> SchemaValidationResult:
    """
    Validate a response string against a Pydantic schema.

    Args:
        response_text: Raw response text from the model
        schema_class: Pydantic model class to validate against

    Returns:
        SchemaValidationResult: Validation result with errors if any
    """
    import json
    from pydantic import ValidationError as PydanticValidationError

    try:
        # First try to parse as JSON
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            return SchemaValidationResult(
                is_valid=False,
                errors=[ValidationError(field="json_parse", error=f"Invalid JSON: {str(e)}", value=response_text[:100])]
            )

        # Then validate with Pydantic
        validated_data = schema_class.model_validate(data)
        return SchemaValidationResult(
            is_valid=True,
            errors=[],
            parsed_data=validated_data.model_dump()
        )

    except PydanticValidationError as e:
        errors = []
        for error in e.errors():
            field_path = " -> ".join(str(x) for x in error['loc']) if error['loc'] else "root"
            errors.append(ValidationError(
                field=field_path,
                error=error['msg'],
                value=str(error.get('input'))[:100] if error.get('input') is not None else None
            ))

        return SchemaValidationResult(
            is_valid=False,
            errors=errors
        )

    except Exception as e:
        return SchemaValidationResult(
            is_valid=False,
            errors=[ValidationError(field="general", error=str(e))]
        )


def create_openai_response_format(pydantic_model: BaseModel) -> dict:
    """
    Create OpenAI response format from Pydantic model.
    Compatible with both Chat Completions and Response API.

    Args:
        pydantic_model: Pydantic model class

    Returns:
        dict: OpenAI-compatible response format
    """
    return SchemaRegistry.convert_to_openai_schema(pydantic_model)