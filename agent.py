"""
Multi-Agent System for Co-Investor Selection
Migrated and enhanced from Archive/agent.py with Pydantic integration
"""

import json
import math
import random
from typing import Dict, Any, List, Optional
from utilities import openai_client
from prompts import Prompts, PydanticSchemas, UnifiedSchemas
from config import config
from project_types import llm_response_format


class PlannerAgent:
    """
    Planner Agent - Designs the multi-agent system architecture for each investment case.
    Enhanced with Pydantic validation and structured outputs.
    """

    def __init__(self, name: str, role: str, ability: str, model: str, temperature: float,
                 prompt_strategy: str = 'default', reasoning_effort: str = 'medium', verbosity: str = 'medium', api_secret_path: str = None):
        self.name = name
        self.role = role
        self.ability = ability
        self.model = model
        self.temperature = temperature
        self.prompt_strategy = prompt_strategy
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity

        # Initialize OpenAI client if API path provided
        self.client = None
        if api_secret_path:
            self.client = openai_client(api_secret_path)

    def plan_agents(self, lead_profile: str, target_profile: str, candidate_profiles: Dict) -> Dict:
        """
        Plan and design specialized agents for the current investment case.

        Args:
            lead_profile: Lead investor profile
            target_profile: Target company profile
            candidate_profiles: Dictionary of candidate co-investor profiles

        Returns:
            Dict containing agent configurations and strategic guidance
        """
        # Sample candidates for context (same logic as Archive)
        if candidate_profiles:
            candidate_keys = list(candidate_profiles.keys())
            sample_size = min(2, len(candidate_keys))
            random_keys = random.sample(candidate_keys, sample_size)
            sample_candidates_data = {k: candidate_profiles[k] for k in random_keys}
        else:
            sample_candidates_data = {}

        # Choose the prompt based on the strategy
        if self.prompt_strategy == 'generic':
            prompt_template = Prompts.planner_generic()
            print("Using GENERIC prompt strategy for Planner Agent.")
        else:
            prompt_template = Prompts.planner_business()
            print("Using BUSINESS prompt strategy for Planner Agent.")

        # Format the prompt
        prompt = prompt_template.format(
            name=self.name,
            role=self.role,
            ability=self.ability,
            lead_profile=lead_profile,
            target_profile=target_profile,
            sample_candidates=json.dumps(sample_candidates_data, indent=2)
        )

        # Use Pydantic validation if available and configured
        use_pydantic = config.get("pydantic", {}).get("use_pydantic_validation", True)
        max_retries = config.get("pydantic", {}).get("max_validation_retries", 3)

        if self.client and use_pydantic:
            try:
                pydantic_model = PydanticSchemas.planner_response()

                # Use flexible completion with retry logic
                response = self.client.flexible_completion(
                    user_prompt=prompt,
                    pydantic_model=pydantic_model,
                    model=self.model,
                    max_retries=max_retries,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                    verbosity=self.verbosity
                )

                if hasattr(response, 'validated_data') and response.validated_data:
                    validated_data = response.validated_data
                    return {
                        "agents": validated_data['agents'],
                        "strategic_guidance": validated_data['strategic_guidance'],
                        "raw_response": response.text,
                        "metrics": {
                            "prompt_tokens": response.usage.input,
                            "completion_tokens": response.usage.output + response.usage.reasoning,
                            "total_tokens": response.usage.total,
                            "elapsed_time": 0.0  # Not tracked in new client
                        },
                        "validation_successful": True
                    }

            except Exception as e:
                print(f"⚠️ Pydantic validation failed, falling back to mock mode: {e}")
                import traceback
                traceback.print_exc()

        # Fallback: Use mock response with proper structure for testing
        print("🧪 Using mock response mode for testing (API structured outputs not working)")
        return self._generate_mock_planner_response()

    def _generate_mock_planner_response(self) -> Dict:
        """Generate a mock planner response for testing when API is not available"""
        mock_agents = [
            {
                "name": "Market Analyst",
                "role": "Senior Partner with deep market analysis expertise",
                "ability": "Evaluates market positioning, competitive landscape, and business model viability",
                "profile": "Focuses on market trends, competitive dynamics, and strategic positioning to assess co-investor fit based on market understanding and industry expertise."
            },
            {
                "name": "Financial Evaluator",
                "role": "VP with strong financial analysis background",
                "ability": "Analyzes financial metrics, performance indicators, and investment thesis alignment",
                "profile": "Examines financial health, growth metrics, and investment compatibility to determine co-investor suitability based on financial criteria and investment approach."
            },
            {
                "name": "Strategic Advisor",
                "role": "General Partner with portfolio management experience",
                "ability": "Assesses strategic value-add potential and partnership synergies",
                "profile": "Evaluates strategic alignment, value-add capabilities, and long-term partnership potential to identify co-investors who can contribute beyond capital."
            }
        ]

        return {
            "agents": mock_agents,
            "strategic_guidance": "Mock response: Focus on market expertise, financial alignment, and strategic value-add when evaluating co-investors. Consider complementary skills and shared investment thesis.",
            "raw_response": "Mock planner response generated for testing purposes",
            "metrics": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "elapsed_time": 0.0
            },
            "validation_successful": False
        }

    def _legacy_plan_agents(self, prompt: str, max_retries: int = 2) -> Dict:
        """Legacy planning method from Archive (fallback)"""
        # This would need a legacy call_model function - for now, return structured fallback
        print("⚠️ Using legacy fallback - implement call_model function for full compatibility")

        return {
            "agents": [],
            "strategic_guidance": "Legacy fallback - please configure API client for full functionality",
            "raw_response": "",
            "metrics": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "elapsed_time": 0.0
            },
            "validation_successful": False
        }


class SpecializedAgent:
    """
    Specialized Agent - Evaluates candidates from a specific perspective.
    Enhanced with Pydantic validation and structured outputs.
    """

    def __init__(self, name: str, role: str, ability: str, profile: str,
                 model: str, temperature: float, reasoning_effort: str = 'medium', verbosity: str = 'medium', api_secret_path: str = None):
        self.name = name
        self.role = role
        self.ability = ability
        self.profile = profile
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity

        # Initialize OpenAI client if API path provided
        self.client = None
        if api_secret_path:
            self.client = openai_client(api_secret_path)

    def evaluate_candidates_and_rank(self, candidates_data: Dict, target_profile: str) -> Dict:
        """
        Evaluate and rank candidates from this agent's specialized perspective.

        Args:
            candidates_data: Dictionary of candidate profiles
            target_profile: Target company profile

        Returns:
            Dict containing evaluation results and rankings
        """
        # Calculate dynamic top-k (same logic as Archive)
        total_candidates = len(candidates_data)
        dynamic_top_k = max(1, math.ceil(total_candidates / 3))
        dynamic_top_k = min(dynamic_top_k, total_candidates)

        # Get specialized agent prompt
        prompt_template = Prompts.specialized_agent()
        prompt = prompt_template.format(
            name=self.name,
            role=self.role,
            ability=self.ability,
            profile=self.profile,
            target_profile=target_profile,
            candidates_data=json.dumps(candidates_data, indent=2),
            dynamic_top_k=dynamic_top_k,
            total_candidates=total_candidates
        )

        # Use Pydantic validation if available and configured
        use_pydantic = config.get("pydantic", {}).get("use_pydantic_validation", True)

        if self.client and use_pydantic:
            try:
                pydantic_model = PydanticSchemas.specialized_agent_response()

                response = self.client.structured_completion(
                    user_prompt=prompt,
                    pydantic_model=pydantic_model,
                    model=self.model,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                    verbosity=self.verbosity
                )

                if hasattr(response, 'validated_data') and response.validated_data:
                    validated_data = response.validated_data
                    validated_data['agent_profile'] = self.profile

                    return {
                        "parsed_output": validated_data,
                        "raw_response": response.text,
                        "metrics": {
                            "prompt_tokens": response.usage.input,
                            "completion_tokens": response.usage.output + response.usage.reasoning,
                            "total_tokens": response.usage.total,
                            "elapsed_time": 0.0
                        },
                        "validation_successful": True
                    }

            except Exception as e:
                print(f"❌ WARNING: {self.name} Pydantic validation failed: {e}")

        # Fallback response (similar to Archive fallback)
        print(f"❌ WARNING: {self.name} failed to evaluate candidates. Using fallback response.")
        fallback_output = {
            "evaluation_focus": "Error during evaluation - API client not configured or validation failed",
            "overall_rationale": "The agent could not complete evaluation due to configuration issues",
            "ranked_candidates": [],
            "parsing_error": True,
            "agent_profile": self.profile
        }

        return {
            "parsed_output": fallback_output,
            "raw_response": "",
            "metrics": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "elapsed_time": 0.0
            },
            "validation_successful": False
        }


class SupervisorAgent:
    """
    Supervisor Agent - Makes final co-investor selections based on specialized agent inputs.
    Enhanced with multiple decision modes and Pydantic validation.
    """

    def __init__(self, name: str, role: str, ability: str, profile: str,
                 model: str, temperature: float, mode: str = 'importance',
                 importance_strategy: str = 'business', reasoning_effort: str = 'medium', verbosity: str = 'medium', api_secret_path: str = None):
        self.name = name
        self.role = role
        self.ability = ability
        self.profile = profile
        self.model = model
        self.temperature = temperature
        self.mode = mode
        self.importance_strategy = importance_strategy
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity

        # Initialize OpenAI client if API path provided
        self.client = None
        if api_secret_path:
            self.client = openai_client(api_secret_path)

    def select_final_shortlist(self, supervisor_input: Dict, top_k: int) -> Dict:
        """
        Make final candidate selection based on specialized agent evaluations.

        Args:
            supervisor_input: Input containing agent evaluations and context
            top_k: Number of candidates to select

        Returns:
            Dict containing final selection and rationale
        """
        print(f"\n\033[1m👨‍💼 Supervisor operating in '{self.mode}' mode...\033[0m")

        if self.mode == 'importance':
            return self._select_by_importance(supervisor_input, top_k)
        elif self.mode == 'weight':
            return self._select_by_weight(supervisor_input, top_k)
        elif self.mode == 'majority_vote':
            return self._select_by_majority_vote(supervisor_input, top_k)
        else:
            print(f"⚠️ Unknown supervisor mode '{self.mode}'. Defaulting to 'importance' mode.")
            return self._select_by_importance(supervisor_input, top_k)

    def _select_by_importance(self, supervisor_input: Dict, top_k: int) -> Dict:
        """Selection using importance-based ranking of agents"""
        print("\n\033[1m👨‍💼 Supervisor is first ranking the importance of each specialized agent...\033[0m")

        # Step 1: Rank agent importance
        agent_importance_result = self._rank_agent_importance(
            supervisor_input['target_profile'],
            supervisor_input['all_agent_evaluations']
        )

        agent_importance_parsed = agent_importance_result.get("parsed_output")

        if not isinstance(agent_importance_parsed, dict) or 'agent_importance_ranking' not in agent_importance_parsed:
            print("⚠️ Supervisor failed to parse the agent importance ranking. Proceeding with equal weighting.")
            agent_importance_section = "Agent importance could not be determined. Evaluate all agent inputs equally."
        else:
            print("\n\033[1m✅ Supervisor has ranked the agents by importance:\033[0m")
            for rank_info in agent_importance_parsed.get('agent_importance_ranking', []):
                print(f"  - \033[1mRank {rank_info.get('rank')}:\033[0m {rank_info.get('agent_name')} (Rationale: {rank_info.get('rationale')})")
            agent_importance_section = f"# Agent Importance Ranking (Your Strategic Assessment):\n{json.dumps(agent_importance_parsed, indent=2)}"

        planner_guidance = supervisor_input.get('planner_strategic_guidance', 'No strategic guidance was provided.')
        print(f"\n\033[1m📝 Supervisor is using this Strategic Guidance from the Planner:\033[0m\n  - {planner_guidance}")

        # Step 2: Make final selection
        print("\n\033[1m👨‍💼 Supervisor is now making the final selection using the agent importance ranking...\033[0m")

        final_selection_result = self._make_importance_based_selection(
            supervisor_input, top_k, agent_importance_section, planner_guidance
        )

        return {
            "agent_importance_ranking": agent_importance_result,
            "final_selection": final_selection_result
        }

    def _select_by_weight(self, supervisor_input: Dict, top_k: int) -> Dict:
        """Selection using numerical weight assignment"""
        print("\n\033[1m👨‍💼 Supervisor is determining numerical weights for each specialized agent...\033[0m")

        agent_weights_result = self._determine_agent_weights(
            supervisor_input['target_profile'],
            supervisor_input['all_agent_evaluations']
        )

        agent_weights_parsed = agent_weights_result.get("parsed_output")

        if not isinstance(agent_weights_parsed, dict) or 'agent_weights' not in agent_weights_parsed:
            print("⚠️ Supervisor failed to determine agent weights. Proceeding without weighted analysis.")
            agent_weights_section = "Agent weights could not be determined. Evaluate all agent inputs equally."
        else:
            print("\n\033[1m✅ Supervisor has assigned the following weights:\033[0m")
            for weight_info in agent_weights_parsed.get('agent_weights', []):
                print(f"  - \033[1mWeight {weight_info.get('weight'):.2f}:\033[0m {weight_info.get('agent_name')} (Rationale: {weight_info.get('rationale')})")
            agent_weights_section = f"# Agent Weights (Your Strategic Assessment):\n{json.dumps(agent_weights_parsed, indent=2)}"

        print("\n\033[1m👨‍💼 Supervisor is now making the final selection using the determined weights...\033[0m")

        final_selection_result = self._make_weight_based_selection(
            supervisor_input, top_k, agent_weights_section
        )

        return {
            "agent_weight_determination": agent_weights_result,
            "final_selection": final_selection_result
        }

    def _select_by_majority_vote(self, supervisor_input: Dict, top_k: int) -> Dict:
        """Selection using majority vote approach"""
        final_selection_result = self._make_majority_vote_selection(supervisor_input, top_k)

        return {
            "final_selection": final_selection_result
        }

    def _rank_agent_importance(self, target_profile: str, all_agent_evaluations: Dict) -> Dict:
        """Rank agents by importance for this specific investment"""
        agent_profiles = {
            name: evaluation.get('agent_profile', 'No profile found.')
            for name, evaluation in all_agent_evaluations.items()
        }

        if self.client:
            try:
                if self.importance_strategy == 'generic':
                    prompt_template = Prompts.supervisor_rank_importance_generic()
                else:
                    prompt_template = Prompts.supervisor_rank_importance_business()

                prompt = prompt_template.format(
                    name=self.name,
                    role=self.role,
                    target_profile=target_profile,
                    agent_profiles=json.dumps(agent_profiles, indent=2)
                )

                pydantic_model = PydanticSchemas.supervisor_importance_response()
                response = self.client.structured_completion(
                    user_prompt=prompt,
                    pydantic_model=pydantic_model,
                    model=self.model,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                    verbosity=self.verbosity
                )

                if hasattr(response, 'validated_data') and response.validated_data:
                    return {
                        "parsed_output": response.validated_data,
                        "raw_response": response.text,
                        "metrics": {
                            "prompt_tokens": response.usage.input,
                            "completion_tokens": response.usage.output + response.usage.reasoning,
                            "total_tokens": response.usage.total,
                            "elapsed_time": 0.0
                        }
                    }

            except Exception as e:
                print(f"⚠️ Agent importance ranking failed: {e}")

        return {
            "parsed_output": {},
            "raw_response": "",
            "metrics": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "elapsed_time": 0.0}
        }

    def _determine_agent_weights(self, target_profile: str, all_agent_evaluations: Dict) -> Dict:
        """Determine numerical weights for agents"""
        agent_profiles = {
            name: evaluation.get('agent_profile', 'No profile found.')
            for name, evaluation in all_agent_evaluations.items()
        }

        if self.client:
            try:
                prompt_template = Prompts.supervisor_determine_weights()
                prompt = prompt_template.format(
                    name=self.name,
                    role=self.role,
                    target_profile=target_profile,
                    agent_profiles=json.dumps(agent_profiles, indent=2)
                )

                pydantic_model = PydanticSchemas.supervisor_weights_response()
                response = self.client.structured_completion(
                    user_prompt=prompt,
                    pydantic_model=pydantic_model,
                    model=self.model,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                    verbosity=self.verbosity
                )

                if hasattr(response, 'validated_data') and response.validated_data:
                    return {
                        "parsed_output": response.validated_data,
                        "raw_response": response.text,
                        "metrics": {
                            "prompt_tokens": response.usage.input,
                            "completion_tokens": response.usage.output + response.usage.reasoning,
                            "total_tokens": response.usage.total,
                            "elapsed_time": 0.0
                        }
                    }

            except Exception as e:
                print(f"⚠️ Agent weight determination failed: {e}")

        return {
            "parsed_output": {},
            "raw_response": "",
            "metrics": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "elapsed_time": 0.0}
        }

    def _make_importance_based_selection(self, supervisor_input: Dict, top_k: int,
                                       agent_importance_section: str, planner_guidance: str) -> Dict:
        """Make final selection using importance ranking"""
        if self.client:
            try:
                prompt_template = Prompts.supervisor_select_by_importance()
                prompt = prompt_template.format(
                    name=self.name, role=self.role, ability=self.ability, profile=self.profile, top_k=top_k,
                    agent_importance_section=agent_importance_section,
                    target_profile=supervisor_input['target_profile'],
                    agent_evaluations=json.dumps(supervisor_input['all_agent_evaluations'], indent=2),
                    planner_strategic_guidance=planner_guidance
                )

                pydantic_model = PydanticSchemas.supervisor_selection_response()
                response = self.client.structured_completion(
                    user_prompt=prompt,
                    pydantic_model=pydantic_model,
                    model=self.model,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                    verbosity=self.verbosity
                )

                if hasattr(response, 'validated_data') and response.validated_data:
                    return {
                        "parsed_output": response.validated_data,
                        "raw_response": response.text,
                        "metrics": {
                            "prompt_tokens": response.usage.input,
                            "completion_tokens": response.usage.output + response.usage.reasoning,
                            "total_tokens": response.usage.total,
                            "elapsed_time": 0.0
                        }
                    }

            except Exception as e:
                print(f"⚠️ Importance-based selection failed: {e}")

        return self._get_fallback_selection(top_k)

    def _make_weight_based_selection(self, supervisor_input: Dict, top_k: int,
                                   agent_weights_section: str) -> Dict:
        """Make final selection using numerical weights"""
        if self.client:
            try:
                prompt_template = Prompts.supervisor_select_by_weight()
                prompt = prompt_template.format(
                    name=self.name, role=self.role, ability=self.ability, profile=self.profile, top_k=top_k,
                    agent_weights_section=agent_weights_section,
                    target_profile=supervisor_input['target_profile'],
                    agent_evaluations=json.dumps(supervisor_input['all_agent_evaluations'], indent=2),
                    candidates_data=json.dumps(supervisor_input['candidates_data'], indent=2)
                )

                pydantic_model = PydanticSchemas.supervisor_selection_response()
                response = self.client.structured_completion(
                    user_prompt=prompt,
                    pydantic_model=pydantic_model,
                    model=self.model,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                    verbosity=self.verbosity
                )

                if hasattr(response, 'validated_data') and response.validated_data:
                    return {
                        "parsed_output": response.validated_data,
                        "raw_response": response.text,
                        "metrics": {
                            "prompt_tokens": response.usage.input,
                            "completion_tokens": response.usage.output + response.usage.reasoning,
                            "total_tokens": response.usage.total,
                            "elapsed_time": 0.0
                        }
                    }

            except Exception as e:
                print(f"⚠️ Weight-based selection failed: {e}")

        return self._get_fallback_selection(top_k)

    def _make_majority_vote_selection(self, supervisor_input: Dict, top_k: int) -> Dict:
        """Make final selection using majority vote"""
        if self.client:
            try:
                prompt_template = Prompts.supervisor_majority_vote()
                prompt = prompt_template.format(
                    name=self.name, role=self.role, ability=self.ability, profile=self.profile, top_k=top_k,
                    target_profile=supervisor_input['target_profile'],
                    agent_evaluations=json.dumps(supervisor_input['all_agent_evaluations'], indent=2),
                    candidates_data=json.dumps(supervisor_input['candidates_data'], indent=2)
                )

                pydantic_model = PydanticSchemas.supervisor_selection_response()
                response = self.client.structured_completion(
                    user_prompt=prompt,
                    pydantic_model=pydantic_model,
                    model=self.model,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                    verbosity=self.verbosity
                )

                if hasattr(response, 'validated_data') and response.validated_data:
                    return {
                        "parsed_output": response.validated_data,
                        "raw_response": response.text,
                        "metrics": {
                            "prompt_tokens": response.usage.input,
                            "completion_tokens": response.usage.output + response.usage.reasoning,
                            "total_tokens": response.usage.total,
                            "elapsed_time": 0.0
                        }
                    }

            except Exception as e:
                print(f"⚠️ Majority vote selection failed: {e}")

        return self._get_fallback_selection(top_k)

    def _get_fallback_selection(self, top_k: int) -> Dict:
        """Fallback selection when API calls fail"""
        return {
            "parsed_output": {
                "selected_candidates": [],
                "rationale": "Selection failed due to API configuration issues. Please configure API client."
            },
            "raw_response": "",
            "metrics": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "elapsed_time": 0.0}
        }


# Convenience functions for creating agents with config
def create_planner_agent(api_secret_path: str = None) -> PlannerAgent:
    """Create a planner agent using config settings"""
    planner_config = config["planner_agent"]
    return PlannerAgent(
        name="Planner",
        role="System Architect",
        ability="Expert in multi-agent system design and strategic resource allocation",
        model=planner_config["model"],
        temperature=planner_config["temperature"],
        prompt_strategy=planner_config.get("prompt_strategy", "default"),
        reasoning_effort=planner_config.get("reasoning_effort", "medium"),
        verbosity=planner_config.get("verbosity", "medium"),
        api_secret_path=api_secret_path
    )

def create_supervisor_agent(api_secret_path: str = None) -> SupervisorAgent:
    """Create a supervisor agent using config settings"""
    supervisor_config = config["supervisor_agent"]
    return SupervisorAgent(
        name="Graham Paxon",
        role="General Partner, co-founder of the firm; sits on board; has the strongest voice in deciding who gets invited and joins the round",
        ability="Comprehensive strategic oversight with strong leadership and decision-making skills; deep expertise in leading deals and managing limited partner relationships",
        profile="Balances the insights from all angles, ensuring that the firm's co-investor selections align with its strategic vision and goals. Integrates feedback from both partners and VPs to make informed decisions, fostering strong relationships with key stakeholders. Conducts risk assessments to ensure partnerships are sustainable and provides leadership and guidance to the team, driving the firm's success and growth.",
        model=supervisor_config["model"],
        temperature=supervisor_config["temperature"],
        mode=supervisor_config.get("mode", "importance"),
        importance_strategy=supervisor_config.get("importance_strategy", "business"),
        reasoning_effort=supervisor_config.get("reasoning_effort", "medium"),
        verbosity=supervisor_config.get("verbosity", "medium"),
        api_secret_path=api_secret_path
    )