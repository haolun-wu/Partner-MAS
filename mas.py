"""
Multi-Agent System (MAS) Orchestrator
Migrated and enhanced from Archive/mas.py with Pydantic integration
"""

import json
import os
import math
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
from config import config
from utilities import ProfileExtractor
from agent import SpecializedAgent, SupervisorAgent


class MAS:
    """
    Multi-Agent System orchestrator for co-investor selection.
    Coordinates planner, specialized agents, and supervisor for complete evaluation workflow.
    """

    def __init__(self, specialized_agents: List[SpecializedAgent], candidate_profiles: Dict,
                 csv_filename: str, planner_output: Dict, api_secret_path: str = None):
        """
        Initialize the MAS with agents and data.

        Args:
            specialized_agents: List of configured specialized agents
            candidate_profiles: Dictionary of candidate co-investor profiles
            csv_filename: Name of the case file being processed
            planner_output: Output from the planner agent
            api_secret_path: Path to API secrets file
        """
        self.specialized_agents = specialized_agents
        self.candidates = candidate_profiles
        self.csv_filename = csv_filename
        self.api_secret_path = api_secret_path

        # Extract target profile - using ProfileExtractor instead of Archive Profiles class
        self.target_profile = self._extract_target_profile()
        self.planner_guidance = planner_output.get("strategic_guidance", "No strategic guidance was provided.")

        # Initialize session logs
        self.session_logs = {
            "planner_output": planner_output,
            "specialized_agent_evaluations": [],
            "supervisor_evaluation": None,
            "final_summary": None
        }

        # Create supervisor agent
        supervisor_agent_config = config["supervisor_agent"]
        self.supervisor = SupervisorAgent(
            name="Graham Paxon",
            role="General Partner, co-founder of the firm; sits on board; has the strongest voice in deciding who gets invited and joins the round",
            ability="Comprehensive strategic oversight with strong leadership and decision-making skills; deep expertise in leading deals and managing limited partner relationships",
            profile="Balances the insights from all angles, ensuring that the firm's co-investor selections align with its strategic vision and goals. Integrates feedback from both partners and VPs to make informed decisions, fostering strong relationships with key stakeholders. Conducts risk assessments to ensure partnerships are sustainable and provides leadership and guidance to the team, driving the firm's success and growth.",
            model=supervisor_agent_config["model"],
            temperature=supervisor_agent_config["temperature"],
            mode=supervisor_agent_config.get("mode", "importance"),
            reasoning_effort=supervisor_agent_config.get("reasoning_effort", "medium"),
            verbosity=supervisor_agent_config.get("verbosity", "medium"),
            api_secret_path=api_secret_path
        )

    def _extract_target_profile(self) -> str:
        """Extract target profile from the case data"""
        # Try to extract target profile from case file
        try:
            data_file = os.path.join(config.get("data_dir", "data3"), f"{self.csv_filename}.csv")
            if os.path.exists(data_file):
                extractor = ProfileExtractor()
                _, target_profile, _ = extractor.load_from_csv(data_file)
                return target_profile
        except Exception as e:
            print(f"⚠️ Could not extract target profile: {e}")

        return "Target profile unavailable - please configure data directory correctly"

    def conduct_evaluation_phase(self) -> Dict:
        """
        Conduct the evaluation phase with all specialized agents.

        Returns:
            Dict: Agent evaluations for supervisor input
        """
        print("\n🔄 Specialized Agents conducting evaluations and ranking...")
        agent_outputs_for_supervisor = {}

        for agent in self.specialized_agents:
            print(f"\n📝 {agent.name} is evaluating candidates...")
            try:
                evaluation_result = agent.evaluate_candidates_and_rank(
                    candidates_data=self.candidates,
                    target_profile=self.target_profile
                )

                # Log the evaluation
                log_entry = {
                    "agent_name": agent.name,
                    "agent_profile": agent.profile,
                    "parsed_output": evaluation_result.get("parsed_output"),
                    "raw_response": evaluation_result.get("raw_response"),
                    "metrics": evaluation_result.get("metrics"),
                    "validation_successful": evaluation_result.get("validation_successful", False)
                }
                self.session_logs["specialized_agent_evaluations"].append(log_entry)

                # Prepare output for supervisor
                output = evaluation_result.get("parsed_output")
                agent_outputs_for_supervisor[agent.name] = output

                # Display results
                if output and isinstance(output, dict):
                    ranked_candidates = output.get('ranked_candidates', [])
                    print(f"✅ {agent.name} completed evaluation. Found {len(ranked_candidates)} ranked candidates:")
                    print(f"  Focused Features: {output.get('evaluation_focus', 'No focused features.')}")
                    print(f"  Overall Rationale: {output.get('overall_rationale', 'No overall rationale.')}")

                    for cand in ranked_candidates[:3]:  # Show top 3
                        if isinstance(cand, dict):
                            firm_id = cand.get('firm_id', 'N/A')
                            rank = cand.get('rank', 'N/A')
                            alignment_score = cand.get('alignment_score', 'N/A')
                            rationale = cand.get('rationale', 'No rationale provided.')
                            print(f"  - Rank {rank}: Firm ID {firm_id}, Alignment: {alignment_score}/10, Rationale: {rationale[:100]}...")
                else:
                    print(f"⚠️ {agent.name} returned unexpected output format")

            except Exception as e:
                print(f"💥 CRITICAL ERROR during {agent.name}'s evaluation: {e}")
                error_log = {"agent_name": agent.name, "error": str(e)}
                self.session_logs["specialized_agent_evaluations"].append(error_log)
                agent_outputs_for_supervisor[agent.name] = {"ranked_candidates": [], "error": True}
                continue

        return agent_outputs_for_supervisor

    def finalize_shortlist(self, agent_outputs: Dict) -> List[str]:
        """
        Have the supervisor make the final candidate selection.

        Args:
            agent_outputs: Dictionary of agent evaluation results

        Returns:
            List of selected candidate firm IDs
        """
        print("\n👨‍💼 Supervisor reviewing agent evaluations and finalizing shortlist...")

        # Calculate dynamic top-k (same logic as Archive)
        total_candidates = len(self.candidates)
        dynamic_top_k = max(1, math.ceil(total_candidates / 3))
        dynamic_top_k = min(dynamic_top_k, total_candidates)

        # Prepare supervisor input
        supervisor_input = {
            "target_profile": self.target_profile,
            "all_agent_evaluations": agent_outputs,
            "candidates_data": self.candidates,
            "planner_strategic_guidance": self.planner_guidance
        }

        # Get supervisor decision
        supervisor_full_output = self.supervisor.select_final_shortlist(
            supervisor_input=supervisor_input,
            top_k=dynamic_top_k
        )

        # Log supervisor evaluation
        self.session_logs["supervisor_evaluation"] = supervisor_full_output

        # Extract final selection
        final_selection = supervisor_full_output.get("final_selection", {})
        final_decision_parsed = final_selection.get("parsed_output")

        if isinstance(final_decision_parsed, dict):
            shortlist_raw = final_decision_parsed.get("selected_candidates", [])
            rationale = final_decision_parsed.get("rationale", "No rationale provided by supervisor.")
        else:
            print(f"❌ SUPERVISOR ERROR: The supervisor's final decision was not a valid dictionary.")
            print(f"   Raw output received: {final_selection.get('raw_response')}")
            shortlist_raw = []
            rationale = "Supervisor parsing failed. Could not generate a final rationale."

        # Clean and deduplicate shortlist
        shortlist = sorted(list(set(shortlist_raw)), key=lambda x: shortlist_raw.index(x) if x in shortlist_raw else 0)

        # Analyze matches against real data
        match_count, matched_firms = self.check_real_matches(shortlist)
        total_real_matches = self._get_total_real_matches()

        match_rate = (match_count / total_real_matches) * 100 if total_real_matches > 0 else 0

        # Create final summary
        final_summary = {
            "shortlisted_firm_ids": shortlist,
            "supervisor_rationale": rationale,
            "match_statistics": {
                "total_candidates_in_shortlist": len(shortlist),
                "total_real_matches_in_dataset": total_real_matches,
                "matches_found_count": match_count,
                "match_rate_percent": f"{match_rate:.2f}%",
                "matched_firm_ids": matched_firms
            }
        }
        self.session_logs["final_summary"] = final_summary

        # Display results
        print("\n🎯 [Final Shortlisted Candidates by Supervisor]:")
        for i, firm_id in enumerate(shortlist, 1):
            print(f"  {i}. Firm ID {firm_id}")
        print(f"\n📊 Rationale: {rationale}")
        print(f"📈 Match Rate: {match_rate:.2f}% ({match_count}/{total_real_matches} actual co-investors identified)")

        return shortlist

    def save_log(self, total_case_time: float) -> str:
        """
        Save comprehensive logs of the MAS session.

        Args:
            total_case_time: Total time taken to process the case

        Returns:
            Path to the saved log file
        """
        logs_folder = config.get("logs_dir", "logs")
        os.makedirs(logs_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"MAS_Log_{self.csv_filename}_final_selection_{timestamp}.json"
        filepath = os.path.join(logs_folder, filename)

        # Calculate total token usage
        total_prompt_tokens, total_completion_tokens = 0, 0

        # Collect metrics from all sources
        all_metrics_sources = []
        if self.session_logs.get("planner_output"):
            all_metrics_sources.append(self.session_logs["planner_output"].get("metrics", {}))

        for eval_log in self.session_logs.get("specialized_agent_evaluations", []):
            all_metrics_sources.append(eval_log.get("metrics", {}))

        supervisor_log = self.session_logs.get("supervisor_evaluation", {})
        if supervisor_log:
            for key, value in supervisor_log.items():
                if isinstance(value, dict) and 'metrics' in value:
                    all_metrics_sources.append(value.get('metrics', {}))

        # Sum up tokens
        for metrics in all_metrics_sources:
            if metrics:
                total_prompt_tokens += metrics.get("prompt_tokens", 0)
                total_completion_tokens += metrics.get("completion_tokens", 0)

        total_tokens = total_prompt_tokens + total_completion_tokens

        # Create log data
        log_data = {
            "case_info": {
                "timestamp": timestamp,
                "csv_file": self.csv_filename,
                "pydantic_integration": config.get("pydantic", {}).get("use_pydantic_validation", False)
            },
            "performance_summary": {
                "total_case_processing_time_seconds": round(total_case_time, 2),
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens
            },
            "run_details": self.session_logs,
            "config_snapshot": config
        }

        # Save log
        with open(filepath, 'w') as log_file:
            json.dump(log_data, log_file, indent=2)

        print(f"\n📌 Final log saved at: {filepath}")
        return filepath

    def check_real_matches(self, shortlist: List[str]) -> tuple:
        """
        Check how many of the shortlisted candidates are actual co-investors.
        This version includes robust stripping of whitespace to prevent matching errors.

        Args:
            shortlist: List of firm IDs in the shortlist

        Returns:
            Tuple of (match_count, matched_firm_ids)
        """
        data_file = os.path.join(config.get("data_dir", "data3"), f"{self.csv_filename}.csv")
        try:
            df = pd.read_csv(data_file)
            real_matches = df[(df['real'] == 1) & (df['leadornot'] == 0)]['vcfirmid'].astype(str).str.strip().unique()
            shortlist_str = [str(firm_id).strip() for firm_id in shortlist]
            matches = [firm_id for firm_id in shortlist_str if firm_id in real_matches]

            print(f"\n📊 Real Match Analysis: {len(matches)}/{len(real_matches)} actual co-investors identified")
            if matches:
                print(f"   ✅ Correctly identified: {matches}")
            if len(matches) < len(real_matches):
                missed = [firm for firm in real_matches if firm not in shortlist_str]
                print(f"   ❌ Missed co-investors: {missed}")

            return len(matches), matches
        except Exception as e:
            print(f"\n⚠️ Error checking real matches: {e}")
            return 0, []

    def _get_total_real_matches(self) -> int:
        """Get the total number of actual co-investors in the dataset"""
        try:
            data_file = os.path.join(config.get("data_dir", "data3"), f"{self.csv_filename}.csv")
            df = pd.read_csv(data_file)
            real_matches_df = df[(df['real'] == 1) & (df['leadornot'] == 0)]['vcfirmid'].astype(str)
            return len(real_matches_df.unique())
        except Exception as e:
            print(f"⚠️ Error counting real matches: {e}")
            return 1  # Avoid division by zero


# Utility function for creating specialized agents from planner output
def create_specialized_agents_from_planner(planner_output: Dict, api_secret_path: str = None) -> List[SpecializedAgent]:
    """
    Create specialized agents based on planner output.

    Args:
        planner_output: Output from planner agent containing agent configurations
        api_secret_path: Path to API secrets file

    Returns:
        List of configured specialized agents
    """
    agent_configs = planner_output.get("agents", [])
    print(f"🔍 DEBUG: Found {len(agent_configs)} agent configs in planner output")

    specialized_config = config["specialized_agent"]

    specialized_agents = []
    for i, agent_config in enumerate(agent_configs, 1):
        print(f"🔍 DEBUG: Processing agent config {i}: {agent_config}")

        if isinstance(agent_config, dict):
            required_keys = ["name", "role", "ability", "profile"]
            missing_keys = [key for key in required_keys if key not in agent_config]

            if missing_keys:
                print(f"❌ Agent config {i} missing keys: {missing_keys}")
                continue

            print(f"✅ Agent config {i} has all required keys")

            try:
                agent = SpecializedAgent(
                    name=agent_config["name"],
                    role=agent_config["role"],
                    ability=agent_config["ability"],
                    profile=agent_config["profile"],
                    model=specialized_config["model"],
                    temperature=specialized_config["temperature"],
                    reasoning_effort=specialized_config.get("reasoning_effort", "medium"),
                    verbosity=specialized_config.get("verbosity", "medium"),
                    api_secret_path=api_secret_path
                )
                specialized_agents.append(agent)
                print(f"✅ Created specialized agent: {agent_config['name']}")
            except Exception as e:
                print(f"❌ Error creating agent {i}: {e}")
        else:
            print(f"❌ Agent config {i} is not a dictionary: {type(agent_config)}")

    print(f"🔍 DEBUG: Created {len(specialized_agents)} specialized agents total")
    return specialized_agents