#!/usr/bin/env python3
"""
Main execution script for the Multi-Agent System
Enhanced version migrated from Archive with Pydantic integration
"""

import os
import glob
import time
from typing import Optional
from mas import MAS, create_specialized_agents_from_planner
from agent import create_planner_agent, create_supervisor_agent
from utilities import ProfileExtractor
from config import config


def process_csv_file(csv_file_path: str, api_secret_path: Optional[str] = None) -> bool:
    """
    Process a single CSV file with the complete MAS pipeline.

    Args:
        csv_file_path: Path to the CSV case file
        api_secret_path: Path to API secrets file (optional)

    Returns:
        True if processing succeeded, False otherwise
    """
    case_start_time = time.time()
    try:
        print(f"\nProcessing file: {csv_file_path}")
        csv_filename = os.path.splitext(os.path.basename(csv_file_path))[0]

        # Step 1: Load profiles
        print(f"Loading profiles from {csv_file_path}...")
        extractor = ProfileExtractor()
        lead_profile, target_profile, candidates = extractor.load_from_csv(csv_file_path)
        print(f"✅ Loaded {len(candidates)} candidates")

        # Step 2: Create and run planner agent
        print("\n🔧 Executing Planner Agent...")
        planner_agent = create_planner_agent(api_secret_path)
        planner_output = planner_agent.plan_agents(
            lead_profile=lead_profile,
            target_profile=target_profile,
            candidate_profiles=candidates
        )

        # Check if planner succeeded
        specialized_agent_configs = planner_output.get("agents", [])
        if not specialized_agent_configs:
            print(f"❌ Planner Agent failed to generate agent configurations for {csv_filename}")
            return False

        print(f"✅ Planner Agent designed {len(specialized_agent_configs)} specialized agents:")
        for ac in specialized_agent_configs:
            if isinstance(ac, dict):
                print(f"  - {ac.get('name', 'Unknown')}: {ac.get('role', 'Unknown role')}")

        # Step 3: Create specialized agents
        print("\n🔍 Creating Specialized Agents...")
        specialized_agents = create_specialized_agents_from_planner(planner_output, api_secret_path)

        if not specialized_agents:
            print("❌ No specialized agents could be created")
            return False

        # Step 4: Initialize and run MAS
        print(f"\n🚀 Initializing Multi-Agent System for case {csv_filename}...")
        mas_system = MAS(
            specialized_agents=specialized_agents,
            candidate_profiles=candidates,
            csv_filename=csv_filename,
            planner_output=planner_output,
            api_secret_path=api_secret_path
        )

        # Step 5: Conduct evaluation phase
        print("\n📊 Conducting Agent Evaluation Phase...")
        agent_outputs = mas_system.conduct_evaluation_phase()

        # Step 6: Finalize shortlist with supervisor
        print("\n👨‍💼 Supervisor Making Final Selection...")
        final_shortlist = mas_system.finalize_shortlist(agent_outputs)

        # Step 7: Save logs and calculate metrics
        total_case_time = time.time() - case_start_time
        print(f"\n⏱️ Total processing time: {total_case_time:.2f} seconds")

        log_path = mas_system.save_log(total_case_time)
        print(f"📂 Session log saved: {log_path}")

        # Summary
        final_summary = mas_system.session_logs.get("final_summary", {})
        match_stats = final_summary.get("match_statistics", {})

        print(f"\n🎯 CASE SUMMARY:")
        print(f"   📋 Candidates evaluated: {len(candidates)}")
        print(f"   🎯 Final shortlist: {len(final_shortlist)} firms")
        print(f"   📈 Match rate: {match_stats.get('match_rate_percent', 'N/A')}")
        print(f"   ✅ Processing: SUCCESS")

        return True

    except Exception as e:
        print(f"💥 Error processing {csv_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_case_number(filename: str) -> float:
    """Extract case number from filename for sorting"""
    try:
        return int(os.path.basename(filename).split('_')[1].split('.')[0])
    except (IndexError, ValueError):
        return float('inf')


def main():
    """Main execution function"""
    print("🚀 MULTI-AGENT SYSTEM FOR CO-INVESTOR SELECTION")
    print("=" * 60)
    print("Enhanced with Pydantic Integration & Structured Outputs")
    print("=" * 60)

    # Configuration
    data_dir = config.get("data_dir", "data3")
    use_pydantic = config.get("pydantic", {}).get("use_pydantic_validation", True)

    print(f"📂 Data directory: {data_dir}")
    print(f"🔧 Pydantic validation: {'Enabled' if use_pydantic else 'Disabled'}")

    # Find CSV files
    csv_pattern = os.path.join(data_dir, "*.csv")
    csv_files = sorted(glob.glob(csv_pattern), key=get_case_number)

    if not csv_files:
        print(f"❌ No CSV files found in {data_dir}")
        print("💡 Please ensure your data files are in the correct directory")
        return False

    print(f"\n📊 Found {len(csv_files)} CSV files to process:")
    for i, csv_file in enumerate(csv_files[:5], 1):  # Show first 5
        print(f"   {i}. {os.path.basename(csv_file)}")
    if len(csv_files) > 5:
        print(f"   ... and {len(csv_files) - 5} more files")

    # Check for API secrets (optional)
    api_secret_path = None
    if os.path.exists("secret.json"):
        api_secret_path = "secret.json"
        print("🔐 Found secret.json - API integration enabled")
    elif os.path.exists("secrets.json"):
        api_secret_path = "secrets.json"
        print("🔐 Found secrets.json - API integration enabled")
    elif os.path.exists("secrets.txt"):
        print("⚠️ Found secrets.txt - please convert to JSON format for full API integration")
    else:
        print("⚠️ No API secrets found - running in simulation mode")

    # Process files
    print(f"\n🔄 Processing {len(csv_files)} case files...")
    print("=" * 60)

    successful = 0
    failed = 0

    for csv_file in csv_files:
        success = process_csv_file(csv_file, api_secret_path)
        if success:
            successful += 1
        else:
            failed += 1

        print("-" * 40)

    # Final summary
    print("\n" + "=" * 60)
    print("🏁 FINAL SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success rate: {successful / len(csv_files) * 100:.1f}%")

    if successful > 0:
        print(f"📂 Logs saved in: {config.get('logs_dir', 'logs')}")

    return successful == len(csv_files)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)