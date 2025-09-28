import os
import json
import glob
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from utilities import ParallelProcessor, ProfileExtractor
from mas import MAS, create_specialized_agents_from_planner
from agent import create_planner_agent
from config import planner_config, specialized_config, supervisor_config


class ParallelMASTest:
    """
    Parallel testing system for MAS with resume capability and dynamic saving
    """

    def __init__(self, data_dir: str = "data3", results_dir: str = "data3-gpt-5-nano-default-gpt-5-nano-importance"):
        """
        Initialize the parallel test system

        Args:
            data_dir (str): Directory containing case CSV files
            results_dir (str): Directory for storing results with resume capability
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        print(f"🧪 Parallel MAS Test System Initialized")
        print(f"📂 Data directory: {self.data_dir}")
        print(f"📂 Results directory: {self.results_dir}")
        print(f"🕐 Session ID: {self.session_id}")

    def discover_all_cases(self) -> List[str]:
        """
        Discover all case files in the data directory

        Returns:
            List[str]: List of case IDs found in the data directory
        """
        case_pattern = os.path.join(self.data_dir, "case_*.csv")
        case_files = glob.glob(case_pattern)

        case_ids = []
        for filepath in case_files:
            filename = os.path.basename(filepath)
            # Extract case ID from filename (e.g., "case_592.csv" -> "592")
            match = re.match(r"case_(\d+)\.csv", filename)
            if match:
                case_ids.append(match.group(1))

        case_ids.sort(key=int)  # Sort numerically
        print(f"🔍 Discovered {len(case_ids)} cases in {self.data_dir}")
        return case_ids

    def get_completed_cases(self) -> List[str]:
        """
        Get list of already completed cases by checking existing result files

        Returns:
            List[str]: List of case IDs that have been completed
        """
        result_pattern = os.path.join(self.results_dir, "MAS_Log_case_*_final_selection_*.json")
        result_files = glob.glob(result_pattern)

        completed_cases = []
        for filepath in result_files:
            filename = os.path.basename(filepath)
            # Extract case ID from result filename (e.g., "MAS_Log_case_592_final_selection_20250914_213549.json" -> "592")
            match = re.search(r"MAS_Log_case_(\d+)_final_selection", filename)
            if match:
                case_id = match.group(1)
                if case_id not in completed_cases:
                    completed_cases.append(case_id)

        completed_cases.sort(key=int)  # Sort numerically
        print(f"✅ Found {len(completed_cases)} completed cases: {completed_cases}")
        return completed_cases

    def get_pending_cases(self) -> List[str]:
        """
        Get list of cases that still need to be processed (resume functionality)

        Returns:
            List[str]: List of case IDs that need to be processed
        """
        all_cases = self.discover_all_cases()
        completed_cases = self.get_completed_cases()

        # Remove completed cases from all cases
        pending_cases = [case_id for case_id in all_cases if case_id not in completed_cases]

        print(f"⏳ {len(pending_cases)} cases pending processing")
        if len(pending_cases) <= 10:
            print(f"📋 Pending cases: {pending_cases}")
        else:
            print(f"📋 First 10 pending cases: {pending_cases[:10]}...")

        return pending_cases

    def run_single_case(self, case_id: str) -> Dict:
        """
        Run MAS analysis for a single case

        Args:
            case_id (str): The case ID to process

        Returns:
            Dict: Results dictionary with success status and case information
        """
        case_file = os.path.join(self.data_dir, f"case_{case_id}.csv")

        if not os.path.exists(case_file):
            return {
                "case_id": case_id,
                "success": False,
                "error": f"Case file not found: {case_file}",
                "processing_time": 0
            }

        start_time = datetime.now()

        try:
            print(f"🔄 Processing case {case_id}...")
            csv_filename = f"case_{case_id}"

            # Load profiles from CSV
            extractor = ProfileExtractor()
            lead_profile, target_profile, candidate_profiles = extractor.load_from_csv(case_file)
            real_participants = extractor.get_real_participants(case_file)

            # Step 1: Create and run planner agent
            planner_agent = create_planner_agent("secret.json")
            planner_output = planner_agent.plan_agents(
                lead_profile=lead_profile,
                target_profile=target_profile,
                candidate_profiles=candidate_profiles
            )

            # Check if planner succeeded
            specialized_agent_configs = planner_output.get("agents", [])
            if not specialized_agent_configs:
                raise ValueError(f"Planner Agent failed to generate agent configurations for {csv_filename}")

            # Step 2: Create specialized agents
            specialized_agents = create_specialized_agents_from_planner(planner_output, "secret.json")

            if not specialized_agents:
                raise ValueError("No specialized agents could be created")

            # Step 3: Initialize and run MAS
            mas_system = MAS(
                specialized_agents=specialized_agents,
                candidate_profiles=candidate_profiles,
                csv_filename=csv_filename,
                planner_output=planner_output,
                api_secret_path="secret.json"
            )

            # Step 4: Conduct evaluation phase
            agent_outputs = mas_system.conduct_evaluation_phase()

            # Step 5: Finalize shortlist with supervisor
            final_shortlist = mas_system.finalize_shortlist(agent_outputs)

            # Step 6: Save logs and calculate metrics
            processing_time_step = (datetime.now() - start_time).total_seconds()
            log_path = mas_system.save_log(processing_time_step)

            # Prepare result data
            result = {
                "final_shortlist": final_shortlist,
                "real_participants": real_participants,
                "session_logs": mas_system.session_logs,
                "log_path": log_path
            }

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            print(f"✅ Case {case_id} completed in {processing_time:.1f}s")

            return {
                "case_id": case_id,
                "success": True,
                "result": result,
                "processing_time": processing_time,
                "timestamp": end_time.strftime("%Y%m%d_%H%M%S")
            }

        except Exception as e:
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            error_msg = f"Error processing case {case_id}: {str(e)}"
            print(f"❌ {error_msg}")

            return {
                "case_id": case_id,
                "success": False,
                "error": error_msg,
                "processing_time": processing_time
            }

    def save_batch_progress(self, results: List[Dict], batch_number: int):
        """
        Save progress summary for a batch of completed cases

        Args:
            results (List[Dict]): List of case results
            batch_number (int): Batch identifier
        """
        progress_file = os.path.join(self.results_dir, f"batch_progress_{self.session_id}_{batch_number:03d}.json")

        successful_cases = [r for r in results if r['success']]
        failed_cases = [r for r in results if not r['success']]

        total_processing_time = sum(r['processing_time'] for r in results)
        avg_processing_time = total_processing_time / len(results) if results else 0

        progress_data = {
            "batch_info": {
                "batch_number": batch_number,
                "session_id": self.session_id,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "total_cases": len(results)
            },
            "performance_summary": {
                "successful_cases": len(successful_cases),
                "failed_cases": len(failed_cases),
                "success_rate": len(successful_cases) / len(results) * 100 if results else 0,
                "total_processing_time_seconds": total_processing_time,
                "average_processing_time_seconds": avg_processing_time
            },
            "case_results": {
                "successful": [r['case_id'] for r in successful_cases],
                "failed": [{"case_id": r['case_id'], "error": r['error']} for r in failed_cases]
            }
        }

        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)

        print(f"💾 Batch {batch_number} progress saved: {len(successful_cases)}/{len(results)} successful")

    def run_parallel_tests(self, max_workers: int = 4) -> Dict:
        """
        Run parallel MAS tests with resume capability and dynamic saving

        Args:
            max_workers (int): Maximum number of parallel workers

        Returns:
            Dict: Overall test results and statistics
        """
        print(f"🚀 Starting Parallel MAS Testing")
        print(f"⚙️ Max workers: {max_workers}")
        print(f"💾 Immediate saving: Each case saved when completed")
        print("=" * 60)

        # Get pending cases (resume functionality)
        pending_cases = self.get_pending_cases()

        if not pending_cases:
            print("🎉 All cases are already completed!")
            return {
                "status": "completed",
                "message": "All cases already processed",
                "total_cases": 0,
                "new_cases_processed": 0
            }

        # Initialize parallel processor
        processor = ParallelProcessor(num_workers=max_workers)

        # Immediate file writing with progress callback (no batches needed!)
        start_time = datetime.now()

        print(f"\n🚀 Processing {len(pending_cases)} cases with immediate saving...")

        # Progress tracking variables
        completed_count = 0
        successful_count = 0
        failed_count = 0

        # Thread-safe file writing with progress callback
        import threading
        file_lock = threading.Lock()

        def progress_callback(index, result, completed, total):
            nonlocal completed_count, successful_count, failed_count

            case_id = pending_cases[index]
            completed_count += 1

            if result and result.get('success', False):
                successful_count += 1
                print(f"✅ Case {case_id} completed ({completed_count}/{total}) - Success")

                # Write result immediately to file (like RefCode approach)
                log_filename = f"MAS_Log_case_{case_id}_final_selection_{result.get('timestamp', self.session_id)}.json"
                log_path = os.path.join(self.results_dir, log_filename)

                with file_lock:
                    try:
                        # Save the individual case result immediately
                        with open(log_path, 'w') as f:
                            json.dump(result.get('result', {}).get('session_logs', {}), f, indent=2)
                    except Exception as e:
                        print(f"⚠️ Warning: Failed to save log for case {case_id}: {e}")

            else:
                failed_count += 1
                error_msg = result.get('error', 'Unknown error') if result else 'Task failed'
                print(f"❌ Case {case_id} failed ({completed_count}/{total}) - {error_msg}")

            # Progress update every 10 cases or batch completion
            if completed_count % 10 == 0 or completed_count == total:
                print(f"📊 Progress: {completed_count}/{total} ({completed_count/total*100:.1f}%) | ✅ {successful_count} | ❌ {failed_count}")

        # Prepare task arguments
        task_args = [(case_id,) for case_id in pending_cases]

        # Process all cases in parallel with immediate callback
        all_results = processor.process_with_callback(
            task_func=self.run_single_case,
            task_args_list=task_args,
            callback_func=progress_callback
        )

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Generate final summary
        successful_results = [r for r in all_results if r['success']]
        failed_results = [r for r in all_results if not r['success']]

        summary = {
            "session_info": {
                "session_id": self.session_id,
                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_processing_time_seconds": total_time
            },
            "configuration": {
                "max_workers": max_workers,
                "data_directory": self.data_dir,
                "results_directory": self.results_dir
            },
            "results_summary": {
                "total_cases_processed": len(all_results),
                "successful_cases": len(successful_results),
                "failed_cases": len(failed_results),
                "success_rate_percent": len(successful_results) / len(all_results) * 100 if all_results else 0,
                "average_processing_time_seconds": sum(r['processing_time'] for r in all_results) / len(all_results) if all_results else 0
            },
            "case_details": {
                "successful_case_ids": [r['case_id'] for r in successful_results],
                "failed_cases": [{"case_id": r['case_id'], "error": r['error']} for r in failed_results]
            }
        }

        # Save final summary
        summary_file = os.path.join(self.results_dir, f"parallel_test_summary_{self.session_id}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 60)
        print("🎯 PARALLEL TESTING COMPLETE")
        print(f"📊 Processed: {len(all_results)} cases")
        print(f"✅ Successful: {len(successful_results)} cases")
        print(f"❌ Failed: {len(failed_results)} cases")
        print(f"📈 Success rate: {summary['results_summary']['success_rate_percent']:.1f}%")
        print(f"⏱️ Total time: {total_time:.1f} seconds")
        print(f"⚡ Average time per case: {summary['results_summary']['average_processing_time_seconds']:.1f} seconds")
        print(f"💾 Results saved in: {self.results_dir}")
        print("=" * 60)

        return summary


def run_parallel_mas_test(max_workers: int = 4, data_dir: str = "data3", results_dir: str = "data3-gpt-5-nano-default-gpt-5-nano-importance"):
    """
    Convenience function to run parallel MAS tests

    Args:
        max_workers (int): Maximum number of parallel workers (default: 4)
        data_dir (str): Directory containing case files (default: "data3")
        results_dir (str): Directory for results (default: "data3-gpt-5-nano-default-gpt-5-nano-importance")

    Returns:
        Dict: Test results summary
    """
    tester = ParallelMASTest(data_dir=data_dir, results_dir=results_dir)
    return tester.run_parallel_tests(max_workers=max_workers)


if __name__ == "__main__":
    # Example usage with different configurations
    print("🧪 MAS Parallel Testing System")
    print("Choose configuration:")
    print("1. Conservative (2 workers)")
    print("2. Balanced (4 workers) [Default]")
    print("3. Aggressive (8 workers)")
    print("4. Custom")

    choice = input("Enter choice (1-4, or press Enter for default): ").strip()

    if choice == "1":
        max_workers = 2
    elif choice == "3":
        max_workers = 8
    elif choice == "4":
        max_workers = int(input("Enter max workers: "))
    else:
        max_workers = 4

    print(f"\n🚀 Starting with {max_workers} workers")

    # Run the parallel tests
    results = run_parallel_mas_test(max_workers=max_workers)

    print(f"\n🎉 Testing completed! Check results in data3-gpt-5-nano-default-gpt-5-nano-importance/")