import os
import json
import pandas as pd
from typing import Dict, List, Tuple
from tabulate import tabulate
import argparse

def load_csv_data(csv_path: str) -> Tuple[List[str], int]:
    """Load CSV data and return real matches and total count."""
    df = pd.read_csv(csv_path)
    real_matches = df[(df['real'] == 1) & (df['leadornot'] == 0)]['vcfirmid'].astype(str).str.strip().unique().tolist()
    return real_matches, len(real_matches)

def load_log_data(log_path: str) -> Dict:
    """Load JSON log data and return relevant statistics and evaluations."""
    with open(log_path, 'r', encoding='utf-8') as f:
        log_data = json.load(f)

    final_summary = log_data.get('final_summary', {})
    specialized_evals = log_data.get('specialized_agent_evaluations', [])

    if not final_summary and not specialized_evals:
        session_logs = log_data.get('result', {}).get('session_logs', {})
        if not session_logs:
            session_logs = log_data.get('run_details', {})
        final_summary = session_logs.get('final_summary', {})
        specialized_evals = session_logs.get('specialized_agent_evaluations', [])
    
    match_stats = final_summary.get('match_statistics', {})
    shortlist = final_summary.get('shortlisted_firm_ids', [])

    return {
        'shortlist': [str(firm_id).strip() for firm_id in shortlist if isinstance(firm_id, (str, int))],
        'match_stats': match_stats,
        'specialized_evals': specialized_evals
    }

def analyze_case(case_id: str, data_dir: str, log_dir: str) -> Dict:
    """Analyze a single case and return detailed statistics, including best agent performance."""
    csv_path = os.path.join(data_dir, f"case_{case_id}.csv")
    if not os.path.exists(csv_path):
        return {"error": f"CSV file not found for case {case_id} at {csv_path}"}

    log_files = [f for f in os.listdir(log_dir) if f"case_{case_id}" in f and f.endswith(".json")]
    if not log_files:
        return {"error": f"No log file found for case {case_id} in {log_dir}"}

    log_path = os.path.join(log_dir, sorted(log_files)[-1])

    real_matches, total_real_matches_in_csv = load_csv_data(csv_path)
    log_data = load_log_data(log_path)

    shortlist = log_data['shortlist']
    specialized_evals = log_data['specialized_evals']

    # --- Supervisor Performance Analysis ---
    actual_matches_in_shortlist = [firm_id for firm_id in shortlist if firm_id in real_matches]
    actual_match_count = len(actual_matches_in_shortlist)
    supervisor_accuracy = (actual_match_count / total_real_matches_in_csv) * 100 if total_real_matches_in_csv > 0 else 0.0

    # --- Best Specialized Agent Analysis ---
    best_agent_name = "N/A"
    best_agent_accuracy = 0.0
    if specialized_evals and total_real_matches_in_csv > 0:
        for agent_eval in specialized_evals:
            agent_name = agent_eval.get('agent_name', 'Unknown Agent')
            parsed_output = agent_eval.get('parsed_output', {})
            if not parsed_output or not isinstance(parsed_output, dict): continue

            agent_recommendations = [str(cand.get('firm_id')).strip() for cand in parsed_output.get('ranked_candidates', [])]
            unique_agent_shortlist = list(set(agent_recommendations))

            matches_by_agent = [firm_id for firm_id in unique_agent_shortlist if firm_id in real_matches]
            agent_accuracy = (len(matches_by_agent) / total_real_matches_in_csv) * 100

            if agent_accuracy >= best_agent_accuracy:
                best_agent_accuracy = agent_accuracy
                best_agent_name = agent_name

    # --- Final Data Compilation ---
    df_original = pd.read_csv(csv_path)
    total_candidates_evaluated = len(df_original[df_original['leadornot'] == 0]['vcfirmid'].astype(str).unique())

    return {
        'case_id': case_id,
        'case_number': int(case_id),
        'total_candidates_evaluated': total_candidates_evaluated,
        'total_real_matches_in_csv': total_real_matches_in_csv,
        'shortlist': shortlist,
        'actual_top_k_selected': len(shortlist),
        'supervisor_accuracy': supervisor_accuracy,
        'best_agent_name': best_agent_name,
        'best_agent_accuracy': best_agent_accuracy,
        'log_file': log_path
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze multi-agent system performance by comparing log data with ground truth CSVs.")
    parser.add_argument('--log_dir', type=str, default='data3_test-gpt-5-nano-business-gpt-5-nano-importance-business',
                        help="Path to the directory containing MAS log files.")
    parser.add_argument('--data_dir', type=str, default='data3_test',
                        help="Path to the directory containing original CSV data files.")
    args = parser.parse_args()

    # --- Directory and File Validation ---
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist!")
        return
    case_ids = [fname[5:-4] for fname in os.listdir(args.data_dir) if fname.startswith("case_") and fname.endswith(".csv")]
    print(f"Found {len(case_ids)} case files in {args.data_dir} directory.")

    if not os.path.exists(args.log_dir):
        print(f"Error: Log directory '{args.log_dir}' does not exist!")
        return

    # --- Analysis Loop ---
    results = []
    for case_id in sorted(case_ids, key=int):
        result = analyze_case(case_id, args.data_dir, args.log_dir)
        if "error" in result:
            print(f"Skipping case {case_id}: {result['error']}")
            continue
        results.append(result)

    if not results:
        print("\nNo valid results to analyze! Please check if logs exist for the cases in your data directory.")
        return

    # --- Detailed Per-Case Output ---
    print("\n" + "="*80)
    print(" " * 28 + "Agent Performance Analysis")
    print("="*80)
    print(f"Total cases analyzed: {len(results)}")

    for result in results:
        print(f"\n--- Case {result['case_id']} ---")
        print(f"  - Total candidates evaluated: {result['total_candidates_evaluated']}")
        print(f"  - Total real matches in dataset: {result['total_real_matches_in_csv']}")
        print(f"  - Supervisor's Accuracy (Match Rate): {result['supervisor_accuracy']:.2f}%")
        print(f"  - Best Performing Agent: '{result['best_agent_name']}' with {result['best_agent_accuracy']:.2f}% accuracy")

    # --- Summary Table Output ---
    print("\n" + "="*100)
    print(" " * 39 + "Overall Performance Summary")
    print("="*100)
    headers = [
        "Case ID", "Total Candidates", "Top K", "Supervisor Acc. (%)",
        "Best Agent", "Best Agent Acc. (%)"
    ]
    table_data = []

    for result in results:
        row = [
            result['case_id'],
            result['total_candidates_evaluated'],
            result['actual_top_k_selected'],
            f"{result['supervisor_accuracy']:.2f}",
            result['best_agent_name'],
            f"{result['best_agent_accuracy']:.2f}"
        ]
        table_data.append(row)

    if results:
        avg_supervisor_acc = sum(r['supervisor_accuracy'] for r in results) / len(results)
        avg_best_agent_acc = sum(r['best_agent_accuracy'] for r in results) / len(results)

        table_data.append(["-"*8, "-"*16, "-"*5, "-"*21, "-"*20, "-"*21]) # Separator
        table_data.append([
            "Average",
            f"{sum(r['total_candidates_evaluated'] for r in results) / len(results):.2f}",
            f"{sum(r['actual_top_k_selected'] for r in results) / len(results):.2f}",
            f"{avg_supervisor_acc:.2f}",
            "---",
            f"{avg_best_agent_acc:.2f}"
        ])

    print(tabulate(table_data, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()