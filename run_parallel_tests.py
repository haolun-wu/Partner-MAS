#!/usr/bin/env python3
"""
Simple launcher script for parallel MAS testing
Easy-to-use interface for running parallel tests with different configurations
"""

import argparse
from config import config
from parallel_test import run_parallel_mas_test


def main(workers=None):
    """
    Main function with optional workers parameter.
    Other parameters are sourced from config.py by default.
    """
    parser = argparse.ArgumentParser(description='Run parallel MAS tests with customizable parameters')

    # The number of workers can be set via command line
    parser.add_argument('--workers', '-w', type=int, default=4,
                      help='Maximum number of parallel workers (default: 4)')

    # USE the imported config for the default data directory
    parser.add_argument('--data-dir', '-d', type=str, default=config.get('data_dir'),
                      help=f"Directory containing case CSV files (default from config: {config.get('data_dir')})")

    # USE the imported config for the default results directory
    parser.add_argument('--results-dir', '-r', type=str,
                      default=config.get('logs_dir'),
                      help=f"Directory for storing results (default from config: {config.get('logs_dir')})")

    args = parser.parse_args()

    final_workers = workers if workers is not None else args.workers

    print(f"🚀 Starting Parallel MAS Testing")
    print(f"⚙️  Configuration:")
    print(f"   - Workers: {final_workers}")
    print(f"   - Data directory: {args.data_dir}")
    print(f"   - Results directory: {args.results_dir}")
    print(f"   - Immediate saving: Each case saved when completed")
    print("=" * 60)

    # Run the parallel tests
    results = run_parallel_mas_test(
        max_workers=final_workers,
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )

    # Print final summary
    print(f"\n🎊 Testing Complete!")
    print(f"📊 Final Statistics:")
    print(f"   - Total processed: {results['results_summary']['total_cases_processed']}")
    print(f"   - Success rate: {results['results_summary']['success_rate_percent']:.1f}%")
    print(f"   - Total time: {results['session_info']['total_processing_time_seconds']:.1f} seconds")
    print(f"   - Avg time per case: {results['results_summary']['average_processing_time_seconds']:.1f} seconds")

    if results['results_summary']['failed_cases'] > 0:
        print(f"⚠️  Failed cases: {results['results_summary']['failed_cases']}")
        print("   Check individual logs for error details")

    print(f"\n📂 Results saved in: {args.results_dir}")


if __name__ == "__main__":
    WORKERS = 20
    main(workers=WORKERS)