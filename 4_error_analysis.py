import json

# Compare the two runs at n=7, the hybrid's best setting.
BASELINE_FILE = 'full_log_baseline_n7.json' 
HYBRID_FILE = 'full_log_hybrid_n7.json'

def export_analysis_to_json():
    try:
        with open(BASELINE_FILE, 'r') as f:
            baseline = json.load(f)
        with open(HYBRID_FILE, 'r') as f:
            hybrid = json.load(f)
    except FileNotFoundError:
        print("Error: Ensure both JSON files are in the current directory.")
        return

    analysis_results = {
        "fixed_by_hybrid": [],
        "broken_by_hybrid": []
    }

    for query, h_data in hybrid.items():
        if query not in baseline:
            continue
            
        b_correct = baseline[query].get('is_correct', False)
        h_correct = h_data.get('is_correct', False)

        # FIXED: Hybrid won where Baseline failed
        if h_correct and not b_correct:
            analysis_results["fixed_by_hybrid"].append({
                "query": query,
                "ground_truth": h_data['ground_truth'],
                "hybrid_prediction": h_data['prediction'],
                "baseline_prediction": baseline[query]['prediction'],
                "reasoning_type": h_data.get('type', 'N/A')
            })

        # BROKEN: Hybrid failed where Baseline won
        if b_correct and not h_correct:
            analysis_results["broken_by_hybrid"].append({
                "query": query,
                "ground_truth": h_data['ground_truth'],
                "baseline_prediction": baseline[query]['prediction'],
                "hybrid_prediction": h_data['prediction'],
                "reasoning_type": h_data.get('type', 'N/A')
            })

    # Save to a new JSON file
    with open('hybrid_error_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=4)
    
    print(f"File saved! Found {len(analysis_results['fixed_by_hybrid'])} fixed and {len(analysis_results['broken_by_hybrid'])} broken cases.")

if __name__ == "__main__":
    export_analysis_to_json()