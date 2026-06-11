#!/usr/bin/env python
"""Simple TrackEval runner for boat tracking."""

import sys
import os
from pathlib import Path

# Add TrackEval to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'TrackEval'))

from trackeval import Evaluator
from trackeval.datasets import mot_challenge_2d_box

def main():
    # Set up paths
    gt_folder = "trackeval_data/gt/mot_challenge"
    trackers_folder = "trackeval_data/trackers/mot_challenge"
    output_folder = "results/trackeval_output"
    
    # Create output directory
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Configuration - class agnostic (evaluate all classes together)
    config = {
        'GT_FOLDER': gt_folder,
        'TRACKERS_FOLDER': trackers_folder,
        'OUTPUT_FOLDER': output_folder,
        'TRACKERS_TO_EVAL': ['exp00_baseline'],
        'CLASSES_TO_EVAL': ['boat_rgb', 'boat_thermal'],  # Your boat classes
        'BENCHMARK': 'boat_tracking',
        'SPLIT_TO_EVAL': 'train',
        'PRINT_CONFIG': False,
        'USE_PARALLEL': False,
        'NUM_PARALLEL_CORES': 1,
        'TRACKER_SUB_FOLDER': 'data',
        'SKIP_SPLIT_FOL': True,
    }
    
    print("=" * 60)
    print("TrackEval for Boat Tracking (Class Agnostic)")
    print("=" * 60)
    print(f"GT Folder: {gt_folder}")
    print(f"Trackers: {trackers_folder}")
    print(f"Output: {output_folder}")
    print("=" * 60)
    
    # Run evaluation
    dataset = mot_challenge_2d_box(config)
    evaluator = Evaluator(dataset)
    results = evaluator.evaluate()
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    for tracker_name, tracker_results in results.items():
        print(f"\nTracker: {tracker_name}")
        
        # CLEAR metrics (MOTA, etc.)
        if 'CLEAR' in tracker_results:
            clear = tracker_results['CLEAR']
            print(f"  MOTA: {clear.get('MOTA', 0):.3f}")
            print(f"  MOTP: {clear.get('MOTP', 0):.3f}")
            print(f"  ID Switches: {clear.get('IDSW', 0)}")
            print(f"  FP: {clear.get('CLR_FP', 0)}")
            print(f"  FN: {clear.get('CLR_FN', 0)}")
            print(f"  Precision: {clear.get('CLR_TP', 0) / (clear.get('CLR_TP', 0) + clear.get('CLR_FP', 0) + 1e-9):.3f}")
            print(f"  Recall: {clear.get('CLR_TP', 0) / (clear.get('CLR_TP', 0) + clear.get('CLR_FN', 0) + 1e-9):.3f}")
        
        # Identity metrics (IDF1)
        if 'Identity' in tracker_results:
            identity = tracker_results['Identity']
            print(f"  IDF1: {identity.get('IDF1', 0):.3f}")
        
        # HOTA metrics
        if 'HOTA' in tracker_results:
            hota = tracker_results['HOTA']
            print(f"  HOTA: {hota.get('HOTA', 0):.3f}")
            print(f"  DetA: {hota.get('DetA', 0):.3f}")
            print(f"  AssA: {hota.get('AssA', 0):.3f}")
        
        # VACE metrics (MT, ML, Frag)
        if 'VACE' in tracker_results:
            vace = tracker_results['VACE']
            print(f"  Mostly Tracked: {vace.get('MT', 0)}")
            print(f"  Mostly Lost: {vace.get('ML', 0)}")
            print(f"  Fragments: {vace.get('Frag', 0)}")
    
    print("\n" + "=" * 60)
    print(f"Full results saved to: {output_folder}")
    print("=" * 60)

if __name__ == "__main__":
    main()
