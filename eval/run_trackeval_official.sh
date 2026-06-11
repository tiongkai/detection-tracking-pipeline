#!/bin/bash

cd /Users/xhshen/Desktop/detection-tracking-pipeline

# Run TrackEval official script with a trick - evaluate as 'pedestrian' but your data is boats
# This works because TrackEval is class agnostic at the metric level
python TrackEval/scripts/run_mot_challenge.py \
  --GT_FOLDER trackeval_data/gt/mot_challenge \
  --TRACKERS_FOLDER trackeval_data/trackers/mot_challenge \
  --TRACKERS_TO_EVAL exp00_baseline \
  --BENCHMARK boat_tracking \
  --SPLIT_TO_EVAL train \
  --CLASSES_TO_EVAL pedestrian \
  --USE_PARALLEL False \
  --NUM_PARALLEL_CORES 1 \
  2>&1 | tee trackeval_output.log

echo "Done! Check trackeval_output.log"
