export PYTHONPATH=.:$PYTHONPATH

python scripts/evaluate.py \
  --dataset "data/news20/news20_test/" \
  --model_dir "models/news20/" \
  --output_results "models/news20/evaluation.json" \
  --output_predictions "models/news20/predictions.csv" \
  --cache
