export PYTHONPATH=.:$PYTHONPATH

python scripts/train.py \
  --dataset "data/news20/news20_train/" \
  --model_dir "models/news20/" \
  --embedding "one_hot" \
  --overwrite
