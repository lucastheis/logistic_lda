DATA_DIR:='data'
METADATA_DIR:='metadata_new'
TRAINING_DATASET_NG20:='tokenized_ng20_train.pkl'
TRAINING_DATASET_PINTEREST:='tf_pinterest_train'

# this would be the value for variables that should not be used by the models
INF:=$(shell python -c "import sys; print(-sys.maxsize -1);")


ng20_unsupervised:
	python train.py \
		--experiment_name 'ng20_unsupervised' \
		--data_dir ${DATA_DIR} \
		--metadata_dir ${METADATA_DIR} \
		--filename ${TRAINING_DATASET_NG20} \
		--output_file '' \
		--model 'logistic_lda' \
		--vocab_file 'vocab2K.pkl' \
		--vocab_size  ${INF} \
		--batch_size 16 \
		--author_topic_weight 1. \
		--topic_bias_regularization 5.0 \
		--model_regularization 100.0 \
		--initial_learning_rate 0.0005 \
		--learning_rate_decay 0.8 \
		--learning_rate_decay_steps 2000 \
		--max_steps 50000 \
		--num_valid 0 \
		--items_per_author ${INF} \
		--n_author_topic_iterations ${INF} \
		--use_author_topics 0 \
		--n_unsupervised_topics 50 \
		--hidden_units '128' \
		--embedding 'glove' ; \
	python evaluate.py \
		--experiment_name 'ng20_unsupervised' \
		--evaluate_unsupervised 1 \
		--data_dir ${DATA_DIR} \
		--metadata_dir ${METADATA_DIR}


ng20_supervised_ce:
	python train.py \
		--experiment_name 'ng20_supervised_ce' \
		--data_dir ${DATA_DIR} \
		--metadata_dir ${METADATA_DIR} \
		--filename ${TRAINING_DATASET_NG20} \
		--output_file 'validation.txt' \
		--model 'logistic_lda_ce' \
		--vocab_file '' \
		--vocab_size 50000 \
		--batch_size 16 \
		--author_topic_weight ${INF} \
		--topic_bias_regularization 1.0 \
		--model_regularization ${INF} \
		--initial_learning_rate 0.001 \
		--learning_rate_decay 0.7 \
		--learning_rate_decay_steps 2000 \
		--max_steps 100000 \
		--num_valid 0.15 \
		--items_per_author ${INF} \
		--n_author_topic_iterations 20 \
		--use_author_topics 1 \
		--n_unsupervised_topics ${INF}\
		--hidden_units 1024 512 \
		--embedding 'glove' ; \
	python evaluate.py \
		--experiment_name 'ng20_supervised_ce' \
		--evaluate_unsupervised 0 \
		--data_dir ${DATA_DIR} \
		--metadata_dir ${METADATA_DIR}

search_ng_20_supervised_ce:
		$(eval EXP_ID:=$(shell python -c "import random; print(random.randint(0, 100000));")) \
		python train.py \
			--experiment_name 'ng20_supervised_ce_'${EXP_ID} \
			--data_dir ${DATA_DIR} \
			--metadata_dir ${METADATA_DIR} \
			--filename ${TRAINING_DATASET_NG20} \
			--output_file 'validation.txt' \
			--model 'logistic_lda_ce' \
			--vocab_file '' \
			--vocab_size 50000 \
			--batch_size $(shell python -c "import random; print(random.choice([16, 32]));") \
			--topic_bias_regularization $(shell python -c "import random; print(random.choice([1., 5.]));") \
			--model_regularization $(shell python -c "import random; print(random.choice([0., 5., 20.]));") \
			--initial_learning_rate $(shell python -c "import random; print(random.choice([0.001, 0.003]));") \
			--learning_rate_decay $(shell python -c "import random; print(random.choice([0.9, 0.8, 0.7, 0.5]));") \
			--max_steps $(shell python -c "import random; print(random.choice([50000, 100000, 200000]));") \
			--hidden_units $(shell python -c "import random; print(random.choice(['1024', '512', '1024 512']));") \
			--learning_rate_decay_steps 2000 \
			--num_valid 0.15 \
			--n_author_topic_iterations 20 \
			--author_topic_weight ${INF} \
			--items_per_author ${INF} \
			--n_unsupervised_topics ${INF}\
			--use_author_topics 1 \
			--embedding 'glove' ; \
		python evaluate.py \
		--experiment_name 'ng20_supervised_ce_'${EXP_ID} \
		--evaluate_unsupervised 0 \
		--data_dir '${DATA_DIR}' \
		--metadata_dir '${METADATA_DIR}'


ng20_supervised:
	python train.py \
		--experiment_name 'ng20_supervised' \
		--data_dir ${DATA_DIR} \
		--metadata_dir ${METADATA_DIR} \
		--filename ${TRAINING_DATASET_NG20} \
		--output_file 'validation.txt' \
		--model 'logistic_lda' \
		--vocab_file '' \
		--vocab_size 50000 \
		--batch_size 16 \
		--author_topic_weight 8000.0 \
		--topic_bias_regularization 1.0 \
		--model_regularization 20. \
		--initial_learning_rate 0.003 \
		--learning_rate_decay 0.9 \
		--learning_rate_decay_steps 2000 \
		--max_steps 100000 \
		--num_valid 0.15 \
		--items_per_author ${INF} \
		--n_author_topic_iterations 1 \
		--use_author_topics 1 \
		--n_unsupervised_topics ${INF}\
		--hidden_units 512 256 \
		--embedding 'glove' ; \
	python evaluate.py \
		--experiment_name 'ng20_supervised' \
		--evaluate_unsupervised 0 \
		--data_dir ${DATA_DIR} \
		--metadata_dir ${METADATA_DIR}

search_ng_20_supervised:
	$(eval EXP_ID:=$(shell python -c "import random; print(random.randint(0, 100000));")) \
	python train.py \
		--experiment_name 'ng20_supervised_'${EXP_ID} \
		--data_dir ${DATA_DIR} \
		--metadata_dir ${METADATA_DIR} \
		--filename ${TRAINING_DATASET_NG20} \
		--output_file 'validation.txt' \
		--model 'logistic_lda' \
		--vocab_file '' \
		--vocab_size 50000 \
		--batch_size $(shell python -c "import random; print(random.choice([ 8, 16, 32]));") \
		--author_topic_weight $(shell python -c "import random; print(random.choice([4000.0, 8000.0]));") \
		--topic_bias_regularization $(shell python -c "import random; print(random.choice([1., 5.]));") \
		--model_regularization $(shell python -c "import random; print(random.choice([0., 5., 20., 50., 100.]));") \
		--initial_learning_rate $(shell python -c "import random; print(random.choice([0.001, 0.003]));") \
		--learning_rate_decay $(shell python -c "import random; print(random.choice([0.8, 0.9, 0.7]));") \
		--max_steps $(shell python -c "import random; print(random.choice([50000, 100000, 200000]));") \
		--hidden_units $(shell python -c "import random; print(random.choice(['1024', '512', '1024 512', '512 256', '256 128']));") \
		--learning_rate_decay_steps 2000 \
		--num_valid 0.15 \
		--items_per_author ${INF} \
		--n_unsupervised_topics ${INF}\
		--n_author_topic_iterations 1 \
		--use_author_topics 1 \
		--embedding 'glove' ; \
	python evaluate.py \
	--experiment_name 'ng20_supervised_'${EXP_ID} \
	--evaluate_unsupervised 0 \
	--data_dir '${DATA_DIR}' \
	--metadata_dir '${METADATA_DIR}' ; \


pinterest_unsupervised:
	python train.py \
		--experiment_name 'pinterest_unsupervised' \
		--data_dir ${DATA_DIR} \
		--metadata_dir ${METADATA_DIR} \
		--filename ${TRAINING_DATASET_PINTEREST} \
		--output_file '' \
		--model 'logistic_lda_online' \
		--vocab_file '' \
		--vocab_size ${INF} \
		--batch_size 4 \
		--author_topic_weight 1. \
		--topic_bias_regularization 1.0 \
		--model_regularization 1000. \
		--initial_learning_rate 0.0001 \
		--learning_rate_decay 0.8 \
		--learning_rate_decay_steps 2000 \
		--max_steps 50000 \
		--num_valid 0 \
		--items_per_author 100 \
		--n_author_topic_iterations 1 \
		--use_author_topics 0 \
		--n_unsupervised_topics 14 \
		--hidden_units 128 \
		--embedding 'mobilenet_v2' ; \
	python evaluate.py \
		--experiment_name 'pinterest_unsupervised' \
		--evaluate_unsupervised 1 \
		--data_dir ${DATA_DIR} \
		--metadata_dir ${METADATA_DIR}

pinterest_supervised:
	python train.py \
		--experiment_name 'pinterest_supervised' \
		--data_dir ${DATA_DIR} \
		--metadata_dir ${METADATA_DIR} \
		--filename ${TRAINING_DATASET_PINTEREST} \
		--output_file 'validation.txt' \
		--model 'logistic_lda_online' \
		--vocab_file '' \
		--vocab_size ${INF} \
		--batch_size 8 \
		--author_topic_weight 8000.0 \
		--topic_bias_regularization 5.0 \
		--model_regularization 0.0 \
		--initial_learning_rate 0.003 \
		--learning_rate_decay 0.7 \
		--learning_rate_decay_steps 2000 \
		--max_steps 100000 \
		--num_valid 2000 \
		--items_per_author 100 \
		--n_author_topic_iterations 1 \
		--use_author_topics 1 \
		--n_unsupervised_topics ${INF} \
		--hidden_units 256 128 \
		--embedding 'mobilenet_v2' ; \
	python evaluate.py \
			--experiment_name 'pinterest_supervised' \
			--evaluate_unsupervised 0 \
			--data_dir ${DATA_DIR} \
			--metadata_dir ${METADATA_DIR}

search_pinterest_supervised:
	$(eval EXP_ID:=$(shell python -c "import random; print(random.randint(0, 100000));")) \
	python train.py \
		--experiment_name 'pinterest_supervised_'${EXP_ID} \
		--data_dir ${DATA_DIR} \
		--metadata_dir ${METADATA_DIR} \
		--filename ${TRAINING_DATASET_PINTEREST} \
		--output_file 'validation.txt' \
		--model 'logistic_lda_lucas_online' \
		--vocab_file '' \
		--vocab_size ${INF} \
		--batch_size $(shell python -c "import random; print(random.choice([8, 16]));") \
		--author_topic_weight $(shell python -c "import random; print(random.choice([4000., 8000.0]));") \
		--topic_bias_regularization $(shell python -c "import random; print(random.choice([1., 5.]));") \
		--model_regularization $(shell python -c "import random; print(random.choice([0., 5., 20., 100.]));") \
		--initial_learning_rate $(shell python -c "import random; print(random.choice([0.003, 0.001]));") \
		--learning_rate_decay $(shell python -c "import random; print(random.choice([0.8, 0.9, 0.7]));") \
		--learning_rate_decay_steps 2000 \
		--max_steps $(shell python -c "import random; print(random.choice([50000, 100000]));") \
		--num_valid 2000 \
		--items_per_author 100 \
		--n_author_topic_iterations $(shell python -c "import random; print(random.choice([1, 4]));") \
		--use_author_topics 1 \
		--n_unsupervised_topics ${INF} \
		--hidden_units $(shell python -c "import random; print(random.choice(['128', '256', '512', '256 128']));") \
		--embedding 'mobilenet_v2' ; \
	python evaluate.py \
		--experiment_name 'pinterest_supervised_'${EXP_ID} \
		--evaluate_unsupervised 0 \
		--data_dir ${DATA_DIR} \
		--metadata_dir ${METADATA_DIR}


pinterest_mlp:
	python train.py \
		--experiment_name 'pinterest_mlp' \
		--data_dir ${DATA_DIR} \
		--metadata_dir ${METADATA_DIR} \
		--filename ${TRAINING_DATASET_PINTEREST} \
		--output_file '' \
		--model 'mlp_online' \
		--vocab_file '' \
		--vocab_size ${INF} \
		--batch_size 16 \
		--author_topic_weight ${INF} \
		--topic_bias_regularization ${INF} \
		--model_regularization 0.1 \
		--initial_learning_rate 0.003 \
		--learning_rate_decay 0.8 \
		--learning_rate_decay_steps 2000 \
		--max_steps 50000 \
		--num_valid 2000 \
		--items_per_author ${INF} \
		--n_author_topic_iterations ${INF} \
		--use_author_topics ${INF} \
		--n_unsupervised_topics ${INF} \
		--hidden_units 256 128 \
		--embedding 'mobilenet_v2' ; \
	python evaluate.py \
		--experiment_name 'pinterest_mlp' \
		--evaluate_unsupervised 0 \
		--data_dir ${DATA_DIR} \
		--metadata_dir ${METADATA_DIR}


search_pinterest_mlp:
	$(eval EXP_ID:=$(shell python -c "import random; print(random.randint(0, 100000));")) \
	python train.py \
		--experiment_name 'pinterest_mlp_'${EXP_ID} \
		--data_dir ${DATA_DIR} \
		--metadata_dir ${METADATA_DIR} \
		--filename ${TRAINING_DATASET_PINTEREST} \
		--output_file 'validation.txt' \
		--model 'mlp_online' \
		--vocab_file '' \
		--vocab_size ${INF} \
		--batch_size $(shell python -c "import random; print(random.choice([8, 16]));") \
		--author_topic_weight ${INF} \
		--topic_bias_regularization ${INF} \
		--model_regularization $(shell python -c "import random; print(random.choice([0., 0.1, 0.01, 1.]));") \
		--initial_learning_rate $(shell python -c "import random; print(random.choice([0.003, 0.001]));") \
		--learning_rate_decay $(shell python -c "import random; print(random.choice([0.8, 0.9, 0.7]));") \
		--learning_rate_decay_steps 2000 \
		--max_steps $(shell python -c "import random; print(random.choice([50000, 100000]));") \
		--num_valid 2000 \
		--items_per_author ${INF} \
		--n_author_topic_iterations ${INF} \
		--use_author_topics ${INF} \
		--n_unsupervised_topics ${INF} \
		--hidden_units $(shell python -c "import random; print(random.choice(['128', '256', '512', '256 128']));") \
		--embedding 'mobilenet_v2' ; \
	python evaluate.py \
		--experiment_name 'pinterest_mlp_'${EXP_ID} \
		--evaluate_unsupervised 0 \
		--data_dir ${DATA_DIR} \
		--metadata_dir ${METADATA_DIR}