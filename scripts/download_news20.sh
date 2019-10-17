mkdir -p data
cd data

# download data
curl -OL "https://storage.googleapis.com/topicmodeling/tokenized_ng20_test.pkl"
curl -OL "https://storage.googleapis.com/topicmodeling/tokenized_ng20_train.pkl"

# download vocabulary
curl -OL "https://storage.googleapis.com/topicmodeling/vocab2K.pkl"

# download GLoVe embeddings
curl -OL "http://nlp.stanford.edu/data/glove.6B.zip"
unzip glove.6B.zip
