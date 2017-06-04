'''
---------------------------------------
dataset file path
---------------------------------------
'''
data_path_amazon_movies = './data/reviews_Movies_and_TV_5.json.gz'
data_path_amazon_office = './data/reviews_Office_Products_5.json.gz'
data_amazon_columns = ["reviewerID", "asin", "reviewerName", "helpful", 
                       "reviewText", "overall", "summary", "unixReviewTime", "reviewTime"]

data_path_yelp_reviews = "/Users/lucas/Desktop/Code/DL/recsys/data/" \
    "yelp_dataset_challenge_round9/yelp_academic_dataset_review.json"
data_path_yelp_users = "/Users/lucas/Desktop/Code/DL/recsys/data/" \
    "yelp_dataset_challenge_round9/yelp_academic_dataset_user.json"
data_yelp_columns = ["business_id", "cool", "date", "funny", "review_id", 
                     "stars", "text", "type", "useful", "user_id"]

'''
---------------------------------------
load dataset
---------------------------------------
'''
import gzip
from pprint import pprint
from model import JsonDatasrc

print("loading dataset...")

gz_open_func = lambda file_path: gzip.open(file_path, 'r')

amazon_data_json = JsonDatasrc(data_path_amazon_office, gz_open_func, eval)

print(len(amazon_data_json.raw_json))
pprint(amazon_data_json.raw_json[0])

amazon_data_json.describe(["helpful"])
amazon_data_json.count_by_col("reviewerID")
amazon_data_json.count_by_col("asin")

print("done.")

'''
---------------------------------------
load embeddings and build
---------------------------------------
'''
from model import NlpUtil, TextCollection

print("loading pretrained word2vec...")

pretrained_word2vec300_file_path = "./data/GoogleNews-vectors-negative300.bin"
embeds_word2vec300 = NlpUtil.load_word2vec(pretrained_word2vec300_file_path)

print("done.")
print("building index and word embeddings...")

amazon_data_text_collection = TextCollection(amazon_data_json, ["reviewerID", "asin"], 
                            ["overall", "reviewTime", "reviewText", "summary"])
amazon_data_text_collection.build_key_index()
amazon_data_text_collection.build_embeddings(embeds_word2vec300, "summary")
#amazon_data_text_collection.build_embeddings(embeds_word2vec300, "reviewText")

print("done")

collection = amazon_data_text_collection
collection.target_name = "overall"