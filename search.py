import utils
from nltk.tokenize import word_tokenize
import pickle
from collections import defaultdict
import time
import os
import json
import build_index

with open(os.path.join(build_index.INDEX_PATH, "skip_tokens.json"), "r") as f:
    skip_tokens: dict = json.load(f)

def process_query(query):
    query = utils.clean_document(query)
    query_tokens = word_tokenize(query)
    query_tokens = utils.stem(query_tokens)

    return query_tokens

def retrieve(query:str, n:int=1):
    stamp = time.time()
    tokenized_query = process_query(query)
    sorted_accumulator = ranking(tokenized_query)

    elapsed_time = time.time()-stamp
    if elapsed_time > 0.3:
        print("Warning! This retrieval takes more than 300 ms!")
    print(f"It takes {elapsed_time} seconds to fetch {len(sorted_accumulator)} results.")

    return list(sorted_accumulator)

def calculate_final_score(tfidf_score, important_multiplier):
    score = tfidf_score * important_multiplier
    return score

def check_loc(loc_set, loc):
    for loc_list in loc_set:
        if loc_list.intersection(loc):
            return True, 1
    else:
        return False, 1

def ranking(tokenized_query):
    posting_list = []
    max_posting_num = 500
    url_set_list = []
    loc_set = []

    for token in tokenized_query:
        token_shard_num = build_index.get_token_shard_num(skip_tokens, token)
        token_shard_path = os.path.join(build_index.INDEX_PATH, f"token_shard{token_shard_num}.json")
        with open(token_shard_path, "r") as f:
            inverted_index =  json.load(f)
        posting_list += inverted_index[token][:max_posting_num]

        url_list = list(zip(*posting_list))[0]
        url_set_list.append(set(url_list))
    
    common_url_list = set.intersection(*url_set_list)

    accumulator = defaultdict(float)
    for posting in posting_list:
        (url, loc, tfidf_score, important_multiplier) = posting
        
        
        if url in common_url_list:
            final_score = calculate_final_score(tfidf_score, important_multiplier)
            """
            check, score = check_loc(loc_set, loc)
            if check:
                loc_score = 1.2
            else:
                loc_score = 1
            
            loc_score *= 1.5
            """

            accumulator[url] = accumulator[url] + final_score
        
        loc_set.append(loc)
        
    sorted_accumulator = sorted(accumulator.items(), key=lambda x: x[1], reverse=True)
    return sorted_accumulator

if __name__ == "__main__":
    for idx in range(20):
        file_name = f"q{idx + 1}.txt"
        with open(os.path.join("tests", file_name)) as f:
            test_query = f.read()
        sorted_accumulator = retrieve(test_query)
        if sorted_accumulator:
            urls, _ = zip(*sorted_accumulator)
        else:
            urls = []
        
        with open(os.path.join("tests", f"q{idx + 1}_result.txt"), "w") as f:
            topk = 10

            for i in range(topk):
                url = urls[i]
                f.write(f"{i}: {url}\n")





