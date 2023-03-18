import os
import json
from os.path import join
import utils
import tqdm
from collections import defaultdict
import pandas as pd
from simhash import Simhash, SimhashIndex


corpus_path = "pageset"
URL_SHARD_SIZE = 5000
TOKEN_SHARD_SIZE = 5000
INDEX_PATH = "index"

def load_doc_file_list():
    doc_list = []
    
    for (root, dirs, files) in os.walk(corpus_path):
        doc_list += [join(root, name) for name in files]

    return doc_list

def load_tfidf(doc_file_list):
    vectorizer_file_name = "tfidfmatrix.pkl"
    if os.path.exists(vectorizer_file_name):
        df = pd.read_pickle(vectorizer_file_name)
        print("tf-idf score matrix loaded")
    else:
        df = utils.get_tf_idf(doc_file_list)
        df.to_pickle(vectorizer_file_name)
    return df

def build_inverted_index_shard(doc_file_list, df):
    inverted_index = defaultdict(list)
    simhash_index = SimhashIndex([], k=3)
    token_set = set()
    duplicate_count = 0
    url_count = 0
    shard_num = 0

    for doc in tqdm.tqdm(doc_file_list, desc='Building inverted index'):
        with open(doc, "r") as f:
            data = json.load(f)
            url = data["url"]
            content = data["content"]
            encoding = data["encoding"]

            if url in utils.filtered_list:
                continue

            s = Simhash(utils.get_simhash_features(content))
            if simhash_index.get_near_dups(s):
                duplicate_count += 1
                continue
            else:
                simhash_index.add(url, s)

            stemmed_word_tokens = utils.get_tokens(utils.html2text(content))
            important_word_dict = utils.get_important_word_dict(content)

            loc_dict = get_loc_dict(stemmed_word_tokens)
            
            # for (token, loc_list) in loc_dict.items():
            for loc_list, token in enumerate(stemmed_word_tokens):
                try:
                    tfidf_score = df.loc[url, token]

                    important_multiplier = get_important_multiplier(important_word_dict, token)

                    posting = (url, loc_list, tfidf_score, important_multiplier)
                    inverted_index[token].append(posting)
                    token_set.add(token)
                except:
                    pass

            url_count += 1
        
            if url_count != 0 and url_count % URL_SHARD_SIZE == 0:
                shard_path = os.path.join(INDEX_PATH, f"shard{shard_num}.json")
                with open(shard_path, "w") as f:
                    json.dump(inverted_index, f)
                
                shard_num += 1
                inverted_index = defaultdict(list)
    
    if url_count % URL_SHARD_SIZE != 0:
        shard_path = os.path.join(INDEX_PATH, f"shard{shard_num}.json")
        with open(shard_path, "w") as f:
            json.dump(inverted_index, f)
        
        shard_num += 1

    print("------------------------------------------------result------------------------------------------------")
    print(f"The number of duplicate pages: {duplicate_count}")
    print(f"The number of filtered urls: {len(utils.filtered_list)}")
    print(f"The number of indexed tokens: {len(token_set)}")
    print(f"The number of indexed urls: {url_count}")
    print(f"The number of shards: {shard_num}")
    print(f"The size of shards: {URL_SHARD_SIZE}")
    print("------------------------------------------------------------------------------------------------------")

    return shard_num, token_set 

def get_important_multiplier(important_word_dict, token):
    important_multiplier = 1
    for (multiplier, token_list) in important_word_dict.items():
        if token in token_list:
            important_multiplier = max(important_multiplier, multiplier)
    return important_multiplier

def get_loc_dict(stemmed_word_tokens):
    loc_dict = defaultdict(list)
    for loc, token in enumerate(stemmed_word_tokens):
        loc_dict[token].append(loc)
    return loc_dict

def get_token_shard_num(skip_tokens, token):
    for idx, skip_token in enumerate(skip_tokens):
        if skip_token > token:
            shard_num = idx - 1
            break
    else:
        shard_num = len(skip_tokens) - 1
    return shard_num

if __name__ == "__main__":
    doc_file_list = load_doc_file_list()
    df = load_tfidf(doc_file_list)
    
    print("step1: build shards")
    if True:
        shard_num, token_set = build_inverted_index_shard(doc_file_list, df)
        sorted_token_list = sorted(token_set)
        with open(os.path.join(INDEX_PATH, "tokens.json"), "w") as f:
            json.dump(sorted_token_list, f)
    else:
        shard_num = 7
        with open(os.path.join(INDEX_PATH, "tokens.json"), "r") as f:
            sorted_token_list: dict = json.load(f)

    print("step2: merge shards")
    skip_tokens = sorted_token_list[::TOKEN_SHARD_SIZE]
    with open(os.path.join(INDEX_PATH, "skip_tokens.json"), "w") as f:
        json.dump(skip_tokens, f)
    token_shards = []
    for _ in range(len(skip_tokens)):
        token_shards.append(defaultdict(list))

    for idx in tqdm.trange(shard_num):
        shard_path = os.path.join(INDEX_PATH, f"shard{idx}.json")
        with open(shard_path, "r") as f:
            inverted_index: dict = json.load(f)
        
        for (token, posting_list) in tqdm.tqdm(inverted_index.items()):
            token_shard_num = get_token_shard_num(skip_tokens, token)
            # FIXME: this should be list concat
            token_shards[token_shard_num][token] += posting_list
            
    for token_shard_num in tqdm.trange(len(skip_tokens)):
        token_shard_path = os.path.join(INDEX_PATH, f"token_shard{token_shard_num}.json")
        for (token, posting_list) in token_shards[token_shard_num].items():
            token_shards[token_shard_num][token] = sorted(posting_list, key=lambda x: x[2], reverse=True)

        with open(token_shard_path, "w") as f:
            json.dump(token_shards[token_shard_num], f)
    
    print("------------------------------------------------result------------------------------------------------")
    print(f"The size of token shard: {TOKEN_SHARD_SIZE}")
    print(f"The number of token shards: {len(skip_tokens)}")
    print("------------------------------------------------------------------------------------------------------")
            







    
    
    
    
    