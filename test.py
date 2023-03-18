import time
import utils
import pickle

stamp = time.time()

with open(utils.inverted_index_path, "rb") as f:
    inverted_index: dict = pickle.load(f)
elapsed_time = time.time()-stamp

print(elapsed_time)