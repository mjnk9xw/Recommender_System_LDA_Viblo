import requests
from time import sleep

from tqdm import tqdm
import pymongo
from pymongo import MongoClient
# %%
import logging

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO
# %%
fmt_url = 'https://api.viblo.asia/posts?page=1'
# %%
client = MongoClient('localhost', 27017)
db = client['rsframgia']
col = db['viblo_posts']


req = requests.get(fmt_url)
if req.status_code != 200:
    data = req.json()['data']
    for post in data:
        col.insert_one({
            'id': post['id'],
            'title': post['title'],
            'slug': post['slug'],
            'url': post['url'],
            'content': post['contents']
        })
