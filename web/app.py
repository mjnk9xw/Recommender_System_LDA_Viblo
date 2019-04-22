import os
import logging
import random

from flask import Flask, jsonify, render_template
import numpy as np
import pymongo

import settings
from src.distances import get_most_similar_documents
from src.utils import markdown_to_text
from gensim.utils import simple_preprocess

client = pymongo.MongoClient(settings.MONGODB_SETTINGS["host"])
db = client[settings.MONGODB_SETTINGS["db"]]
mongo_col = db[settings.MONGODB_SETTINGS["collection"]]

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "framgia123")

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

def make_texts_corpus(sentences):
    for sentence in sentences:
        yield simple_preprocess(sentence, deacc=True)

# load file models
def load_model():
    import gensim  # noqa
    from sklearn.externals import joblib  # noqa
    # load LDA model
    lda_model = gensim.models.LdaModel.load(
        settings.PATH_LDA_MODEL
    )
    # load corpus
    corpus = gensim.corpora.MmCorpus(
        settings.PATH_CORPUS
    )
    # load dictionary
    id2word = gensim.corpora.Dictionary.load(
        settings.PATH_DICTIONARY
    )
    # load documents topic distribution matrix
    doc_topic_dist = joblib.load(
        settings.PATH_DOC_TOPIC_DIST
    )
    # doc_topic_dist = np.array([np.array(dist) for dist in doc_topic_dist])

    return lda_model, corpus, id2word, doc_topic_dist


lda_model, corpus, id2word, doc_topic_dist = load_model()


@app.route('/ping', methods=['GET'])
def ping_pong():
    return jsonify({
        'call': 'success',
        'message': 'pong!'
    })


@app.route('/posts/', methods=["GET"])
def show_posts():
    idrss = random.sample(range(0, mongo_col.count()), 10)
    posts = mongo_col.find({"idrs": {"$in": idrss}})
    random_posts = [
        {
            "idrs": post["idrs"],
            "url": post["url"],
            "title": post["title"],
            "slug": post["slug"]
        }
        for post in posts
    ]
    return render_template('index.html', random_posts=random_posts)


@app.route('/posts/<slug>', methods=["GET"])
def show_post(slug):
    main_post = mongo_col.find_one({"slug": slug})
    main_post = {
        "url": main_post["url"],
        "title": main_post["title"],
        "slug": main_post["slug"],
        "content": main_post["content"]
    }

    # preprocessing
    content = markdown_to_text(main_post["content"])
    text_corpus = make_texts_corpus([content])
    bow = id2word.doc2bow(next(text_corpus))
    # sử dụng dictionary và LDA model đã train và lưu lại để thu được vector document_dist, ứng với phân bố các topic của document đó
    doc_distribution = np.array(
        [doc_top[1] for doc_top in lda_model.get_document_topics(bow=bow)]
    )

    # recommender posts
    most_sim_ids = list(get_most_similar_documents(
        doc_distribution, doc_topic_dist))[1:]

    most_sim_ids = [int(id_) for id_ in most_sim_ids]
    posts = mongo_col.find({"idrs": {"$in": most_sim_ids}})
    related_posts = [
        {
            "url": post["url"],
            "title": post["title"],
            "slug": post["slug"]
        }
        for post in posts
    ][1:]

    return render_template(
        'index.html', main_post=main_post, posts=related_posts
    )


if __name__ == "__main__":
    app.run(host='127.0.0.1', debug=False)
