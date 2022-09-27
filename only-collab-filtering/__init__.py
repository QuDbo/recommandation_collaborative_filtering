import logging
import numpy as np
import pandas as pd
import azure.functions as func
import json
from io import BytesIO
import implicit
import scipy.sparse as sparse
import time

def load_model_implicit(input_blob_1,input_blob_2,input_blob_3):
    blob_bytes = input_blob_1.read()
    blob_stream = BytesIO(blob_bytes)
    blob_stream.seek(0)
    full_blob = blob_stream.read()
    logging.info("blob 1 read")
    blob_bytes = input_blob_2.read()
    blob_stream = BytesIO(blob_bytes)
    blob_stream.seek(0)
    full_blob += blob_stream.read()
    logging.info("blob 2 read")
    blob_bytes = input_blob_3.read()
    blob_stream = BytesIO(blob_bytes)
    blob_stream.seek(0)
    full_blob += blob_stream.read()
    logging.info("blob 3 read")
    logging.info(f"full blob : {type(full_blob)}")
    in_memory = BytesIO(full_blob)
    in_memory.seek(0)
    logging.info("file write")
    model = implicit.cpu.als.AlternatingLeastSquares.load(in_memory)
    logging.info(f"{model}")
    return model

def load_corr_article(input_blob):
    blob_bytes = input_blob.read()
    blob_stream = BytesIO(blob_bytes)
    blob_stream.seek(0)
    corr = pd.read_csv(blob_stream)
    return dict(zip(corr['article_id'],corr['click_article_id']))


def load_sparse_matrix(input_blob):
    blob_bytes = input_blob.read()
    blob_stream = BytesIO(blob_bytes)
    blob_stream.seek(0)
    sparse_matrix = sparse.load_npz(blob_stream)
    return sparse_matrix

def calculate_CF_reco(model,u_id,u_i_sparse,corr):
    recommended = model.recommend(u_id, u_i_sparse[u_id],
                              N=5,filter_already_liked_items=True,
                             )
    articles_idx = recommended[0]
    articles_recommended = {}
    scores = recommended[1]

    for ii,(id_article,score) in enumerate(zip(articles_idx,scores)):
        articles_recommended[ii+1] = {
            'article':corr[id_article],
            'score':score.astype(float)
        }
    return articles_recommended

def main(req: func.HttpRequest,
         modelimplicitpart1: func.InputStream,
         modelimplicitpart2: func.InputStream,
         modelimplicitpart3: func.InputStream,
         useritemsparse: func.InputStream,
         corrarticle: func.InputStream,
        ) -> func.HttpResponse:
    logging.info("Python HTTP trigger collaborative filtering recommandation function processed a request.")

    report_time = {}
    u_id = int(req.route_params.get('u_id'))
    art_id = int(req.route_params.get('art_id'))

    # Get collaborative filtering recommandation list

    t0 = time.time()
    logging.info("Go into load implicit model")
    model = load_model_implicit(modelimplicitpart1,
                                modelimplicitpart2,
                                modelimplicitpart3,
                               )
    dt = time.time() - t0
    report_time['load_model_implicit'] = dt
    
    t0 = time.time()
    user_item_sparse = load_sparse_matrix(useritemsparse)
    dt = time.time() - t0
    report_time['load_sparse_matrix'] = dt
    
    t0 = time.time()
    corr_article = load_corr_article(corrarticle)
    dt = time.time() - t0
    report_time['load_correspondance_article'] = dt
    
    t0 = time.time()
    best_article_CF = calculate_CF_reco(model,
                                        u_id,
                                        user_item_sparse,
                                        corr_article
                                       )
    dt = time.time() - t0
    report_time['compute_collaborative_filtering'] = dt

    return func.HttpResponse(json.dumps({'cf':best_article_CF,'times':report_time}))