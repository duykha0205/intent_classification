from flask import Flask
from flask_restx import Api
import os
# from interface.ensemble.restful import ensemble_api
# from interface.bert.restful import bert_api
# from interface.fasttext.restful import fasttext_api
# from interface.fasttext_vi.restful import fasttext_vi_api
# from interface.distilbert.restful import distilbert_api
from interface.classifyapi.restful import classify_api
# from interface.phobert.restful import  phobert_api
from interface.face.restful import  face_api


app = Flask(__name__)
api = Api(app)

# api.add_namespace(ensemble_api)
# api.add_namespace(bert_api)
# api.add_namespace(phobert_api)
# api.add_namespace(fasttext_api)
# api.add_namespace(fasttext_vi_api)
# api.add_namespace(distilbert_api)
api.add_namespace(classify_api)
api.add_namespace(face_api)
