from flask_restx import Namespace
from flask_restx import Resource
from framework.utilities import HTTP_CODE
import sys, signal
from flask import request
from interface.fasttext.serializers import inference_inputs
from utilities.helper import parse_sentences
from utilities.distilbert.utils import distilbert_model

distilbert_api = Namespace("distilbert")


@distilbert_api.route("")
class DistilBertAPI(Resource):
    @distilbert_api.doc("DistilBert",
                  description="DistilBert - Inference",
                  responses={
                    HTTP_CODE["HTTP_OK"]: "OK",
                    HTTP_CODE["HTTP_BAD_REQUEST"]: "Invalid configuration",
                    HTTP_CODE["HTTP_SERVER_ERROR"]: "Internal processing error"
                  })
    @distilbert_api.expect(inference_inputs(distilbert_api.parser()), validate=True)
    def get(self):
        input_sentences = parse_sentences(request.args.get('input'))
        output_type = request.args.get('output_type')
        output = distilbert_model.inference(input_sentences, output_type)

        return (
            {
                "message": "OK",
                "data": output,
            }
        )
