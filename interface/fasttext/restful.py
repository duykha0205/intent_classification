from flask_restx import Namespace
from flask import request
from interface.fasttext.serializers import inference_inputs

from flask_restx import Resource
from framework.utilities import HTTP_CODE
from utilities.fasttext.utils import fasttext_model
from utilities.helper import parse_sentences

fasttext_api = Namespace("fasttext")


@fasttext_api.route("")
class FasttextAPI(Resource):
    @fasttext_api.doc("Fasttext",
            description="Fasttext - Inference ",
            responses={
                HTTP_CODE["HTTP_OK"]: "OK",
                HTTP_CODE["HTTP_BAD_REQUEST"]: "Invalid configuration",
                HTTP_CODE["HTTP_SERVER_ERROR"]: "Internal processing error"
            })
    @fasttext_api.expect(inference_inputs(fasttext_api.parser()), validate=True)
    def get(self):
        input_sentences = parse_sentences(request.args.get('input'))
        output_type = request.args.get('output_type')

        output = fasttext_model.inference(input_sentences, output_type)

        return (
            {
                "message": "OK",
                "data": output,
            }
        )
