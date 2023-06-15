from flask_restx import Namespace
from flask import request
from interface.fasttext_vi.serializers import inference_inputs

from flask_restx import Resource
from framework.utilities import HTTP_CODE
from utilities.fasttext_vi.utils import fasttext_vi_model
from utilities.helper import parse_sentences

fasttext_vi_api = Namespace("fasttext-vi")


@fasttext_vi_api.route("")
class FasttextViAPI(Resource):
    @fasttext_vi_api.doc("Fasttext Vietnamese",
            description="Fasttext Vietnamese - Inference ",
            responses={
                HTTP_CODE["HTTP_OK"]: "OK",
                HTTP_CODE["HTTP_BAD_REQUEST"]: "Invalid configuration",
                HTTP_CODE["HTTP_SERVER_ERROR"]: "Internal processing error"
            })
    @fasttext_vi_api.expect(inference_inputs(fasttext_vi_api.parser()), validate=True)
    def get(self):
        input_sentences = parse_sentences(request.args.get('input'))
        output_type = request.args.get('output_type')

        output = fasttext_vi_model.inference(input_sentences, output_type)

        return (
            {
                "message": "OK",
                "data": output,
            }
        )
