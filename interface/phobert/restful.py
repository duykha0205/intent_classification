from flask_restx import Namespace
from flask_restx import Resource
from framework.utilities import HTTP_CODE
from flask import request
from interface.phobert.serializers import inference_inputs
from utilities.helper import parse_sentences
from utilities.phobert.utils import phobert_model

phobert_api = Namespace("phobert")


@phobert_api.route("")
class PhoBertAPI(Resource):
    @phobert_api.doc("PhoBert",
                  description="PhoBert - Inference",
                  responses={
                    HTTP_CODE["HTTP_OK"]: "OK",
                    HTTP_CODE["HTTP_BAD_REQUEST"]: "Invalid configuration",
                    HTTP_CODE["HTTP_SERVER_ERROR"]: "Internal processing error"
                  })
    @phobert_api.expect(inference_inputs(phobert_api.parser()), validate=True)
    def get(self):
        input_sentences = parse_sentences(request.args.get('input'))
        output_type = request.args.get('output_type')
        output = phobert_model.inference(input_sentences, output_type)

        return (
            {   
                "message": "OK",
                "data": output,
            }
        )
