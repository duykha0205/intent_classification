from flask_restx import Namespace
from flask import request
from interface.bert.serializers import inference_inputs

from flask_restx import Resource
from framework.utilities import HTTP_CODE
from utilities.bert.utils import bert_model
from utilities.helper import parse_sentences

bert_api = Namespace("bert")


@bert_api.route("")
class BertAPI(Resource):
    @bert_api.doc("Create new stream for specified user",
                  description="BERT - Inference ",
                  responses={
                      HTTP_CODE["HTTP_OK"]: "OK",
                      HTTP_CODE["HTTP_BAD_REQUEST"]: "Invalid configuration",
                      HTTP_CODE["HTTP_SERVER_ERROR"]: "Internal processing error"
                  })
    @bert_api.expect(inference_inputs(bert_api.parser()), validate=True)
    def get(self):
        output_type = request.args.get('output_type')
        output = bert_model.inference(parse_sentences(request.args.get('input')), output_type)

        return (
            {
                "message": "OK",
                "data": output,
            }
        )
