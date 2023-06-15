from flask_restx import Namespace
from flask_restx import Resource
from framework.utilities import HTTP_CODE
import sys, signal
from flask import request
from interface.ensemble.serializers import inference_inputs
from utilities.helper import parse_sentences
from utilities.ensemble.utils import ensemble_model

classify_api = Namespace("nlp")


# @classify_api.route("/voting")
# class VotingAPI(Resource):
#     @classify_api.doc("Ensemble Voting",
#                       description="Ensemble Voting - Inference",
#                       responses={
#                           HTTP_CODE["HTTP_OK"]: "OK",
#                           HTTP_CODE["HTTP_BAD_REQUEST"]: "Invalid configuration",
#                           HTTP_CODE["HTTP_SERVER_ERROR"]: "Internal processing error"
#                       })
#     @classify_api.expect(inference_inputs(classify_api.parser()), validate=True)
#     def get(self):
#         input_sentences = parse_sentences(request.args.get('input'))
#         language = request.args.get('language')
#         output_type = request.args.get('output_type')
#         outputs = ensemble_model.inference(input_sentences, language, output_type, ensemble_model.voting)

#         datas = []
#         for output in outputs:
#             datas.append(output.get("ensemble_output"))
#         return (
#             {
#                 "message": "OK",
#                 "data": datas,
#             }
#         )


# @classify_api.route("/stacking")
@classify_api.route("/is_buy")
class StackingAPI(Resource):
    @classify_api.doc("Classify buying intent",
                      description="Classify buying intent from text",
                      responses={
                          HTTP_CODE["HTTP_OK"]: "OK",
                          HTTP_CODE["HTTP_BAD_REQUEST"]: "Invalid configuration",
                          HTTP_CODE["HTTP_SERVER_ERROR"]: "Internal processing error"
                      })
    @classify_api.expect(inference_inputs(classify_api.parser()), validate=True)
    def get(self):
        input_sentences = parse_sentences(request.args.get('input'))
        language = request.args.get('language')
        # output_type = request.args.get('output_type')
        output_type = "probability"
        outputs = ensemble_model.inference(input_sentences, language, output_type, ensemble_model.stacking)

        datas = []
        for output in outputs:
            datas.append(output.get("ensemble_output"))

        return (
            {
                "message": "OK",
                "data": {
                  "confidence": datas
                }
            }
        )
