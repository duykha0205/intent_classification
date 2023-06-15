from flask_restx import Namespace
from flask_restx import Resource
from framework.utilities import HTTP_CODE
import sys, signal
from flask import request
from interface.ensemble.serializers import inference_inputs
from utilities.helper import parse_sentences
from utilities.ensemble.utils import ensemble_model

ensemble_api = Namespace("ensemble")


@ensemble_api.route("/voting")
class VotingAPI(Resource):
    @ensemble_api.doc("Ensemble Voting",
                      description="Ensemble Voting - Inference",
                      responses={
                          HTTP_CODE["HTTP_OK"]: "OK",
                          HTTP_CODE["HTTP_BAD_REQUEST"]: "Invalid configuration",
                          HTTP_CODE["HTTP_SERVER_ERROR"]: "Internal processing error"
                      })
    @ensemble_api.expect(inference_inputs(ensemble_api.parser()), validate=True)
    def get(self):
        input_sentences = parse_sentences(request.args.get('input'))
        language = request.args.get('language')
        output_type = request.args.get('output_type')

        output = ensemble_model.inference(input_sentences, language, output_type,  ensemble_model.voting)


        return (
            {
                "message": "OK",
                "data": output,
            }
        )


@ensemble_api.route("/stacking")
class StackingAPI(Resource):
    @ensemble_api.doc("Ensemble Stacking",
                      description="Ensemble Stacking - Inference",
                      responses={
                          HTTP_CODE["HTTP_OK"]: "OK",
                          HTTP_CODE["HTTP_BAD_REQUEST"]: "Invalid configuration",
                          HTTP_CODE["HTTP_SERVER_ERROR"]: "Internal processing error"
                      })
    @ensemble_api.expect(inference_inputs(ensemble_api.parser()), validate=True)
    def get(self):
        input_sentences = parse_sentences(request.args.get('input'))
        language = request.args.get('language')
        # output_type = request.args.get('output_type')
        output_type = "probability"
        output = ensemble_model.inference(input_sentences, language, output_type,  ensemble_model.stacking)

        return (
            {
                "message": "OK",
                "data": {
                  "confidence": output
                },
            }
        )
