import base64
import numpy as np
import cv2
from flask_restx import Namespace
from flask import request, jsonify
from interface.face.serializers import face_field, face_embed_field
from flask_restx import Resource
from framework.utilities import HTTP_CODE
from utilities.helper import face_helper

face_api = Namespace("vision")


@face_api.route("/face_matching")
class FaceAPI(Resource):
    @face_api.doc("Face matching",
            description="Face Matching",
            responses={
                HTTP_CODE["HTTP_OK"]: "OK",
                HTTP_CODE["HTTP_BAD_REQUEST"]: "Invalid configuration",
                HTTP_CODE["HTTP_SERVER_ERROR"]: "Internal processing error"
            })
    @face_api.expect(face_api.model("face_field", face_field), validate=True)
    def post(self):
        request_json = request.get_json()

        face1 = request_json["face1"]
        face2 = request_json["face2"]

        # Recreate image from base64 text
        face1_origin = base64.b64decode(face1)
        face2_origin = base64.b64decode(face2)

        face1 = np.frombuffer(face1_origin, dtype=np.uint8)
        face1 = cv2.imdecode(face1, flags=1)

        face2 = np.frombuffer(face2_origin, dtype=np.uint8)
        face2 = cv2.imdecode(face2, flags=1)

        dist = face_helper.match(face1,face2)

        is_match = False
        similarity = 1 - (dist*dist/4)
        if similarity[0] > 0.7:
            is_match = True
        return (
            {
                "message": "OK",
                "data": [{
                    "confidence": similarity[0],
                    "is_match": is_match
                }]
            }
        )


@face_api.route("/face_embed")
class FaceEmbedAPI(Resource):
    @face_api.doc("Face feature extraction",
            description="Extract face feature",
            responses={
                HTTP_CODE["HTTP_OK"]: "OK",
                HTTP_CODE["HTTP_BAD_REQUEST"]: "Invalid configuration",
                HTTP_CODE["HTTP_SERVER_ERROR"]: "Internal processing error"
            })
    @face_api.expect(face_api.model("face_embed_field", face_embed_field), validate=True)
    def post(self):
        request_json = request.get_json()

        face1 = request_json["face"]

        # Recreate image from base64 text
        face1_origin = base64.b64decode(face1)

        face1 = np.frombuffer(face1_origin, dtype=np.uint8)
        face1 = cv2.imdecode(face1, flags=1)

        feature = face_helper.extract_feature(face1)

        return (
            {
                "message": "OK",
                "data": {
                    "feature": feature.tolist()
                }
            }
        )

