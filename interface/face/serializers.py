from flask_restx import fields

face_field = {
    "face1": fields.String(
        required=True, description="Base64 image face 1", help="User name cannot be blank!",
    ),
    "face2": fields.String(
        required=True, description="Base64 image face 2", help="User name cannot be blank!",
    )
}

face_embed_field = {
    "face": fields.String(
        required=True, description="Base64 image face 1", help="User name cannot be blank!",
    )
}

