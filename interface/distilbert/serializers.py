def inference_inputs(parser):
    parser.add_argument(
        "input",
        type=str, required=False,
        help="Input to inference"
    )
    parser.add_argument(
        "output_type",
        type=str, required=False,
        help="label/probability",
        default="label"
    )

    return parser
