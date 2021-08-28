import argparse
import otbApplication
import constants
import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.WARNING,
                    datefmt='%Y-%m-%d %H:%M:%S')

encodings = {"unsigned_char": otbApplication.ImagePixelType_uint8,
             "short": otbApplication.ImagePixelType_int16,
             "unsigned_short": otbApplication.ImagePixelType_uint16,
             "int": otbApplication.ImagePixelType_int32,
             "unsigned_int": otbApplication.ImagePixelType_uint32,
             "float": otbApplication.ImagePixelType_float,
             "double": otbApplication.ImagePixelType_double}

parser = argparse.ArgumentParser()
parser.add_argument("--input", help="Input LR image. Must be in the same dynamic as the lr_patches used in the "
                                    "train.py application.", required=True)
parser.add_argument("--savedmodel", help="Input SavedModel (provide the path to the folder).", required=True)
parser.add_argument("--output", help="Output HR image", required=True)
parser.add_argument('--encoding', type=str, default="auto", const="auto", nargs="?", choices=encodings.keys(),
                    help="Output HR image encoding")
parser.add_argument('--pad', type=int, default=64, const=64, nargs="?", choices=constants.pads,
                    help="Margin size for blocking artefacts removal")
parser.add_argument('--ts', default=512, type=int, help="Tile size. Tune this to process larger output image chunks, "
                                                        "and speed up the process.")
params = parser.parse_args()


def get_encoding_name():
    """
    Get the encoding of input image pixels
    """
    infos = otbApplication.Registry.CreateApplication('ReadImageInfo')
    infos.SetParameterString("in", params.input)
    infos.Execute()
    return infos.GetParameterString("datatype")


if __name__ == "__main__":

    gen_fcn = params.pad
    efield = params.ts  # OTBTF expression field
    if efield % min(constants.factors) != 0:
        logging.fatal("Please chose a tile size that is consistent with the network.")
        quit()
    ratio = 1.0 / float(max(constants.factors))  # OTBTF Spacing ratio
    rfield = int((efield + 2 * gen_fcn) * ratio)  # OTBTF receptive field

    # pixel encoding
    encoding = params.encoding
    if encoding == "auto":
        encoding = encodings[get_encoding_name()]

    # call otbtf
    logging.info("Receptive field: {}, Expression field: {}".format(rfield, efield))
    ph = "{}{}".format(constants.outputs_prefix, params.pad)
    infer = otbApplication.Registry.CreateApplication("TensorflowModelServe")
    infer.SetParameterStringList("source1.il", [params.input])
    infer.SetParameterInt("source1.rfieldx", rfield)
    infer.SetParameterInt("source1.rfieldy", rfield)
    infer.SetParameterString("source1.placeholder", constants.lr_input_name)
    infer.SetParameterString("model.dir", params.savedmodel)
    infer.SetParameterString("model.fullyconv", "on")
    infer.SetParameterStringList("output.names", [ph])
    infer.SetParameterInt("output.efieldx", efield)
    infer.SetParameterInt("output.efieldy", efield)
    infer.SetParameterFloat("output.spcscale", ratio)
    infer.SetParameterInt("optim.tilesizex", efield)
    infer.SetParameterInt("optim.tilesizey", efield)
    infer.SetParameterInt("optim.disabletiling", 1)
    out_fn = "{}{}&gdal:co:COMPRESS=DEFLATE".format(params.output, "?" if "?" not in params.output else "")
    out_fn += "&streaming:type=tiled&streaming:sizemode=height&streaming:sizevalue={}".format(efield)
    infer.SetParameterString("out", out_fn)
    infer.SetParameterOutputImagePixelType("out", encoding)
    infer.ExecuteAndWriteOutput()
