import sys
from typing import Iterable, Dict

import numpy as np
import open3d as o3d

import arithmeticcoding
import context_model
import simple_pcc


def main(args):
    # Handle command line arguments
    if len(args) != 3:
        sys.exit(
            "Usage: python cxt_arith_decompress.py InputFile InputContextFile OutputFile"
        )

    input_file, context_file, output_file = args
    with open(input_file, "rb") as input_data, open(
        context_file, "rb"
    ) as input_context:
        bitin_data = arithmeticcoding.BitInputStream(input_data)
        bitin_context = arithmeticcoding.BitInputStream(input_context)
        decoded_bytes = decompress_to_bytes(bitin_data, bitin_context)

        points = simple_pcc.Decoder.bytes_to_points(decoded_bytes, depth=8)
        point_cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(list(points)))
        o3d.io.write_point_cloud(output_file, point_cloud)


def decompress_to_bytes(
    bitin_data: arithmeticcoding.BitInputStream,
    bitin_context: arithmeticcoding.BitInputStream,
) -> Iterable[int]:
    context_num = 9
    all_context = tuple(range(context_num))
    symbol_num = 0x100  # = 2^8
    # Init the data Decoder, just like we do in Encoder {{{
    pos_to_freqs: Dict[int, arithmeticcoding.SimpleFrequencyTable] = {}
    for pos in all_context:
        init_freq_table = None
        if pos != 0:
            init_freq_table = arithmeticcoding.FlatFrequencyTable(symbol_num)
        else:
            init_freq_table = arithmeticcoding.FlatFrequencyTable(symbol_num + 1)
        freq_table = arithmeticcoding.SimpleFrequencyTable(init_freq_table)
        pos_to_freqs[pos] = freq_table
    model = context_model.ContextModel(all_context, pos_to_freqs)
    decoder_data = arithmeticcoding.ArithmeticDecoder(32, bitin_data)
    # }}}

    # Init the context Decoder {{{
    init_freq_table = arithmeticcoding.FlatFrequencyTable(context_num)
    context_freq_table = arithmeticcoding.SimpleFrequencyTable(init_freq_table)
    decoder_context = arithmeticcoding.ArithmeticDecoder(32, bitin_context)
    # }}}

    # 0x100
    eof_symbol = symbol_num

    while True:
        symbol_context = decoder_context.read(context_freq_table)
        symbol_data = decoder_data.read(model.get(symbol_context))

        if (symbol_context == 0) and (symbol_data == eof_symbol):
            break
        yield symbol_data

        context_freq_table.increment(symbol_context)
        model.increment(symbol_context, symbol_data)


if __name__ == "__main__":
    main(sys.argv[1:])
