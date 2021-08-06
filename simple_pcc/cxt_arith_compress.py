import contextlib
import sys
from typing import Iterable, List, Tuple, Dict

import open3d as o3d

import arithmeticcoding
import context_model
import simple_pcc


class ContextEncoder(simple_pcc.Encoder):
    @staticmethod
    def get_octree_encoding(
        point_cloud: o3d.geometry.PointCloud, max_depth: int
    ) -> Iterable[Tuple[int, int]]:
        # generate the OCTREE
        point_cloud = simple_pcc.Encoder._refine_point_cloud(point_cloud)
        octree = simple_pcc.Encoder._get_octree(point_cloud, max_depth)
        # {{{ Do BFS and encoding
        root: o3d.geometry.OctreeInternalNode = octree.root_node
        unmeet_nodes: List[o3d.geometry.OctreeInternalNode] = [root]
        # The root's position in parent is 0
        positions_in_parent: List[int] = [0]
        while len(unmeet_nodes) > 0:
            cur_node = unmeet_nodes.pop(0)
            pos = positions_in_parent.pop(0)
            # {{{ Do encoding
            byte = 0x0
            for index, child in enumerate(cur_node.children):
                if child is not None:
                    byte |= 0x1 << (7 - index)
                    # BFS
                    if isinstance(child, o3d.geometry.OctreeInternalPointNode):
                        unmeet_nodes.append(child)
                        positions_in_parent.append(index + 1)
            yield byte, pos
            # }}}
        # }}}


def main(args):
    # Handle command line arguments
    if len(args) != 3:
        sys.exit(
            "Usage: python cxt_arith_compress.py InputFile OutputFile OutputContextFile"
        )
    input_file, output_file, context_file = args

    # Perform file compression
    point_cloud: o3d.geometry.PointCloud = o3d.io.read_point_cloud(input_file)
    context_encoding_stream = ContextEncoder.get_octree_encoding(
        point_cloud, max_depth=8
    )

    with contextlib.closing(
        arithmeticcoding.BitOutputStream(open(output_file, "wb"))
    ) as bitout_data, contextlib.closing(
        arithmeticcoding.BitOutputStream(open(context_file, "wb"))
    ) as bitout_context:
        compress(context_encoding_stream, bitout_data, bitout_context)


def compress(stream: Iterable[Tuple[int, int]], bitout_data, bitout_context):
    # Encoder for data {{{
    # Context: 0, 1, 2, ..., 8
    # 0 is for root node and the EOF
    context_num = 9
    all_context = tuple(range(context_num))
    # symbol:  0x00, ..., 0xFF
    symbol_num = 0x100  # = 2^8
    # Create the context model {{{
    pos_to_freqs: Dict[int, arithmeticcoding.SimpleFrequencyTable] = {}
    for pos in all_context:
        init_freq_table = None
        if pos != 0:
            # 0x00, ..., 0xFF
            init_freq_table = arithmeticcoding.FlatFrequencyTable(symbol_num)
        else:
            # Context 0 is for root node and the EOF.
            # root node has differs from 0x00 to 0xFF,
            # EOF is set to 0x100, thus we have (symbol_num + 1) symbols
            init_freq_table = arithmeticcoding.FlatFrequencyTable(symbol_num + 1)
        freq_table = arithmeticcoding.SimpleFrequencyTable(init_freq_table)
        pos_to_freqs[pos] = freq_table
    model = context_model.ContextModel(all_context, pos_to_freqs)
    # }}}
    encoder_data = arithmeticcoding.ArithmeticEncoder(32, bitout_data)
    # }}}

    # Encoder for context {{{
    init_freq_table = arithmeticcoding.FlatFrequencyTable(context_num)
    context_freq_table = arithmeticcoding.SimpleFrequencyTable(init_freq_table)
    encoder_context = arithmeticcoding.ArithmeticEncoder(32, bitout_context)
    # }}}

    # append the EOF to end of stream {{{
    def stream_with_eof(stream: Iterable[Tuple[int, int]]) -> Iterable[Tuple[int, int]]:
        eof_symbol = symbol_num
        yield from stream
        yield eof_symbol, 0

    # }}}

    for symbol, pos in stream_with_eof(stream):
        # compress the data
        freq_table = model.get(pos)
        encoder_data.write(freq_table, symbol)
        model.increment(pos, symbol)
        # compress the context
        encoder_context.write(context_freq_table, pos)
        context_freq_table.increment(pos)
    encoder_data.finish()
    encoder_context.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
