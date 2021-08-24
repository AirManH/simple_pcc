import contextlib
import sys
from typing import Iterable, List, Tuple, Dict, Sequence, BinaryIO
from collections import deque

import open3d as o3d

import arithmeticcoding
import context_model
import simple_pcc

# TODO Move these global variables to a single file or class
# Number of sub-nodes that each octree node has
OCTREE_N = 8
# Number of valid symbols (0x0, ..., 0xFF)
VALID_SYMBOL_N = 0x100


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
        unmeet_nodes: deque = deque([root])
        # The root's position in parent is -1
        positions_in_parent: deque = deque([-1])
        while len(unmeet_nodes) > 0:
            cur_node = unmeet_nodes.popleft()
            pos = positions_in_parent.popleft()
            # {{{ Do encoding
            byte = 0x0
            for index, child in enumerate(cur_node.children):
                if child is not None:
                    byte |= 0x1 << (7 - index)
                    # BFS
                    if isinstance(child, o3d.geometry.OctreeInternalPointNode):
                        unmeet_nodes.append(child)
                        positions_in_parent.append(index)
            yield byte, pos
            # }}}
        # }}}


def main(args):
    # Handle command line arguments
    if len(args) != 3:
        sys.exit(
            "Usage: python cxt_arith_compress.py InputFile OutputFile OutputAuxFile"
        )
    input_file, output_file, aux_file = args

    # TODO ugly design here
    depth = 9

    point_cloud: o3d.geometry.PointCloud = o3d.io.read_point_cloud(input_file)
    octree_coding = list(
        ContextEncoder.get_octree_encoding(point_cloud, max_depth=depth)
    )
    model = generate_model(octree_coding)

    # Generate the aux file {{{
    with contextlib.closing(
        arithmeticcoding.BitOutputStream(open(aux_file, "wb"))
    ) as aux_bitout:
        write_root_node(octree_coding, aux_bitout)
        write_model(aux_bitout, model)
    # }}}

    # Open multiple files
    # See also https://docs.python.org/3/library/contextlib.html#supporting-a-variable-number-of-context-managers
    with contextlib.ExitStack() as stack:
        # Open multiple files at once {{{
        # 8 here is because each octree node has 8 sub-nodes
        file_names = [output_file + "." + str(i) for i in range(OCTREE_N)]
        bitout_streams = []
        for f_name in file_names:
            file = stack.enter_context(
                contextlib.closing(arithmeticcoding.BitOutputStream(open(f_name, "wb")))
            )
            bitout_streams.append(file)
        # }}}
        compress(octree_coding, bitout_streams, model)


def write_root_node(
    data_stream: Sequence[Tuple[int, int]], out_bit: arithmeticcoding.BitOutputStream
):
    """
    Write the root node to bit stream in 8 bits.
    Important! : the first element of data_stream with be popped.
    :param data_stream:
    :param out_bit:
    :return:
    """
    first_byte, first_pos = data_stream[0]
    write_int(bitout=out_bit, numbits=8, value=first_byte)


def generate_model(
    data_stream: Iterable[Tuple[int, int]]
) -> context_model.ContextModel:
    # Create the context model {{{
    # Context: 0, 1, 2, ..., 7
    all_context = tuple(range(OCTREE_N))
    # symbol:  0x00, ..., 0xFF, 0x100.
    # (2^8 + 1) symbols in total. The 0x100 is for EOF.
    symbol_num = VALID_SYMBOL_N + 1
    pos_to_freqs: Dict[int, arithmeticcoding.SimpleFrequencyTable] = {}
    for pos in all_context:
        freq_table = arithmeticcoding.SimpleFrequencyTable([0] * symbol_num)
        pos_to_freqs[pos] = freq_table
    model = context_model.ContextModel(all_context, pos_to_freqs)
    # }}}
    for symbol, context in data_stream:
        if context != -1:
            model.increment(context, symbol)

    eof_symbol = VALID_SYMBOL_N
    for pos in all_context:
        model.increment(pos, eof_symbol)

    return model


def write_int(bitout, numbits, value):
    # Writes an unsigned integer of the given bit width to the given stream.
    for i in reversed(range(numbits)):
        bitout.write((value >> i) & 1)  # Big endian


def write_model(
    bitout: arithmeticcoding.BitOutputStream, model: context_model.ContextModel
):
    for context in range(OCTREE_N):
        for symbol in range(VALID_SYMBOL_N + 1):
            write_int(bitout, 32, model.get(context, symbol))


def compress(
    data_stream: Sequence[Tuple[int, int]],
    bitouts: Sequence[arithmeticcoding.BitOutputStream],
    model: context_model.ContextModel,
):
    if len(bitouts) != OCTREE_N:
        raise ValueError("Length of input data streams not equals {}".format(OCTREE_N))
    contexts = tuple(range(OCTREE_N))
    # TODO Declare this 32 as a global variable
    encoders = [arithmeticcoding.ArithmeticEncoder(32, bitout) for bitout in bitouts]

    # To ignore the root node, data_stream starts from 1.
    for byte, pos in data_stream[1:]:
        encoders[pos].write(model.get(pos), byte)

    # Write EOF
    # TODO eof_symbol should be a global variable
    eof_symbol = VALID_SYMBOL_N
    for context in contexts:
        encoder = encoders[context]
        freq_table = model.get(context)
        encoder.write(freq_table, eof_symbol)
    for encoder in encoders:
        encoder.finish()


if __name__ == "__main__":
    main(sys.argv[1:])
