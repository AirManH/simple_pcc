import collections
import contextlib
import sys
from typing import Iterable, Dict, List, Sequence

import numpy as np
import open3d as o3d

import arithmeticcoding
import context_model
import simple_pcc

# Number of sub-nodes that each octree node has
OCTREE_N = 8
# Number of valid symbols (0x0, ..., 0xFF)
VALID_SYMBOL_N = 0x100


class CxtArithDecoder:
    @staticmethod
    def bytes_to_points(
        bytes_list: List[List[int]],
        root_byte: int,
        depth: int,
        center: np.ndarray = None,
        size: float = None,
    ) -> Iterable[np.ndarray]:
        if len(bytes_list) != OCTREE_N:
            raise ValueError("Length of bytes_list should be {}".format(OCTREE_N))
        if center is None:
            center = np.array([0.0, 0.0, 0.0])
        if size is None:
            size = 1.0
        # Auxiliary array
        # {idx: int, byte position in parent's byte,
        #  dep: int, level(start from 0),
        #  siz: float, edge size of the cube
        #  pos: np.ndarray, position}
        Node = collections.namedtuple("Node", ["idx", "dep", "size", "pos"])
        unmeet_nodes: List[Node] = [Node(-1, 0, size, center)]
        unmeet_bytes: List[int] = [root_byte]

        while len(unmeet_bytes) > 0:
            cur_byte = unmeet_bytes.pop(0)
            cur_node = unmeet_nodes.pop(0)
            if cur_node.dep > depth:
                break
            # If this node is not leaf node, it must has corresponding octree coding,
            # thus we need to read the byte.
            for idx in simple_pcc.Decoder.byte_to_child_indexes(cur_byte):
                child_node_pos = simple_pcc.Decoder.index_to_child_center(
                    cur_node.pos, idx, cur_node.size
                )
                child_node = Node(
                    idx=idx,
                    dep=cur_node.dep + 1,
                    size=cur_node.size * 0.5,
                    pos=child_node_pos,
                )
                unmeet_nodes.append(child_node)
                if child_node.dep == depth:
                    yield child_node.pos
                else:
                    next_byte = bytes_list[idx].pop(0)
                    unmeet_bytes.append(next_byte)


def main(args):
    # Handle command line arguments
    if len(args) != 3:
        sys.exit(
            "Usage: python cxt_arith_decompress.py InputFile InputAuxFile OutputFile"
        )

    input_file, aux_file, output_file = args

    # TODO ugly design here
    depth = 9

    with contextlib.closing(
        arithmeticcoding.BitInputStream(open(aux_file, "rb"))
    ) as bitin_aux:
        root_node = read_root_node(bitin_aux)
        model = read_model(bitin_aux)

    file_names = [input_file + "." + str(i) for i in range(OCTREE_N)]
    bitin_streams = [arithmeticcoding.BitInputStream(open(f, "rb")) for f in file_names]
    bytes_list = decompress(bitin_streams, model)
    for bitin_stream in bitin_streams:
        bitin_stream.close()

    points = list(CxtArithDecoder.bytes_to_points(bytes_list, root_node, depth))

    point_cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np.array(list(points)))
    o3d.io.write_point_cloud(output_file, point_cloud)


def read_int_from_stream(n_bits, bit_in: arithmeticcoding.BitInputStream):
    result = 0
    for _ in range(n_bits):
        result = (result << 1) | bit_in.read_no_eof()  # Big endian
    return result


def read_root_node(bit_in: arithmeticcoding.BitInputStream) -> int:
    return read_int_from_stream(8, bit_in)


def read_model(bit_in: arithmeticcoding.BitInputStream) -> context_model.ContextModel:
    all_context = tuple(range(OCTREE_N))
    # symbol:  0x00, ..., 0xFF, 0x100.
    # (2^8 + 1) symbols in total. The 0x100 is for EOF.
    symbol_num = VALID_SYMBOL_N + 1

    pos_to_freqs: Dict[int, arithmeticcoding.SimpleFrequencyTable] = {}
    for pos in all_context:
        # Read 32 bits ber frequency
        values = [read_int_from_stream(32, bit_in) for _ in range(symbol_num)]
        # Save
        freq_table = arithmeticcoding.SimpleFrequencyTable(values)
        pos_to_freqs[pos] = freq_table
    model = context_model.ContextModel(all_context, pos_to_freqs)
    return model


def decompress(
    bitin_data: Sequence[arithmeticcoding.BitInputStream],
    model: context_model.ContextModel,
) -> List[List[int]]:
    if len(bitin_data) != OCTREE_N:
        raise ValueError("Length of bitin_data should be {}".format(OCTREE_N))

    # Get 8 octree coding {{{
    contexts = tuple(range(OCTREE_N))
    bytes_list: List[List[int]] = [[] for _ in contexts]
    # TODO eof_symbol should be a global variable
    eof_symbol = VALID_SYMBOL_N
    for cxt in contexts:
        bitin = bitin_data[cxt]
        # TODO Declare this 32 as a global variable
        decoder = arithmeticcoding.ArithmeticDecoder(32, bitin)
        while True:
            b = decoder.read(model.get(cxt))
            if b == eof_symbol:
                break
            bytes_list[cxt].append(b)
    # }}}
    return bytes_list


if __name__ == "__main__":
    main(sys.argv[1:])
