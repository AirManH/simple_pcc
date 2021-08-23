import collections
import copy
import logging
from typing import Iterable, List, Generator

import numpy as np
import open3d as o3d


class Decoder:
    @staticmethod
    def bytes_to_points(
        bts: Iterable[int], depth: int, center: np.ndarray = None, size: float = None
    ) -> Iterable[np.ndarray]:
        if center is None:
            center = np.array([0.0, 0.0, 0.0])
        if size is None:
            size = 1.0
        # {{{ auxiliary array
        # {idx: int, byte position in parent's byte,
        #  dep: int, level(start from 0),
        #  siz: float, edge size of the cube
        #  pos: np.ndarray, position}
        NodeInfo = collections.namedtuple("NodeInfo", ["idx", "dep", "size", "pos"])
        nodes_info: List[NodeInfo] = [NodeInfo(-1, 0, size, center)]
        # }}}
        # Read the root node
        cur_byte: int = -1
        for cur_byte in bts:
            cur_node_info = nodes_info.pop(0)

            logging.debug(
                cur_node_info.dep * "*"
                + f"{cur_byte:08b}: [idx:{cur_node_info.idx}, dep:{cur_node_info.dep}, size:{cur_node_info.size}, pos:{cur_node_info.pos}"
            )

            if cur_node_info.dep > depth:
                break
            if cur_node_info.dep == 0 and depth == 0:
                yield center
                break

            child_indexes = Decoder.byte_to_child_indexes(cur_byte)
            logging.debug(" " * 2 + f"Child indexes: {child_indexes}")
            for index in child_indexes:
                child_pos = Decoder.index_to_child_center(
                    cur_node_info.pos, index, cur_node_info.size
                )
                child_node_info = NodeInfo(
                    index, cur_node_info.dep + 1, cur_node_info.size * 0.5, child_pos
                )
                if child_node_info.dep == depth:
                    yield child_pos
                    logging.debug(" " * 2 + f"Add a child: {child_node_info}")
                nodes_info.append(child_node_info)

    @staticmethod
    def byte_to_child_indexes(b: int) -> List[int]:
        """Translate a byte to list of child index
        0xFF -> [0, 1, ..., 7]
        0x11 -> [3, 7]
        :param b: int, in range [0, 256)
        :return: List[int], subset of [0, 1, 2, ..., 7]
        """
        if b < 0x0 or b > 0xFF:
            raise ValueError("Out of range [0, 256)")
        indexes: List[int] = []
        for n in range(0, 8):
            if (b >> (7 - n)) & 0x1 == 0x1:
                indexes.append(n)
        return indexes

    @staticmethod
    def index_to_child_center(
        center: np.ndarray, index: int, size: float
    ) -> np.ndarray:
        idx_to_vec = {
            0: np.array([-1, -1, -1]),
            1: np.array([1, -1, -1]),
            2: np.array([-1, 1, -1]),
            3: np.array([1, 1, -1]),
            4: np.array([-1, -1, 1]),
            5: np.array([1, -1, 1]),
            6: np.array([-1, 1, 1]),
            7: np.array([1, 1, 1]),
        }
        return center + size * 0.25 * idx_to_vec[index]


class Encoder:
    @staticmethod
    def _refine_point_cloud(
        point_cloud: o3d.geometry.PointCloud,
    ) -> o3d.geometry.PointCloud:
        # {{{ move center to [0, 0, 0]
        pcd: o3d.pybind.geometry.PointCloud = copy.deepcopy(point_cloud).translate(
            -point_cloud.get_center()
        )
        # }}}
        # Fit to unit cube
        pcd.scale(
            1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
            center=pcd.get_center(),
        )
        return pcd

    @staticmethod
    def _get_octree(
        point_cloud: o3d.geometry.PointCloud, max_depth: int
    ) -> o3d.geometry.Octree:
        octree = o3d.geometry.Octree(max_depth=max_depth)
        octree.convert_from_point_cloud(point_cloud)
        return octree

    @staticmethod
    def get_octree_encoding(
        point_cloud: o3d.geometry.PointCloud, max_depth: int
    ) -> Iterable[int]:
        # generate the OCTREE
        point_cloud = Encoder._refine_point_cloud(point_cloud)
        octree = Encoder._get_octree(point_cloud, max_depth)
        # {{{ Do BFS and encoding
        root: o3d.geometry.OctreeInternalNode = octree.root_node
        unmeet_nodes: List[o3d.geometry.OctreeInternalNode] = [root]
        while len(unmeet_nodes) > 0:
            cur_node = unmeet_nodes.pop(0)
            # {{{ Do encoding
            byte = 0x0
            for index, child in enumerate(cur_node.children):
                if child is not None:
                    byte |= 0x1 << (7 - index)
                    # BFS
                    if isinstance(child, o3d.geometry.OctreeInternalPointNode):
                        unmeet_nodes.append(child)
            yield byte
            # }}}
        # }}}
