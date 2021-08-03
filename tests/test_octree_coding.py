import pathlib
import sys
from typing import List

import numpy as np
import open3d as o3d
import pytest

from simple_pcc.simple_pcc import Decoder, Encoder

current_file_dir = pathlib.Path(__file__).parent.resolve()
path_to_simple_pcc = (pathlib.Path(__file__).parent / "..").resolve()
sys.path.insert(0, str(path_to_simple_pcc))


def oct_bytes_print(bts: bytes):
    print()
    for bt in bts:
        print(Decoder._byte_to_child_indexes(bt), end=", ")
    print()


@pytest.mark.parametrize("depth", [3])
def test_octree_coding(depth: int):
    test_samples = [
        {
            "bytes": [0x11],
            "ans": [[1 / 4, 1 / 4, -1 / 4], [1 / 4, 1 / 4, 1 / 4]],
            "depth": 1,
        },
        {"bytes": [0x00], "ans": [[0, 0, 0]], "depth": 0},
        {
            "bytes": [0x01, 0x11, 0x01, 0x01],
            "ans": [
                [7 / 16, 7 / 16, 3 / 16],
                [7 / 16, 7 / 16, 7 / 16],
            ],
            "depth": 3,
        },
        {
            "bytes": [0x01, 0x11, 0x01, 0x01, 0x88, 0x04],
            "ans": [
                [13 / 32, 13 / 32, 5 / 32],
                [13 / 32, 13 / 32, 7 / 32],
                [15 / 32, 13 / 32, 15 / 32],
            ],
            "depth": 4,
        },
    ]
    for test_sample in test_samples:
        bts = bytes(test_sample["bytes"])
        pts = Decoder.bytes_to_points(bts, test_sample["depth"], np.array([0, 0, 0]), 1)
        assert np.allclose(pts, np.array(test_sample["ans"]))


@pytest.mark.parametrize(
    "depth, file_path",
    [(3, str(current_file_dir / "data_bunny" / "bun_zipper_res4.ply"))],
)
def test_encoder_decoder(depth: int, file_path: str):
    # {{{ 1th point cloud

    pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(file_path)
    oct: o3d.geometry.Octree = o3d.geometry.Octree(depth)
    oct.convert_from_point_cloud(pcd)
    bts = bytes(Encoder.get_octree_encoding(pcd, depth))
    # visualize
    o3d.visualization.draw_geometries([pcd])
    # }}}

    # {{{ 2th point cloud
    points = Decoder.bytes_to_points(
        bts, depth, np.array(pcd.get_center()), float(oct.size)
    )
    # create the intermedia point cloud from points
    inter_pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    inter_pcd.points = o3d.utility.Vector3dVector(points)

    inter_octree = o3d.geometry.Octree(depth)
    inter_octree.convert_from_point_cloud(inter_pcd)
    inter_bts = list(Encoder.get_octree_encoding(inter_pcd, depth))
    # visualize
    o3d.visualization.draw_geometries([inter_pcd])
    # }}}

    assert np.allclose(list(bts), list(inter_bts))
