import pathlib
import subprocess
import sys

import open3d as o3d

current_file_dir = pathlib.Path(__file__).parent.resolve()
path_to_simple_pcc = (pathlib.Path(__file__).parent / "..").resolve()
sys.path.insert(0, str(path_to_simple_pcc))

import simple_pcc


class TesterBase:
    def __init__(self):
        pass

    def run(
        self, input_file: str, output_file: str, depth: int = None, clean: bool = False
    ) -> int:
        raise NotImplementedError()

    @staticmethod
    def file_size(file: str) -> int:
        """Calculate file size in bits."""
        p = pathlib.Path(file)
        return p.stat().st_size * 8

    @staticmethod
    def get_occupied_voxel_num(file: str, voxel_size: float):
        pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(file)
        pcd = pcd.voxel_down_sample(voxel_size)
        return len(pcd.points)

    def tag(self) -> str:
        raise NotImplementedError()


class CxtArithTester(TesterBase):
    def __init__(self, exe_file: str = None):
        super().__init__()

        if exe_file is None:
            self.exe_file = str(
                (path_to_simple_pcc / "simple_pcc" / "cxt_arith_compress.py").resolve()
            )
        else:
            self.exe_file = str(exe_file)

    def run(
        self, input_file: str, output_file: str, depth=None, clean: bool = False
    ) -> int:
        aux_file = output_file + ".aux"
        args = [
            "-i",
            input_file,
            "-o",
            output_file,
            "-a",
            aux_file,
        ]
        if depth is not None:
            args += ["-d", str(depth)]
        comp_proc = subprocess.run(
            ["python", self.exe_file] + args,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        # File size {{{
        compress_files = [output_file + "." + str(i) for i in range(8)]
        num_bits = 0
        for f in compress_files + [aux_file]:
            num_bits += self.file_size(f)
        # }}}

        if clean:
            for f in compress_files + [aux_file]:
                p = pathlib.Path(f)
                p.unlink()

        return num_bits

    def tag(self) -> str:
        return "P(AC)"


class DracoTester(TesterBase):
    def run(
        self, input_file: str, output_file: str, depth: int = None, clean: bool = False
    ) -> int:
        default_args = [
            "--skip",
            "NORMAL",
            "--skip",
            "TEX_COORD",
            "--skip",
            "GENERIC",
            "-cl",
            "10",
        ]
        subprocess.run(
            [
                "draco_encoder",
                "-point_cloud",
                "-i",
                input_file,
                "-o",
                output_file,
                "-qp",
                str(depth),
            ]
            + default_args,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

        num_bits = self.file_size(output_file)

        if clean:
            p = pathlib.Path(output_file)
            p.unlink()

        return num_bits

    def tag(self) -> str:
        return "Draco"


class OctreeTester(TesterBase):
    def run(
        self, input_file: str, output_file: str, depth=None, clean: bool = False
    ) -> int:
        pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(input_file)
        coding = simple_pcc.Encoder.get_octree_encoding(pcd, max_depth=depth)
        if clean:
            return sum(1 for b in coding) * 8
        else:
            bts = bytes(coding)
            with open(output_file, "wb") as f:
                f.write(bts)
            return len(bts)

    def tag(self) -> str:
        return "OR"


class ArithmeticOctreeTester(TesterBase):
    def __init__(self, script_path: str = None):
        super().__init__()
        if script_path is None:
            script_path = str((current_file_dir / "arithmetic-compress.py").resolve())
        self.script_path = script_path

    def run(
        self, input_file: str, output_file: str, depth: int = None, clean: bool = False
    ) -> int:
        # input -> octree file -> output
        pcd: o3d.geometry.PointCloud = o3d.io.read_point_cloud(input_file)
        bts = bytes(simple_pcc.Encoder.get_octree_encoding(pcd, max_depth=depth))
        octree_file = output_file + ".octree"
        with open(octree_file, "wb") as f:
            f.write(bts)

        subprocess.run(
            ["python", "arithmetic-compress.py", octree_file, output_file],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        num_bits = self.file_size(output_file)
        # The octree coding file must be cleaned
        pathlib.Path(octree_file).unlink()
        if clean:
            pathlib.Path(output_file).unlink()
        return num_bits

    def tag(self) -> str:
        return "AC"


class GpccTester(TesterBase):
    def run(
        self, input_file: str, output_file: str, depth=None, clean: bool = False
    ) -> int:
        subprocess.run(
            [
                "tmc3",
                "--mode=0",
                "--positionQuantizationScale=1",
                "--trisoupNodeSizeLog2=0",
                "--neighbourAvailBoundaryLog2=8",
                "--intra_pred_max_node_size_log2=6",
                "--inferredDirectCodingMode=0",
                "--maxNumQtBtBeforeOt=4",
                "--uncompressedDataPath={}".format(input_file),
                "--compressedStreamPath={}".format(output_file),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        num_bits = self.file_size(output_file)
        if clean:
            pathlib.Path(output_file).unlink()
        return num_bits

    def tag(self) -> str:
        return "MPEG"
