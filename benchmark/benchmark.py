import argparse
import collections
import pathlib
from typing import Dict, Iterable, List, Optional, Set

import altair
import altair_saver
import p_tqdm
import pandas as pd
import tester
import tqdm


class Benchmark:
    Entry = collections.namedtuple(
        "Entry", ["dataset", "frame", "alg", "path_in", "path_out"]
    )
    Result = collections.namedtuple(
        "Result", ["dataset", "frame", "bits", "voxel", "alg"]
    )
    all_datasets: Set[str] = {"andrew9", "david9", "phil9", "ricardo9", "sarah9"}
    all_algorithms: Set[str] = {"pac", "ac", "draco", "or", "mpeg"}
    alg_to_point_shape: Dict[str, str] = {
        "pac": "circle",
        "ac": "square",
        "draco": "cross",
        "or": "diamond",
        "mpeg": "triangle-up",
    }

    def __init__(
        self,
        csv_path: str,
        datasets: Iterable[str],
        algorithms: Iterable[str],
        tmp_dir: str,
    ):
        self.csv_path = csv_path
        self.result = []
        self.df: Optional[pd.DataFrame] = None

        self.entries: List[Benchmark.Entry] = []
        self.datasets = list(datasets)
        name_to_alg = {
            "pac": tester.CxtArithTester,
            "ac": tester.ArithmeticOctreeTester,
            "draco": tester.DracoTester,
            "or": tester.OctreeTester,
            "mpeg": tester.GpccTester,
        }
        self.testers: List[tester.TesterBase] = [
            name_to_alg[name]() for name in algorithms
        ]
        self.tag_to_tester: Dict[str, tester.TesterBase] = {
            t.tag(): t for t in self.testers
        }

        self.output_dir: pathlib.Path = pathlib.Path(tmp_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._generate_input_entries()

    def _generate_input_entries(self):
        for data_set in self.datasets:
            ply_path_dir = pathlib.Path(data_set) / "ply"
            for path_ply_file in ply_path_dir.glob("*.ply"):
                abs_path = str(path_ply_file.resolve())
                # If path_ply_file is "a/b/c.ply", then the file_name is "c"
                file_name = path_ply_file.stem
                # file_name is like "frame0010", "10" indicates the frame number
                frame_num = int(file_name.lstrip("frame"))

                for tst in self.testers:
                    output_file_name = "{}-{}-{}.bin".format(
                        data_set, file_name, tst.tag()
                    )
                    output_path = str((self.output_dir / output_file_name).resolve())
                    e = Benchmark.Entry(
                        dataset=data_set,
                        frame=frame_num,
                        alg=tst.tag(),
                        path_in=abs_path,
                        path_out=output_path,
                    )
                    self.entries.append(e)

    def run(self, parallel: Optional[int] = None):
        if parallel == 1:
            self._run_serial()
        else:
            self._run_parallel(parallel)
        self._result_to_df()

    def _run_parallel(self, n_cpu: Optional[int] = None):
        results: Iterable["Benchmark.Result"] = p_tqdm.p_uimap(
            self.single_run, self.entries, num_cpus=n_cpu
        )
        for result in results:
            self.result.append(result)

    def _run_serial(self):
        for entry in tqdm.tqdm(self.entries):
            res = self.single_run(entry)
            self.result.append(res)

    def single_run(self, entry: "Benchmark.Entry") -> "Benchmark.Result":
        tst = self.tag_to_tester[entry.alg]
        num_bits = tst.run(entry.path_in, entry.path_out, depth=9, clean=True)
        num_voxel = tester.TesterBase.get_occupied_voxel_num(
            entry.path_in, voxel_size=1
        )
        return Benchmark.Result(
            entry.dataset,
            entry.frame,
            bits=num_bits,
            voxel=num_voxel,
            alg=tst.tag(),
        )

    def _result_to_df(self):
        self.df = pd.DataFrame(self.result)
        self.df["bpov"] = self.df["bits"] / self.df["voxel"]

    def save_data(self):
        if self.df is None:
            raise ValueError("No data.")
        self.df.to_csv(self.csv_path, index=False)

    def read_data(self) -> bool:
        try:
            self.df = pd.read_csv(self.csv_path)
            return True
        except FileNotFoundError:
            return False

    def plot(self, png_path: str, scale: float = 1.0):
        if self.df is None:
            raise ValueError("No data.")
        chart = None
        if len(self.datasets) == 1:
            chart = (
                altair.Chart(self.df)
                .mark_line()
                .encode(
                    x=altair.Y("frame", title="Frame"),
                    y=altair.X("bpov", title="bpov (bits per occupied voxel)"),
                    color=altair.Color("alg", title="Algorithms"),
                    strokeDash="alg",
                )
            )
        else:
            chart = (
                altair.Chart(self.df)
                .mark_line()
                .encode(
                    x=altair.Y("frame", title="Frame"),
                    y=altair.X("bpov", title="bpov (bits per occupied voxel)"),
                    color=altair.Color("alg", title="Algorithms"),
                    strokeDash="alg",
                    row="dataset",
                )
            )

        altair_saver.save(chart, png_path, scale_factor=scale)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the simple_pcc.")
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        type=str,
        choices=Benchmark.all_datasets,
        help="datasets to benchmark",
    )
    parser.add_argument(
        "-a",
        "--algorithms",
        nargs="+",
        type=str,
        choices=Benchmark.all_algorithms,
        help="algorithms to benchmark",
    )
    parser.add_argument(
        "-t",
        "--tmp_dir",
        type=str,
        default="tmp",
        help="temporary directory for saving intermediate files (default: tmp)",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="output csv file"
    )
    parser.add_argument("-oi", "--output_image", type=str, help="output image")
    parser.add_argument(
        "-ois",
        "--output_image_scale",
        type=float,
        default=3.0,
        help="output image scale (default: 3.0)",
    )
    parser.add_argument(
        "--cpu", type=int, help="number of CPUs to use (default: use all cpus)"
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    bm = Benchmark(
        csv_path=args.output,
        datasets=args.datasets,
        algorithms=args.algorithms,
        tmp_dir=args.tmp_dir,
    )

    if not bm.read_data():
        n_cpu = args.cpu if "cpu" in args else None
        bm.run(parallel=n_cpu)
        bm.save_data()
    if "output_image" in args:
        bm.plot(args.output_image, scale=args.output_image_scale)


if __name__ == "__main__":
    main()
