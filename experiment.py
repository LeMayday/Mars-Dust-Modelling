# local files
from mars_topography import Region

# default
import argparse
from dataclasses import dataclass
from enum import Enum
import os
from typing import Tuple


class Res(Enum):
    COURSE = 'course'
    FINE = 'fine'


def get_res_multiplier(res: Res):
    assert(res in Res)
    match res:
        case Res.COURSE:
            return 1
        case Res.FINE:
            return 2


@dataclass(frozen=True)
class Experiment():
    name: str
    region: Region = None

    def __post_init__(self):
        first_letters = ['I', 'E']
        second_letters = ['C', 'F']
        if self.name[0] not in first_letters:
            raise ValueError("Unexpected letter in experiment name at position 0.")
        if self.name[1] not in second_letters:
            raise ValueError("Unexpected letter in experiment name at position 1.")
        if self.region is not None:
            self.name += "_" + self.region.to_string()

    def is_implicit(self) -> bool:
        return self.name[0] == 'I'

    def is_3D(self) -> bool:
        return '_3D' in self.name

    def res(self) -> Res:
        match self.name[1]:
            case 'C':
                return Res.COURSE
            case 'F':
                return Res.FINE

    def get_num_cells(self) -> Tuple[int, int, int]:
        res_multiplier = get_res_multiplier(self.res())
        nx1 = 64 * res_multiplier
        nx2 = 256 * res_multiplier
        nx3 = 256 * res_multiplier if self.is_3D() else 1
        return nx1, nx2, nx3


def handle_input(args = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", required=True, type=str, help="Input file")
    parser.add_argument("-r", "--restart-file", type=str, help="Continue integrating from a file")
    args = parser.parse_args(args)

    output_dir, _ = os.path.split(args.input_file)
    input_file = f"{output_dir}/{args.input_file}"

    return input_file, output_dir, args.restart_file
