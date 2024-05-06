# ////////////////////////////////////////////////////////////////////////////
# // This file is part of CRefNet. For more information
# // see <https://github.com/JundanLuo/CRefNet>.
# // If you use this code, please cite our paper as
# // listed on the above website.
# //
# // Licensed under the Apache License, Version 2.0 (the “License”);
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an “AS IS” BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.
# ////////////////////////////////////////////////////////////////////////////


import sys
sys.path.append("../")
import argparse

from configs.config import get_cfg_defaults
from dataset import cgintrinsics_dataset, iiw_dataset, mpi_dataset, mit_intrinsic_dataset
# from dataset import hypersim_dataset
import constants as C


def check_dataset(dataset):
    cfg = get_cfg_defaults()
    cfg.freeze()
    if dataset == C.CGIntrinsics:
        cgintrinsics_dataset.is_CGIntrinsics_complete("../"+cfg.DATASET.CGIntrinsics_dir,
                                                      32, 8,
                                                      "../experiments/check_CGI_dataset/")
    elif dataset == C.Hypersim:
        assert False, "Hypersim dataset is not supported yet."
        # hypersim_dataset.is_Hypersim_complete("../"+cfg.DATASET.Hypersim_dir,
        #                                       32, 8,
        #                                       "../experiments/check_Hypersim_dataset",
        #                                       False)
    elif dataset == C.IIW:
        iiw_dataset.is_IIW_complete("../" + cfg.DATASET.IIW_dir, "../data/CGIntrinsics/IIW/NIID-Net_pred_IIW",
                                    32, 8,
                                    "../experiments/check_IIW_dataset/")
    elif dataset == C.MPI_Sintel:
        mpi_dataset.is_MPI_complete("../" + cfg.DATASET.MPI_Sintel_dir, 32, 8,
                                    "../experiments/check_MPI_dataset")
    elif dataset == C.MIT_Intrinsic:
        mit_intrinsic_dataset.is_MIT_Intrinsic_complete("../"+cfg.DATASET.MIT_Intrinsic_dir,
                                                        1, 1,
                                                        "../experiments/check_MIT_dataset/")
    else:
        print(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help=f'Options: {C.CGIntrinsics}, {C.IIW}, {C.MPI_Sintel}, {C.MIT_Intrinsic}')
    args = parser.parse_args()

    # Run the dataset check
    check_dataset(args.dataset)

