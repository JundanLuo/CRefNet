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


import os
import shutil
import argparse


test_list = ['cup2', 'deer', 'frog2', 'paper2', 'pear',
             'potato', 'raccoon', 'sun', 'teabag1', 'turtle']


def extract_images(input_dir, output_dir):
    print(f"Extracting images from {input_dir} to {output_dir}. "
          f"Test list: {test_list}")
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all subdirectories in the input directory
    for subfolder in test_list:
        subfolder_path = os.path.join(input_dir, subfolder)

        # Check if the item is a directory
        if os.path.exists(subfolder_path):

            # Define mapping of original filenames and new filenames
            filename_mapping = {
                "diffuse.png": f"{subfolder}-input.png",
                "mask.png": f"{subfolder}-label-mask.png",
                "reflectance.png": f"{subfolder}-label-albedo.png",
                "shading.png": f"{subfolder}-label-shading.png"
            }

            # For each file in the mapping, check if it exists and if so, copy and rename it
            for original_filename, new_filename in filename_mapping.items():
                original_filepath = os.path.join(subfolder_path, original_filename)

                if os.path.exists(original_filepath):
                    new_filepath = os.path.join(output_dir, new_filename)
                    shutil.copyfile(original_filepath, new_filepath)
                    print(f"Copied {original_filepath} to {new_filepath}")
                else:
                    print(f"File {original_filepath} does not exist")
        else:
            print(f"Directory {subfolder_path} does not exist")


if __name__ == "__main__":
    # Command:
    # python extract_MIT_test_data.py --input_dir "../data/MIT-intrinsic/data" --output_dir "../data/MIT_test_extracted"

    # Handle command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory path")
    args = parser.parse_args()

    # Call the function
    extract_images(args.input_dir, args.output_dir)
