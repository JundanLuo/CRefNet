# CRefNet: Learning Consistent Reflectance Estimation With a Decoder-Sharing Transformer
[[paper]](https://purehost.bath.ac.uk/ws/portalfiles/portal/304058985/Jundan_s_TVCG_submission.pdf) 
[[supplement doc]](https://drive.google.com/file/d/13yi0vXYD1Ph5-noZr_Ndx0_SHHtoQrwq/view?usp=sharing)
[[supplement materials]](https://drive.google.com/file/d/1B2oe3c2tYZwYyOwvRHGoyCbNBWGH6lJb/view?usp=drive_link)

![architecture](./assets/pipeline.png)


Updates
-
+ 01/Nov/2024: Released the training code.
+ 05/May/2024: Released the trained models and the evaluation code.


Dependencies
-
+ Python 3.6
+ PyTorch 1.8.2
+ We provide the ```tools/install.txt``` file for other dependencies.


Datasets
-
+ Download:
  + Follow [CGIntrinsics](https://github.com/zhengqili/CGIntrinsics) to download the CGI, IIW and SAW datasets. Z. Li and N. Snavely augment the original [IIW](http://opensurfaces.cs.cornell.edu/intrinsic/#) dataset.
  + MIT-intrinsic: [Webpage](https://people.csail.mit.edu/kimo/publications/intrinsic/).
  + MPI_Sintel: [Webpage](http://sintel.is.tue.mpg.de/depth). Contact the authors to request the version rendered for intrinsic image decomposition.
+ Split files: we back up the used split files in the ```dataset/split_files``` folder. 
MIT-intrinsic's split is coded in ```dataset/mit_intrinsic_dataset.py```.
+ Put the datasets in the ```data/``` folder. The final directory structure:
    ```
    CRefNet project
    |---README.md
    |---...
    |---data
        |---CGIntrinsics
            |---intrinsics_final
                |---images   
                |---rendered
                |---...
            |---IIW
                |---data
                |---test_list
                |---...
            |---SAW
                |---saw_images_512
                |---saw_pixel_labels
                |---saw_splits
                |---train_list
        |---MIT-intrinsic
            |---data
                |---apple
                |---...
        |---MPI_Sintel_IID
            |---MPI-main-albedo
            |---MPI-main-clean
            |---...
    ```
+ Paths to the datasets are set in ```configs/config.py```.


Train
-
[Navigate to the Training Doc](docs/train/README.md)


Trained Models
-
General Use
- Main Model: [final_real.pt](https://drive.google.com/file/d/13oVzwJabQt5HmbSWRFbS9AWVc5QtN_2R/view?usp=sharing)
    - Efficient Variant: [crefnet-e.pt](https://drive.google.com/file/d/143XC2f7skidSmAHJYNNdZDmTw4R6sZKg/view?usp=sharing)

Benchmark-Specific Models
- MIT Benchmark: [model_MIT.pt](https://drive.google.com/file/d/13sw22gRJU6VFPp773Uy8vVC3PX-UUrv3/view?usp=sharing) 
- MPI Benchmark:
[model_MPI_scene_front.pt](https://drive.google.com/file/d/13lIyCS7THXeKXXNj-NSI7wEKDKlzshEi/view?usp=sharing) 
and [model_MPI_scene_reverse.pt](https://drive.google.com/file/d/13relZT9mAfgwLF_rDsJdS5-ggWwmT6Ql/view?usp=sharing)


Evaluation
- 
[Navigate to the Evaluation Doc](docs/test/README.md)


Infer
-
+ Infer on images in a single directory:
  ```console
    CUDA_VISIBLE_DEVICES="0" python infer.py \
        --img-dir ./test_examples/ \
        --out-dir ./experiments/out_test_examples/ \
        --min_input_dim 448  \
        --output_original_size \
        --cfg="configs/crefnet.yaml" \
        MODEL.checkpoint "./trained_models/final_real.pt"
    ```
+ ```dataset/imagefolder_dataset.py``` is the dataset class used for loading images in a directory.
+ Other possibly useful options:
  + ```--gamma_correct_input```: convert the input images in linear space into sRGB space.
  + ```--gamma_correct_output```: convert the output images in linear space into sRGB space.
  + ```--max_input_dim```: set the maximum dimension of the input images. 
    + If both ```--min_input_dim``` and ```--max_input_dim``` are set, the aspect ratio may not be preserved.
    + If not set either, the original image size will be used.


Acknowledgements
-
- Test images in ```test_examples/``` are from the [IIW benchmark](http://opensurfaces.cs.cornell.edu/intrinsic/). Image license: [Attribution 2.0 Generic](https://creativecommons.org/licenses/by/2.0/).
- Dense metrics (```solver/metrics_intrinsic_images.py```) for the MPI Sintel benchmark were migrated from the MATLAB code provided by the paper [A Simple Model for Intrinsic Image Decomposition with Depth Cues](https://cqf.io/publications.html).
- We have used/modified codes from the following projects:
  + [taming-transformers](https://github.com/CompVis/taming-transformers):
    + Codes for the encoder and decoder architecture we used in ```modeling/vqgan```.
  + [Swin-Transformer](https://github.com/microsoft/Swin-Transformer):
    + The network structure of the SwinV2 transformer layers we used in ```modeling/swin_transformer_v2.py```
  + [CGIntrinsics](https://github.com/zhengqili/CGIntrinsics):
    + Codes for loading data from the CGI and IIW datasets in ```dataset/cgintrinsics_dataset.py``` and ```dataset/iiw_dataset.py```.
    + Codes for evaluation on the IIW benchmark in ```solver/metrics_iiw.py```. 
    This code is originally provided by [IIW](http://opensurfaces.cs.cornell.edu/intrinsic/#).
    + Codes for evaluation on the SAW benchmark in ```solver/metrics_saw.py``` and ```solver/saw_utils.py```.
    These codes are originally provided by [SAW](http://opensurfaces.cs.cornell.edu/saw/).
- We thank P. Das ([PIE-Net](https://github.com/Morpheus3000/PIE-Net?tab=readme-ov-file)) 
and Z. Wang ([Single Image Intrinsic Decomposition with Discriminative Feature Encoding](https://github.com/screnary/SingleImageIntrinsic))
for their assistance in evaluation and comparison.
    

Citation
-
If you find this code useful for your research, please cite:
  ```
    @article{luo2023crefnet,
      title={CRefNet: Learning Consistent Reflectance Estimation With a Decoder-Sharing Transformer},
      author={Luo, Jundan and Zhao, Nanxuan and Li, Wenbin and Richardt, Christian},
      journal={IEEE Transactions on Visualization and Computer Graphics},
      year={2023},
      publisher={IEEE}
    }
  ```

Contact
-
Please contact Jundan Luo (<jundanluo22@gmail.com>) if you have any questions. 
Feel free to give any feedback.
