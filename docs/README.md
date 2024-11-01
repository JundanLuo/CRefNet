Train
-
### Main model
Our main model (final_real.pt) is trained in the following steps:
1. Copy the CGI's split files (```dataset/split_files/CGIntrinsics/*```) 
   to the ```./data/CGIntrinsics/intrinsics_final/train_test_split/``` folder.
1. Train the reflectance estimation branch on the CGI dataset using 224X224 images:
   1. The training config file: ```configs/crefnet_R_cgi.yaml```
   2. Run
      ```console
          CUDA_VISIBLE_DEVICES="0" python train.py \
            --cfg="configs/crefnet_R_cgi.yaml" \
            DIR "./experiments/CRefNet/R/CGI"
      ```
   3. Get the trained model and rename it to ```experiments/CRefNet/crefnet_r_cgi.pt```
   4. Note that the 224X224 model should be evaluated using the option ```MODEL.input_img_size 224```
      + <details>
        <summary>(click to expand)</summary>
       
        ##### 
        ```console
         CUDA_VISIBLE_DEVICES="0" python evaluate.py \
           --cfg="configs/crefnet_R_cgi.yaml" \
           MODEL.checkpoint "experiments/CRefNet/crefnet_r_cgi.pt" \
           MODEL.input_img_size 224 \
           TEST.dataset IIW \
           TEST.vis_per_iiw 100
        ```
        </details>
2. Finetune the reflectance estimation branch on the CGI+IIW datasets using 224x224 images:
   1. The training config file: ```configs/crefnet_R_cgi_iiw.yaml```
   2. Run
      ```console
         CUDA_VISIBLE_DEVICES="0" python train.py \
         --cfg="configs/crefnet_R_cgi_iiw.yaml" \
         DIR  "./experiments/CRefNet/R/CGI+IIW" \
         MODEL.checkpoint "experiments/CRefNet/crefnet_r_cgi.pt"
      ```
   3. Get the trained model and rename it to ```experiments/CRefNet/crefnet_r_cgi_iiw.pt```
   4. Evaluate: 
      + <details>
        <summary>(click to expand)</summary>
       
        ##### 
        ```console
         CUDA_VISIBLE_DEVICES="0" python evaluate.py \
           --cfg="configs/crefnet_R_cgi_iiw.yaml" \
           MODEL.checkpoint "experiments/CRefNet/crefnet_r_cgi_iiw.pt" \
           MODEL.input_img_size 224 \
           TEST.dataset IIW \
           TEST.vis_per_iiw 100
        ```
        </details>
3. Finetune the reflectance estimation branch on the CGI+IIW datasets using 448x448 images:
   1. The training config file: ```configs/refine_crefnet_R_cgi_iiw_448X.yaml```
   2. Run
      ```console
         CUDA_VISIBLE_DEVICES="0" python train.py \
         --cfg="configs/refine_crefnet_R_cgi_iiw_448X.yaml" \
         DIR  "./experiments/CRefNet/R/CGI+IIW_448X" \
         MODEL.checkpoint "experiments/CRefNet/crefnet_r_cgi_iiw.pt"
      ```      
   3. Get the trained model and rename it to ```experiments/CRefNet/crefnet_r_cgi_iiw_448X.pt```
   4. Evaluate:
        + <details>
            <summary>(click to expand)</summary>
         
            ##### 
            ```console
             CUDA_VISIBLE_DEVICES="0" python evaluate.py \
             --cfg="configs/refine_crefnet_R_cgi_iiw_448X.yaml" \
             MODEL.checkpoint "experiments/CRefNet/crefnet_r_cgi_iiw_448X.pt" \
             MODEL.input_img_size 448 \
             TEST.dataset IIW \
             TEST.vis_per_iiw 100
            ```
            </details>
4. Train the shading estimation branch on the CGI+IIW dataset using 224x224 images:
   1. Infer on the IIW dataset (both train and test) using the [NIID-Net model](https://github.com/zju3dv/NIID-Net).
    The results should be in float32 *.npy format.
   2. The training config file: ```configs/crefnet_S_cgi_iiw.yaml```. 
   3. Modify the path to the NIID-Net's intrinsic image results in the config file at ```DATASET.IIW_pred_by_NIID_Net```.
      We store the inferred results in the ```./data/CGIntrinsics/IIW/NIID-Net_pred_IIW``` folder with the following structure:
       ```
       CRefNet project
       |---README.md
       |---data
           |---CGIntrinsics
               |---IIW
                   |---NIID-Net_pred_IIW
                       |---pred_train
                           |---raw
                               |---64434-r.npy
                               |---64434-s.npy
                               |---...
                       |---pred_test
                              |---raw
                                  |---...
       ```
   4. Run
      ```console
         CUDA_VISIBLE_DEVICES="0" python train.py \
         --cfg="configs/crefnet_S_cgi_iiw.yaml" \
         DIR  "./experiments/CRefNet/S/CGI+IIW" \
         MODEL.checkpoint "experiments/CRefNet/crefnet_r_cgi_iiw_448X.pt"
      ```  
   5. Get the trained model and rename it to ```experiments/CRefNet/crefnet_s_cgi_iiw.pt```
   6. Evaluate:
      + <details>
        <summary>(click to expand)</summary>
       
        ##### 
        ```console
         CUDA_VISIBLE_DEVICES="0" python evaluate.py \
           --cfg="configs/crefnet_S_cgi_iiw.yaml" \
           MODEL.checkpoint "experiments/CRefNet/crefnet_s_cgi_iiw.pt" \
           MODEL.input_img_size 224 \
           TEST.dataset SAW
        ```
        </details>
5. Finetune the shading estimation branch on the CGI+IIW datasets using 448X448 images:
   1. The training config file: ```configs/crefnet_S_cgi_iiw_448X.yaml```
   2. Run
      ```console
         CUDA_VISIBLE_DEVICES="0" python train.py \
         --cfg="configs/refine_crefnet_S_cgi_iiw_448X.yaml" \
         DIR  "./experiments/CRefNet/S/CGI+IIW_448X" \
         MODEL.checkpoint "experiments/CRefNet/crefnet_s_cgi_iiw.pt"
      ```
   3. Get the trained model and rename it to ```experiments/CRefNet/crefnet_s_cgi_iiw_448X.pt```
   4. Evaluate:
      + <details>
        <summary>(click to expand)</summary>
       
        ##### 
        ```console
         CUDA_VISIBLE_DEVICES="0" python evaluate.py \
           --cfg="configs/refine_crefnet_S_cgi_iiw_448X.yaml" \
           MODEL.checkpoint "experiments/CRefNet/crefnet_s_cgi_iiw_448X.pt" \
           MODEL.input_img_size 448 \
           TEST.dataset SAW
        ```
        </details>
 

### Efficient version
The efficient version (crefnet-e.pt) is trained in the above steps with the architecture option ```MODEL.ARCH.name "crefnet-e"```
