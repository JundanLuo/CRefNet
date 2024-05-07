Train
-
Our main model is trained in the following steps:
1. Train the reflectance estimation branch on the CGI dataset using 224X224 images:
   1. The training config file: ```configs/crefnet_R_cgi.yaml```
      2. ```console
          CUDA_VISIBLE_DEVICES="2" python train.py \
            --cfg="configs/crefnet_R_cgi.yaml" \
            DIR "./experiments/CRefNet/R/CGI"
         ```
2. Finetune the reflectance estimation branch on the CGI+IIW datasets using 224x224 images:
   1. The training config file: ```configs/crefnet_R_cgi_iiw.yaml```
3. Finetune the reflectance estimation branch on the CGI+IIW datasets using 448x448 images:
   1. The training config file: ```configs/crefnet_R_cgi_iiw_448X.yaml```
4. Train the shading estimation branch on the CGI+IIW dataset using 224x224 images:
   1. The training config file: ```configs/crefnet_S_cgi_iiw.yaml```
5. Finetune the shading estimation branch on the CGI+IIW datasets using 448X448 images:
   1. The training config file: ```configs/crefnet_S_cgi_iiw_448X.yaml```