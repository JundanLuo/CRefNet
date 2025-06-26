Evaluation
- 
Download the trained models into the ```<project_root>/trained_models/``` folder.

### IIW benchmark
We report the WHDR score in the sRGB space in the paper.
+ Evaluate the main model ```final_real.pt```:
  + ```console
    CUDA_VISIBLE_DEVICES="0" python evaluate.py \
        --cfg="configs/crefnet.yaml" \
        MODEL.checkpoint "./trained_models/final_real.pt" \
        TEST.dataset IIW \
        TEST.vis_per_iiw 1 \
        TEST.save_raw_pred True  # set it False if you do not need the float32 raw predictions
    ```
  + We provide the precomputed results in linear RGB space: [png images](https://drive.google.com/file/d/14WUYdcERhNum4dtvxrVCSruWUhxeIK2H/view?usp=sharing) and [float32 predictions](https://drive.google.com/file/d/1DemA6gPa5sPiLPrYMuFZwhUquZDthlFX/view?usp=sharing). 
+ Evaluate the efficient variant ```crefnet-e.pt```:
  + <details>
    <summary>(click to expand)</summary>
    
    ##### 
    ```console
    CUDA_VISIBLE_DEVICES="0" python evaluate.py \
        --cfg="configs/crefnet-e.yaml" \
        MODEL.checkpoint "./trained_models/crefnet-e.pt" \
        TEST.dataset IIW \
        TEST.vis_per_iiw 1 \
        TEST.save_raw_pred True # set it False if you do not need the float32 raw predictions
    ```
    </details>
  + We provide the precomputed results in linear RGB space: [png images](https://drive.google.com/file/d/15LmHI5uAqAApOIhGBkjLmNRcZwHjPuDL/view?usp=sharing) and [float32 predictions](https://drive.google.com/file/d/1DHLpQcO1BhGH50CbUIU5NlxS5oeq2Zvc/view?usp=sharing).


### SAW benchmark
+ Evaluate the main model ```final_real.pt```:
  + ```console
    CUDA_VISIBLE_DEVICES="0" python evaluate.py \
        --cfg="configs/crefnet.yaml" \
        MODEL.checkpoint "./trained_models/final_real.pt" \
        TEST.dataset SAW
    ```  
+ Evaluate the efficient variant ```crefnet-e.pt```:
  + <details>
    <summary>(click to expand)</summary>
    
    ##### 
    ```console
    CUDA_VISIBLE_DEVICES="0" python evaluate.py \
        --cfg="configs/crefnet-e.yaml" \
        MODEL.checkpoint "./trained_models/crefnet-e.pt" \
        TEST.dataset SAW
    ```
    </details>

    
### MIT benchmark
+ Extract the MIT test data: 
  + <details>
    <summary>(click to expand)</summary>
    
    ##### 
    ```console
    python tools/extract_MIT_test_data.py \
        --input_dir "./data/MIT-intrinsic/data" \
        --output_dir "./data/MIT_test_extracted"
    ```
    </details>
+ Infer on the MIT test data:
  + <details>
    <summary>(click to expand)</summary>
    
    ##### 
    ```console
    CUDA_VISIBLE_DEVICES="0" python infer.py \
        --img-dir ./data/MIT_test_extracted \
        --out-dir ./experiments/MIT_test_out \
        --loading_mode "MIT_test" \
        --output_original_size \
        --cfg="configs/crefnet.yaml" \
        MODEL.checkpoint "./trained_models/model_MIT.pt"
    ```
    </details>
  + We provide the precomputed results: [download link](https://drive.google.com/file/d/1DlgvJlZ9YRwvStXEuI0VyfXpYDoq_1I1/view?usp=sharing).
+ Quantitative evaluation: download the evaluation code from the repository [IntrinsicImage](https://github.com/fqnchina/IntrinsicImage/tree/master/evaluation).
Use the MATLAB script ```compute_MIT_error.m``` and modify the corresponding ground truth and prediction paths.

### MPI-Sintel benchmark
+ We adopt the two-fold validation for the MPI benchmark:
  + Evaluate ```model_MPI_scene_front.pt``` on  ```MPI_main_sceneSplit-fullsize-NoDefect-test.txt```:
    + <details>
      <summary>(click to expand)</summary>
    
      ##### 
      ```console
      CUDA_VISIBLE_DEVICES="0" python evaluate.py \
        --cfg="configs/mpi_benchmark/crefnet_S_mpi_scene_front.yaml" \
        DIR "experiments/MPI_out/" \
        MODEL.checkpoint \
        "./trained_models/model_MPI_scene_front.pt" \
        TEST.dense_task "R" \
        TEST.batch_size_cgi 4 \
        TEST.workers_cgi 4 \
        TEST.vis_per_mpi 100 \
        TEST.save_raw_pred True
      ```
      </details>
  + Evaluate ```model_MPI_scene_reverse.pt``` on  ```MPI_main_sceneSplit-fullsize-NoDefect-train.txt```:
    + <details>
      <summary>(click to expand)</summary>
    
      ##### 
      ```console
      CUDA_VISIBLE_DEVICES="0" python evaluate.py \
          --cfg="configs/mpi_benchmark/crefnet_S_mpi_scene_reverse.yaml" \
          DIR "experiments/MPI_out/" \
          MODEL.checkpoint \
          "./trained_models/model_MPI_scene_reverse.pt" \
          TEST.dense_task "R" \
          TEST.batch_size_cgi 4 \
          TEST.workers_cgi 4 \
          TEST.vis_per_mpi 100 \
          TEST.save_raw_pred True
      ```
      </details>    
  + ```TEST.dense_task```: ```"R"```(reflectance) or ```"S"```(shading)
+ Average the printed results of the two folds.


### Model complexity
+ Assess the model complexity of CRefNet: ```python model_complexity.py --cfg="configs/crefnet.yaml"```
  + CRefNet-E: ```python model_complexity.py --cfg="configs/crefnet-e.yaml"```
