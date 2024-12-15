# Code for Paper Few-shot Retinal Disease Classification on the Brazilian Multilabel Ophtalmological Dataset

Any questions/comments/issues, please feel free to reach out gabrieljp (at) usp (dot) br.

## Install
To install the necessary dependencies for this repo, run

```
conda env create -f env.yml
```

## Data Preparation
First, it is necessary to add BRSET images and labels path to the beginning of data/preprocess.py.

To clean the data and pre-process the images, run

```
cd data
python preprocess.py c
python preprocess.py p
```

To generate class count before and after cleaning, run

```
cd data
python preprocess.py cdb
python preprocess.py cda
```

## Baseline 1
To run the evaluation procedure in the models pre-trained on ImageNet, run

```
python eval_pretrained.py \
    --model <resnet50.a3_in1k or swin_s3_tiny_224.ms_in1k or vit_small_patch32_224.augreg_in21k_ft_in1k> \
    --save-path <path_to_save_results> \
    --e-learning-rate <evaluation_learning_rate>
    --e-runs <evaluation_number_of_runs>
```

## Baseline 2
To train and evaluate models using baseline 2, run

```
python train_baseline.py \
    --model <resnet50.a3_in1k or swin_s3_tiny_224.ms_in1k or vit_small_patch32_224.augreg_in21k_ft_in1k> \
    --save-path <path_to_save_results> \
    --e-learning-rate <evaluation_learning_rate> \
    --e-runs <evaluation_number_of_runs> \
    --p-learning-rate <pretraining_learning_rate> \
    --p-epochs <pretraining_epochs> \
    --p-batch-size <pretraining_batch_size>
```

## Reptile
To train and evaluate models using Reptile, run

```
python train_baseline.py \
    --model <resnet50.a3_in1k or swin_s3_tiny_224.ms_in1k or vit_small_patch32_224.augreg_in21k_ft_in1k> \
    --save-path <path_to_save_results> \
    --e-learning-rate <evaluation_learning_rate> \
    --e-runs <evaluation_number_of_runs> \
    --ways <N> \
    --shots <K> \
    --augment <basic or cutmix or mixup or leave empy> \
    --augment-where <i or e or i+e or leave empty> \
    --inner-loop-learning-rate <...> \
    --inner-loop-steps <number_of_updates_inner_loop> \
    --outer-loop-learning-rate <...> \
    --outer-loop-updates <number_of_updates_outer_loop> \
    --outer-loop-batch-size <task_batch_size_outer_loop> 

```

## Confidence Distributions

To generate confidence distribution for a specific model, run

```
python confidence.py \
    --model-path <...> \
    --ways <N> \
    --shots <K> \
    --augment <basic or cutmix or mixup or leave empy> \
    --save-path <path_to_save_results> 
```

## Exaplainability

To generate the GradCam heatmaps, run

```
python explain.py \
    --model-path <...> \
    --ways <N> \
    --shots <K> \
    --augment <basic or cutmix or mixup or leave empy> \
    --save-path <path_to_save_results> \
    --image <image_id> \
    --target-class <image_class> \
    --classes <classes_to_evaluate_separated_by_commas>
```

