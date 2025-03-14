
<div align="center">
<h1>
    <a href="https://arxiv.org/abs/2412.09910" target="_blank">
        Prompt2Perturb (P2P): Text-Guided Diffusion-Based Adversarial Attacks on Breast Ultrasound Images (CVPR 2025)
    </a>
</h1>

 <h3>

<image src="images/attack paper_final_version-1.png" width="600">
</div>

<div align="center"> <h3> Abstract </h3>  </div>
<div align="justify">

<br>

*Deep neural networks (DNNs)  offer significant promise for improving breast cancer diagnosis in medical imaging. However, these models are highly susceptible to adversarial attacks—small, imperceptible changes that can mislead classifiers—raising critical concerns about their reliability and security. Traditional attacks rely on fixed-norm perturbations, misaligning with human perception. In contrast, diffusion-based attacks require pre-trained models, demanding substantial data when these models are unavailable, limiting practical use in data-scarce scenarios. In medical imaging, however, this is often unfeasible due to the limited availability of datasets. Building on recent advancements in learnable prompts, we propose Prompt2Perturb (P2P), a novel language-guided attack method capable of generating meaningful attack examples driven by text instructions. During the prompt learning phase, our approach leverages learnable prompts within the text encoder to create subtle, yet impactful, perturbations that remain imperceptible while guiding the model towards targeted outcomes.
In contrast to current prompt learning-based approaches, our P2P stands out by directly updating text embeddings, avoiding the need for retraining diffusion models. Further, we leverage the finding that optimizing only the early reverse diffusion steps boosts efficiency while ensuring that the generated adversarial examples incorporate subtle noise, thus preserving ultrasound image quality without introducing noticeable artifacts. We show that our method outperforms state-of-the-art attack techniques across three breast ultrasound datasets in FID and LPIPS. Moreover, the generated images are both more natural in appearance and more effective compared to existing adversarial attacks.*

<br>

<div align="center">
<image src="images/Perturbation Figure color.png" width="600">
</div>

## 💥 News 💥
- **`26.02.2025`** | Accepted in CVPR 2025! 🥳
- **`10.12.2024`** | Code is released!

# Getting started

## Requirements
The code requires Python 3.8 or later. The file [requirements.txt](requirements.txt) contains the full list of required Python modules.
```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
cd examples/text_to_image
pip install -r requirements.txt
```
## Resources
The code was tested on a GeForce RTX 3080 32GB but should work on other cards with at least 32GB VRAM.
## Training Classifiers
You can train the classifiers using the command below:
```bash
python evaluate.py --data_path <dir_of_dataset_local>\
 --data_test_path <path_of_test_dataset>\
 --data_valid_path <path_of_validation_set>\
 --output_path <dir_of_output>\
 --adjective_list "" \
 --baselines 'densenet121','resnet34','squeezenet1.1'\
 --adjective_flag 0\
 --batch_size 32\
 --num_class 2\
 --num_epochs 100\
 --train 0\
 --output_file_name <name_of_result_text_and_table>\
 --size 224
```
Each of the dataset directories should be formatted so that each class has its own directory of images. Then the `--num_class` argument should reflect the number of classes in the dataset. For instance, if there is a 'benign' class and a 'malignant' class, then within each data directory should be a 'benign' directory consisting of benign images and a 'malignant' directory consisting of malignant images with `--num_class` set to 2.
- `dir_of_dataset_local` path to training dataset
- `path_of_test_dataset` path to test dataset
- `path_of_validation_set` path to validation dataset
- `dir_of_output` location to save each of the classifiers
- `name_of_result_text_and_table` name of file to display classifier training output (will output to same path of `dir_of_output`)
## P2P
You can generate adversarial attack by our pipeline using the command below:
```bash
  accelerate launch /P2P/P2P.py \
    --pretrained_model_name_or_path=<path_to_pretrained_model> \
    --train_data_dir=<data_dir> \
    --learnable_property="object" \
    --placeholder_token="<${class}>" --initializer_token="${class}" \
    --resolution=224 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=6000 \
    --learning_rate=5.0e-04 --scale_lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --output_dir=<checkpoint_path> \
    --saving_image_dir=<image_save_path> \
    --class_name=<class> \
    --classifier_name=<classifier> \
    --classifier_path=<path_classifier> \
    --csv_path=<path_csv> \
    --num_class=2
```
- `path_to_pretrained_model` path to pre-trained stable diffusion model
- `train_data_dir` path to images to apply attack to
- `class` name of class to apply attacks to
- `output_dir` where the model predictions and checkpoints will be written
- `image_save_path` directory for attacked images
- `classifier` name of classifier
- `path_classifier` path where the classifier resides
- `path_csv` path to save csv output of training metrics 

## Acknowledgement
We thank the authors of [huggingface diffusers library](https://github.com/huggingface/diffusers) for their code repository.


## Citation
```
@article{medghalchi2024prompt2perturb,
  title={Prompt2Perturb (P2P): Text-Guided Diffusion-Based Adversarial Attacks on Breast Ultrasound Images},
  author={Medghalchi, Yasamin and Heidari, Moein and Allard, Clayton and Sigal, Leonid and Hacihaliloglu, Ilker},
  journal={arXiv preprint arXiv:2412.09910},
  year={2024}
}
```
