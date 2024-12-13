
<div align="center">
<h1> Prompt2Perturb (P2P): Text-Guided Diffusion-Based Adversarial Attacks on Breast Ultrasound Images </h1>
<h3>

<image src="images/attack paper_final_version-1.png" width="600">
</div>

<div align="center"> <h3> Abstract </h3>  </div>
<div align="justify">

</br>
*Deep neural networks (DNNs)  offer significant promise for improving breast cancer diagnosis in medical imaging. However, these models are highly susceptible to adversarial attacks—small, imperceptible changes that can mislead classifiers—raising critical concerns about their reliability and security. Traditional attacks rely on fixed-norm perturbations, misaligning with human perception. In contrast, diffusion-based attacks require pre-trained models, demanding substantial data when these models are unavailable, limiting practical use in data-scarce scenarios. In medical imaging, however, this is often unfeasible due to the limited availability of datasets. Building on recent advancements in learnable prompts, we propose Prompt2Perturb (P2P), a novel language-guided attack method capable of generating meaningful attack examples driven by text instructions. During the prompt learning phase, our approach leverages learnable prompts within the text encoder to create subtle, yet impactful, perturbations that remain imperceptible while guiding the model towards targeted outcomes.
In contrast to current prompt learning-based approaches, our P2P stands out by directly updating text embeddings, avoiding the need for retraining diffusion models. Further, we leverage the finding that optimizing only the early reverse diffusion steps boosts efficiency while ensuring that the generated adversarial examples incorporate subtle noise, thus preserving ultrasound image quality without introducing noticeable artifacts. We show that our method outperforms state-of-the-art attack techniques across three breast ultrasound datasets in FID and LPIPS. Moreover, the generated images are both more natural in appearance and more effective compared to existing adversarial attacks.*
</br>
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
# Training Classifiers
You can train the classifiers using the command below:
```bash
  python evaluate_new.py --data_path <dir_of_dataset_local>\
   --data_test_path <path_of_test_dataset>\
   --data_valid_path <path_of_validation_set>\
   --output_path <dir_of_output>\
   --adjective_list "" \
   --baselines 'densenet121','resnet34','squeezenet1.1'\
   --adjective_flag 0\
   --batch_size 32\
   --num_class 3\
   --num_epochs 100\
   --train 0\
   --output_file_name <name_of_result_text_and_table>\
   --size 224
```


