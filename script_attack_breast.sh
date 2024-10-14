#!/bin/bash
#SBATCH --job-name=5dens # Specify the job name
#SBATCH --account=st-ilker-1-gpu   # Specify your allocation code
#SBATCH --nodes=1              # Request 1 node
#SBATCH --ntasks=1             # Request 1 task
#SBATCH --gpus-per-node=4
#SBATCH --mem=32G               # Request 32 GB of memory
#SBATCH --time=24:00:00        # Request 10 hours of runtime
#SBATCH --constraint=gpu_mem_32
#SBATCH -e /scratch/st-ilker-1/yasamin/logs2/error_0.5-%j.txt    # Specify the error file. The %j will be replaced by the Slurm job id.
#SBATCH -o /scratch/st-ilker-1/yasamin/logs2/output_0.5-%j.txt      # Specify the output file

 
# Load necessary software modules


source /arc/project/st-ilker-1/yasamin/diffusers/examples/text_to_image/lora/bin/activate
 
# "no_adjective","colorful","stylized","high-contrast","low-contrast","posterized","solarized","sheared","bright","dark"
#  --baselines 'resnet34','squeezenet1.1','densenet121'\ 
# Add your executable commands here
#./your_executable
export HF_HOME=/scratch/st-ilker-1/yasamin/huggingface/
export TORCH_HOME=/scratch/st-ilker-1/yasamin/



# export MODEL_NAME="CompVis/stable-diffusion-v1-4"
#export MODEL_NAME=""/scratch/st-ilker-1/yasamin/huggingface/hub/split_dataset_fused"
# export MODEL_NAME="/scratch/st-ilker-1/yasamin/Textual_attack/dreamnooth_checkpoints_splits/split2/normal_ckpt"
# #export DATA_DIR="/arc/project/st-ilker-1/yasamin/corrupted_examples"
# export DATA_DIR="/arc/project/st-ilker-1/yasamin/Breast_Data/Dataset/Data_breast_splits/split_2/train/normal"
# #export DATA_DIR="/arc/project/st-ilker-1/yasamin/Breast_Data/Dataset/Original/train/benign" # name of class desk
# nvidia-smi
# accelerate launch /scratch/st-ilker-1/yasamin/Textual_attack/textual_inversion_new_breast.py\
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --train_data_dir=$DATA_DIR \
#   --learnable_property="object" \
#   --placeholder_token="<no>" --initializer_token="no" \
#   --resolution=224 \
#   --train_batch_size=1 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=6000 \
#   --learning_rate=5.0e-04 --scale_lr \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --output_dir="textual_inversion_normal_split2"\
#   --saving_image_dir "/scratch/st-ilker-1/yasamin/Textual_attack/images_splits/split2/normal"
  #
# Loop over the splits


# Array of splits

#"resnet34" "squeezenet1.1" "densenet121"
#"malignant" "normal"
#"2" "3" "4" "5"
splits=( "5" )
classes=( "benign" "malignant" "normal")
classifiers=("densenet121" )

# Loop over the splits
for split in "${splits[@]}"; do
  for class in "${classes[@]}"; do
    for classifier in "${classifiers[@]}"; do
      #export MODEL_NAME="/scratch/st-ilker-1/yasamin/Textual_attack/dreamnooth_checkpoints_splits/split${split}/${class}_ckpt"
      export MODEL_NAME="CompVis/stable-diffusion-v1-4"
      export DATA_DIR="/arc/project/st-ilker-1/yasamin/Breast_Data/Dataset/Data_breast_splits/split_${split}/train/${class}"
      #export DATA_DIR="/arc/project/st-ilker-1/yasamin/Breast_missing_images/resnet34/split2/benign"
      export OUTPUT_DIR="textual_inversion_${class}_split_new${split}"
      #export SAVING_DIR="/scratch/st-ilker-1/yasamin/Textual_attack/images_splits_new/${classifier}/split${split}/${class}"
      export SAVING_DIR="/scratch/st-ilker-1/yasamin/Textual_attack/images_splits_no_pt/${classifier}/split${split}/${class}"
      export Classifier_path="/scratch/st-ilker-1/yasamin/Breast/result_split_${split}/result_original"
      export csv_path="/scratch/st-ilker-1/yasamin/Textual_attack/images_splits_new/${classifier}/split${split}/${class}/attack_info.csv"

      # Remove the existing saving directory if it exists
      # if [ -d "$SAVING_DIR" ]; then
      #   rm -r "$SAVING_DIR"
      # fi
      if [ -d "$OUTPUT_DIR" ]; then
        rm -r "$OUTPUT_DIR"
      fi

      if [ -d "$csv_path" ]; then
        rm -r "$csv_path"
      fi
      echo "Running split ${split} for class ${class} using classifier ${classifier}"
      nvidia-smi

      accelerate launch /scratch/st-ilker-1/yasamin/Textual_attack/textual_inversion_new_breast_loop.py \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --train_data_dir=$DATA_DIR \
        --learnable_property="object" \
        --placeholder_token="<${class}>" --initializer_token="${class}" \
        --resolution=224 \
        --train_batch_size=1 \
        --gradient_accumulation_steps=4 \
        --max_train_steps=6000 \
        --learning_rate=5.0e-04 --scale_lr \
        --lr_scheduler="constant" \
        --lr_warmup_steps=0 \
        --output_dir=$OUTPUT_DIR \
        --saving_image_dir=$SAVING_DIR \
        --class_name="${class}" \
        --classifier_name="${classifier}" \
        --classifier_path=$Classifier_path \
        --csv_path=$csv_path
    done
  done
done
#5.0e-04
#--placeholder_token="<${class}>" --initializer_token="${class}"