#!/bin/bash
#SBATCH --job-name=norm5  # Specify the job name
#SBATCH --account=st-ilker-1-gpu   # Specify your allocation code
#SBATCH --nodes=1              # Request 1 node
#SBATCH --ntasks=1             # Request 1 task
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G               # Request 2 GB of memory
#SBATCH --time=2:00:00        # Request 10 hours of runtime
#SBATCH --constraint=gpu_mem_32
#SBATCH -e /scratch/st-ilker-1/yasamin/logs/error_0.5-%j.txt    # Specify the error file. The %j will be replaced by the Slurm job id.
#SBATCH -o /scratch/st-ilker-1/yasamin/logs/output_0.5-%j.txt      # Specify the output file
#SBATCH --mail-user=yasimed@student.ubc.ca  # Email address for job notifications
#SBATCH --mail-type=ALL        # Receive email notifications for all job events
 
# Load necessary software modules


source /arc/project/st-ilker-1/yasamin/diffusers/examples/text_to_image/lora/bin/activate
 
# "no_adjective","colorful","stylized","high-contrast","low-contrast","posterized","solarized","sheared","bright","dark"
#  --baselines 'resnet34','squeezenet1.1','densenet121'\ 
# Add your executable commands here
#./your_executable
export HF_HOME=/scratch/st-ilker-1/yasamin/huggingface/
export TORCH_HOME=/scratch/st-ilker-1/yasamin/

export MODEL_NAME="/scratch/st-ilker-1/yasamin/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"
export INSTANCE_DIR="/project/st-ilker-1/yasamin/Breast_Data/Dataset/Data_breast_splits/split_5/train/normal"
export OUTPUT_DIR="/scratch/st-ilker-1/yasamin/Textual_attack/dreamnooth_checkpoints_splits/split5/normal_ckpt"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="ultrasound image of no tumor in breast" \
  --resolution=224 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \