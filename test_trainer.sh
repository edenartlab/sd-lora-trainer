# Set GPU ID to run these jobs on:
GPU_ID="device=0"

python main.py train_configs/training_args_face_sdxl.json
python main.py train_configs/training_args_face_sd15.json
python main.py train_configs/training_args_object.json
python main.py train_configs/training_args_style_sd15.json
python main.py train_configs/training_args_style_sdxl.json
