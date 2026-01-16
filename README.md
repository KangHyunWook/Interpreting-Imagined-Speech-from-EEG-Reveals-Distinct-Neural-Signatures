# Interpreting-Imagined-Speech-from-EEG-Reveals-Distinct-Neural-Signatures

FEIS=/home/Hyunwook/codes/BrainCon-revision/FEIS-v1.1/scottwellington-FEIS-7e726fd/experiments

To train and list accuracies for all subjects, run:
python main.py --model_name [model_name] --dataset_dir [dataset]

Settings for different data:
 -BCIComp: --num_electrodes 64 --n_subjects 15
 -FEIS: --num_electrodes 14 --n_subjects 21


To plot EEGNet GradCam, run:
python grad_cam.py --dataset_dir [dataset_path]



set --dataset_dir to choose data (BCIComp, FEIS)

To plot shap run:

python plot_attention_topo.py --dataset_dir [root_data_folder_path]
