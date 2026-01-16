FEIS=/home/Hyunwook/codes/BrainCon-revision/FEIS-v1.1/scottwellington-FEIS-7e726fd/experiments
python main.py --model_name Conformer --dataset_dir $FEIS --num_electrodes 14 --n_subjects 21 --exper-setting indep --save_file_name indep_conformer_results.csv
