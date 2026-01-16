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

<p align="center">
  <img width="800" src="Interpretability for imagined speech.png">
</p>

### Data Download and set up environments

 - Download [SEED](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html)
 - set seed_path in run.sh file to its respective path and run the following command.

```
bash run.sh
```

- in the above shell file, optional modulator argument activates the modulator

### Citation

If this code is helpful for your research, please cite us at:

```
@article{kang2025CCMTL,
  title={Convolutional Channel Modulator for Transformer and LSTM Networks in EEG-based Emotion Recognition},
  author={Kang, Hyunwook and Choi, Jin Woo and Kim, Byung Hyung},
  journal={Biomedical Engineering Letters},
  volume={15},
  pages={749-761}
  year={2025}
}
```

### Contact

For any questions, please email at [hyunwook.kang@inha.edu](mailto:hyunwook.kang@inha.edu)
