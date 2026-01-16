# Interpreting-Imagined-Speech-from-EEG-Reveals-Distinct-Neural-Signatures


<p align="center">
  <img width="800" src="vis.png">
</p>

### Data Download and set up environments

 - Download [FEIS](https://zenodo.org/records/3554128)
 - set FEIS to where the directory '/FEIS-v1.1/scottwellington-FEIS-7e726fd/experiments' is, and run as:

```
python main.py --model_name Conformer --dataset_dir $FEIS --num_electrodes 14 --n_subjects 21 --exper-setting indep --save_file_name indep_conformer_results.csv
```


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


### Citation

If this code is helpful for your research, please cite us at:

```
@article{kang2026,
  title={Interpreting imagined speech from {EEG} reveals distinct neural signatures},
  author={Kang, Hyunwook and Kim, Sejin and Lee, Young-Eun and Lee, Minji},
  journal={Brain Connectivity},
  volume={},
  pages={}
  year={2026}
}
```

### Contact

For any questions, please email at [hyunwook.kang@catholic.ac.kr](mailto:hyunwook.kang@catholic.ac.kr)
