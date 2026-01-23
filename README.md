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


Settings for different data: <br>
 -BCIComp: --num_electrodes 64 --n_subjects 15 <br>
 -FEIS: --num_electrodes 14 --n_subjects 21 --n_classes 2

To plot shap run with the above settings for the respective dataset:
```
python plot_attention_topo.py --dataset_dir [root_data_folder_path] --model_name EEG_Transformer
```

### Citation

If this code is helpful for your research, please cite us at:

```
@article{Kang2026INTE,
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
