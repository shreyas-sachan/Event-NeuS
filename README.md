# EventNeuS (3DV 2026)

Official code for the paper "EventNeuS: 3D Mesh Reconstruction from a Single Event Camera" (3DV 2026). Built on top of [EventNeRF](https://github.com/r00tman/EventNeRF) and [NeuS](https://github.com/Totoro97/NeuS).

## Data

Download the synthetic datasets from our Nextcloud **[here](https://nextcloud.mpi-inf.mpg.de/index.php/s/7cwjx3Dd9DMjneZ)**. The synthetic data is split into 5 zip files (`train.zip`, `test.zip`, `validation.zip`, `inference.zip`, `blender_scenes.zip`).

For the real-world scenes, please download the real dataset from the original EventNeRF repository [here](https://nextcloud.mpi-klsb.mpg.de/index.php/s/xDqwRHiWKeSRyes). Place the downloaded datasets in your preferred directory and extract them (e.g. `data/`).

## Create environment

```bash
conda env create --file environment1.yml
conda activate event_neus

# Install additional dependencies
pip install ninja
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install git+https://github.com/JustusThies/PyMarchingCubes
```

## Training and Testing

Please replace `<PATH_TO_DATASET>` and `<PATH_TO_LOGS>` in the corresponding `.txt` config files.

Use the `ddp_train_neus.py` and `ddp_test_neus.py` scripts for training and testing:
```bash
python ddp_train_neus.py --config configs/synthetic/chair.txt
python ddp_test_neus.py --config configs/synthetic/chair.txt
```

## Models

 - `configs/synthetic/*` -- synthetic data (chair, drums, hotdog, lego, mic)
 - `configs/real/*` -- real data (bottle, chicken, controller, cube, dragon, microphone, multimeter, plant, sewing, tapes)

## Mesh Extraction

To extract the mesh from a trained model, run:

```bash
python ddp_mesh_neus.py --config configs/synthetic/chair.txt
```

Replace `configs/synthetic/chair.txt` with the path to your trained model config. The extracted mesh will be saved in `<PATH_TO_LOGS>/<expname>/mesh/`.

## SLURM

We provide launcher scripts in the `scripts/` folder for submitting jobs on SLURM clusters. Update the placeholders in `scripts/launcher_job.sh` (`<YOUR_GPU_PARTITION>`, `<YOUR_CONDA_ENV>`, `<YOUR_SLURM_LOG_DIR>`) and run:
```bash
./scripts/launcher.sh python ddp_train_neus.py --config configs/real/bottle.txt
```
See `scripts/launch_ablations.sh` for batch launching examples.

## Citation

Please cite our work if you use the code.

```
@article{sachan2026eventneus,
  title={EventNeuS: 3D Mesh Reconstruction from a Single Event Camera},
  author={Sachan, Shreyas and Rudnev, Viktor and Elgharib, Mohamed and Theobalt, Christian and Golyanik, Vladislav},
  journal={arXiv preprint arXiv:2602.03847},
  year={2026}
}
```

## License

This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. To view a copy of this license, visit [http://creativecommons.org/licenses/by-nc-sa/4.0/](http://creativecommons.org/licenses/by-nc-sa/4.0/) or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
