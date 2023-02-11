# Contrastive Active Learning (CAL) 

(`-> Original`)[https://github.com/mourga/contrastive-active-learning/]

modified version to make the dataloaders work for the `ORNL` datasets. There were some major problems with version mismatching especially around the `datasets` package which does not exist on this version.

**IMPORTANT** ORNL may only work on certain systems after you pre-download the files manually and place them in the `data/ORNL8`/`data/ORNL26` folders (`test.csv`, `train.csv` and `classes.txt`)

Modified files:
 - `/aquisition`
 - `/analysis` 
 - `/utilities`
 - `run_al.py`
 - `sys_config.py`
