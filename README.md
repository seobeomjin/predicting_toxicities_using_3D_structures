# predicting_toxicity_using_3D_structures

# How to use

### version 2 > was network that add FC layers for predicting pIC50 to based model (but need more work)

### version 3 > predicting toxicities using 3D structures
    - source activate pafnucy_env 
    - python ligand_prepare.py -l dataset/ligand/*.sdf --save_path dataset/ligand_/train # also, have to run on valid and test
    - python protein_prepate.py -p dataset/protein/ER_1a52.pdb --save_path dataset/protein/train -num_copies 1302 # also have to run on valid and test 
    - python prepare_3.py -l dataset/ligand_/train/*.sdf --ligand_format sdf -p dataset/protein/train/*.pdb --pocket_format pdb -t dataset/toxicities/exp_toxicities.csv -o dataset/train_valid_test_data/training_set.hdf
    - python training_3.py -i dataset/train_valid_test_data/exp3/ -o dataset/results/exp3_ar_er/"exp3" --num_epochs 50 

### version 4 > (maybe) predicting toxicities using multitask learning with 3D structures , In this case target proteins are Androgen and Estrogen
    - python training_4.py -i dataset/train_valid_test_data/exp3/ -o dataset/results/exp4_multitask/"exp4_(2)" -n dataset/results/exp4_multitask/exp4-2020-05-21T00:21:03-best --num_epochs 1