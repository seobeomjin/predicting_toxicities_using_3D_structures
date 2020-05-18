"""
18.05.2020 
Beomjin Seo

Usage Example 
python ligand_prepare.py -l dataset/ligand/estrogenSDF/*.sdf 
"""


import os 

def input_file(path):
    """ Check if input file exists. """

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError("File %s does not exists." %path)
    return path 


import argparse 


parser = argparse.ArgumentParser(description='sdf file preprocess')

parser.add_argument('--ligand','-l',required=True, type=input_file, nargs='+',
                    help='files with ligands\' structures')
parser.add_argument('--ligand_format', default='sdf', type=str, help='file format for the pocket')
parser.add_argument('--save_path','-s',default='dataset/ligand_', type=str, help='please choose one type between train, validation and test.')   
args = parser.parse_args()

for ligand_file in args.ligand : 

    proteintype =os.path.split(os.path.splitext(os.path.split(ligand_file)[0])[0])[1]
    name = os.path.splitext(os.path.split(ligand_file)[1])[0]
    # os.path.split >>> splitting it two part, root and file name including format
    # os.path.splitext > splitting it two part, root and file format without filename
    save_path = args.save_path + '/'
    
    with open(ligand_file ,'r') as inf:
        lines = inf.readlines()
        lines.pop(3)
    
    outf = open(save_path+name+'.'+args.ligand_format,'w')
    outf.writelines(lines)
    outf.close()