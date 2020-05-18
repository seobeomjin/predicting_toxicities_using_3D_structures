"""
18.05.2020 
Beomjin Seo


Usage Example  
python protein_prepare.py -p dataset/protein/AR_2am9.pdb --save_path dataset/protein/test --num_copies 10

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

parser.add_argument('--protein','-p',required=True, type=input_file, nargs='+',
                    help='files with proteins\' structures')
parser.add_argument('--pocket_format', default='pdb', type=str, help='file format for the pocket')
parser.add_argument('--save_path','-s',default='dataset/protein', type=str, help='please choose one type between train, validation and test.')
parser.add_argument('--num_copies', type=int, required=True, help='number of copies')

args = parser.parse_args()

for protein_file in args.protein : 

    proteintype =os.path.split(os.path.splitext(os.path.split(protein_file)[0])[0])[1]
    name = os.path.splitext(os.path.split(protein_file)[1])[0]
    # os.path.split >>> splitting it two part, root and file name including format
    # os.path.splitext > splitting it two part, root and file format without filename
    save_path = args.save_path +'/'
    
    with open(protein_file ,'r') as inf:
        lines = inf.readlines()
    
    for i in range(args.num_copies):
        outf = open(save_path + name + '_' + str(i) + '.' + args.pocket_format, 'w')
        outf.writelines(lines)
        outf.close()