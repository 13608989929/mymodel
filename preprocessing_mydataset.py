import os
from Bio.PDB import PDBParser, PDBIO, Select
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import SDWriter
from math import sqrt
import shutil
from tqdm import tqdm
import pickle
import pandas as pd

class LigandSelect(Select):
    def __init__(self, ligand_resnames):
        self.ligand_resnames = ligand_resnames

    def accept_residue(self, residue):
        return residue.get_resname() in self.ligand_resnames

class ProteinSelect(Select):
    def __init__(self, exclude_resnames):
        self.exclude_resnames = exclude_resnames

    def accept_residue(self, residue):
        return residue.get_resname() not in self.exclude_resnames

class PocketSelect(Select):
    def __init__(self, atom_coords, structure, cutoff=8.0, ligand_resnames=("UNK",), metal_resnames=("ZN", "CA", "MG", "FE", "MN", "CU", "CO", "NI")):
        self.cutoff = cutoff
        self.ligand_resnames = ligand_resnames
        self.metal_resnames = metal_resnames

        # 1. 收集所有距离小分子8Å内的残基
        self.pocket_residues = set()
        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.get_resname().strip()
                    if resname in ligand_resnames:
                        continue
                    for atom in residue:
                        for lig_atom in atom_coords:
                            try:
                                dx = atom.coord[0] - lig_atom.coord[0]
                                dy = atom.coord[1] - lig_atom.coord[1]
                                dz = atom.coord[2] - lig_atom.coord[2]
                                dist = sqrt(dx*dx + dy*dy + dz*dz)
                                if dist <= cutoff:
                                    self.pocket_residues.add(residue)
                                    break
                            except:
                                continue

    def accept_residue(self, residue):
        return residue in self.pocket_residues

def extract_ligand_coords_from_pdb(pdb_file, ligand_resnames):
    lig_atoms = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('HETATM'):
                resname = line[17:20].strip()
                if resname in ligand_resnames:
                    atom_name = line[12:16].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    element = line[76:78].strip() if len(line) >= 78 else atom_name[0]
                    class SimpleAtom:
                        def __init__(self, name, element, coord):
                            self.name = name
                            self.element = element
                            self.coord = coord
                        def get_name(self):
                            return self.name
                        def __sub__(self, other):
                            dx = self.coord[0] - other.coord[0]
                            dy = self.coord[1] - other.coord[1]
                            dz = self.coord[2] - other.coord[2]
                            return sqrt(dx*dx + dy*dy + dz*dz)
                    atom = SimpleAtom(atom_name, element, (x, y, z))
                    lig_atoms.append(atom)
    return lig_atoms

def write_protein_and_ligand(pdb_file, output_dir, ligand_resnames=("UNK",)):
    # 新增：为每个pdb文件创建独立子目录
    base = os.path.splitext(os.path.basename(pdb_file))[0]
    pdb_subdir = os.path.join(output_dir, base)
    os.makedirs(pdb_subdir, exist_ok=True)

        # 新增：复制原始pdb文件到子目录
    pdb_copy_path = os.path.join(pdb_subdir, os.path.basename(pdb_file))
    shutil.copy2(pdb_file, pdb_copy_path)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_file)

    # 1. Write protein
    protein_out = os.path.join(pdb_subdir, f"{base}_protein.pdb")
    io = PDBIO()
    io.set_structure(structure)
    io.save(protein_out, select=ProteinSelect(exclude_resnames=ligand_resnames))
    #print(f"1\Protein written to {protein_out}")
    
    # 2. Write ligand as PDB（保留键信息）
    ligand_hetatm_lines = []
    ligand_conect_lines = []
    ligand_serials = set()

    # 先收集所有UNK的HETATM行和原子编号
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('HETATM') and any(resname in line for resname in ligand_resnames):
                ligand_hetatm_lines.append(line)
                serial = int(line[6:11])
                ligand_serials.add(serial)

    # 再收集相关的CONECT行
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('CONECT'):
                nums = [int(x) for x in line.split()[1:]]
                main_serial = int(line[6:11])
                if main_serial in ligand_serials:
                    ligand_conect_lines.append(line)

    # 写入新PDB文件
    ligand_pdb_out = os.path.join(pdb_subdir, f"{base}_ligand.pdb")
    with open(ligand_pdb_out, 'w') as f:
        for line in ligand_hetatm_lines:
            f.write(line)
        for line in ligand_conect_lines:
            f.write(line)
        f.write("END\n")
    #print(f"2\Ligand written to {ligand_pdb_out}")
    
    # 3. Write pocket within 8 Å
    pocket_out = os.path.join(pdb_subdir, f"{base}_pocket.pdb")
    lig_atoms = extract_ligand_coords_from_pdb(pdb_file, ligand_resnames)
    if lig_atoms:
        io.set_structure(structure)
        io.save(pocket_out, select=PocketSelect(lig_atoms, structure, cutoff=8.0, ligand_resnames=ligand_resnames))
        #print(f"3\Pocket written to {pocket_out}")
    else:
        print(f"No ligand atoms found for pocket extraction in {pdb_file}")


def batch_process(pdb_dir, output_dir, ligand_resnames=("UNK",)):
    os.makedirs(output_dir, exist_ok=True)
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
    failed_files = []
    for i, pdb_file in enumerate(pdb_files):
        try:
            full_path = os.path.join(pdb_dir, pdb_file)
            write_protein_and_ligand(full_path, output_dir, ligand_resnames=ligand_resnames)
            print(f"✅ Processed: {pdb_file} ({i+1}/{len(pdb_files)})")
        except Exception as e:
            print(f"❌ Failed: {pdb_file} due to {e}")
            failed_files.append(pdb_file)
    print("\n处理失败的pdb文件列表：")
    print(failed_files)



#rdkit文件获取
def generate_complex(data_dir,data_df,distance=8):
    pbar = tqdm(total=len(data_df))
    for _,row in data_df.iterrows():
        cid,pka = row['pdbid'],float(row['logKd/Ki'])
        ligand_path = os.path.join(data_dir,str(cid),f'{cid}_ligand.pdb')
        pocket_path = os.path.join(data_dir,str(cid),f'{cid}_pocket.pdb')

        save_path = os.path.join(data_dir,str(cid),f'{cid}_{distance}A.rdkit')
        ligand = Chem.MolFromPDBFile(ligand_path,removeHs=True)
        if ligand == None:
            print(f'Unable to process ligand of {cid}')
            continue
        
        pocket = Chem.MolFromPDBFile(pocket_path,removeHs=True)
        if pocket == None:
            print(F'Unable to process pocket of {cid}')
            continue

        complex = (ligand,pocket)
        with open(save_path,'wb') as f:
            pickle.dump(complex,f)

        pbar.update(1)


if __name__ == "__main__":
    #1.口袋提取
    # batch_process(
    #     pdb_dir="collected_pdbs",
    #     output_dir="processed_pdbs",
    #     ligand_resnames=("UNK",)
    # )

    #2.rdkit文件获取
    distance = 8
    data_root = './data/test/'
    data_dir = os.path.join(data_root,'rdkit_get')
    data_df = pd.read_csv(os.path.join(data_dir,'mydataset_test.csv'),dtype={'pdbid':str})

    generate_complex(data_dir,data_df,distance=distance)
