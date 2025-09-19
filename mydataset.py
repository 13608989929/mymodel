import multiprocessing
import pickle
from rdkit import Chem
from rdkit import RDLogger
import numpy as np
from scipy.spatial import distance_matrix
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data,Batch
from itertools import repeat
import os
import pandas as pd
RDLogger.DisableLog('rdApp.*')

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]



#==============================
# 配体原子特征
#==============================
def atom_features(atom,atom_symbols=['C','N','O','S','F','P','Cl','Br','I','UNK'],explicit_H=True):
    results = one_of_k_encoding_unk(atom.GetSymbol(),atom_symbols) + \
        one_of_k_encoding_unk(atom.GetDegree(),[0,1,2,3,4,5,6]) + \
        one_of_k_encoding_unk(atom.GetImplicitValence(),[0,1,2,3,4,5,6]) + \
        one_of_k_encoding_unk(atom.GetHybridization(),[
            Chem.rdchem.HybridizationType.SP,Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,Chem.rdchem.HybridizationType.UNSPECIFIED
        ]) + \
        one_of_k_encoding_unk(atom.GetIsAromatic(),[False,True])
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    # if explicit_H:
    #     results += one_of_k_encoding_unk(atom.GetTotalNumHs(),[0,1,2,3,4])
        
    return np.array(results).astype(np.float32)

#==============================
# 化学键边特征
#==============================
def bond_features(bond):
    btype = one_of_k_encoding_unk(
        bond.GetBondType(),
        [Chem.rdchem.BondType.SINGLE,
         Chem.rdchem.BondType.DOUBLE,
         Chem.rdchem.BondType.TRIPLE,
         Chem.rdchem.BondType.AROMATIC,
         'UNK']
    )
    stereo = one_of_k_encoding_unk(
        bond.GetStereo(),
        [Chem.rdchem.BondStereo.STEREONONE,
         Chem.rdchem.BondStereo.STEREOZ,
         Chem.rdchem.BondStereo.STEREOE,
         Chem.rdchem.BondStereo.STEREOCIS,
         Chem.rdchem.BondStereo.STEREOTRANS,
         'UNK']
    )
    feats = btype + [bond.GetIsConjugated()] + [bond.IsInRing()] + stereo
    return np.asarray(feats, dtype=np.float32)


def mol2graph(mol):
    #节点特征矩阵
    atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.from_numpy(np.asarray(atom_feats, dtype=np.float32)) 

    #边索引+边特征
    ei = []
    ea = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = torch.from_numpy(bond_features(bond))
        ei.extend([[i,j],[j,i]])
        ea.extend([bf,bf])

    if len(ei) == 0:
        edge_index = torch.empty((2,0),dtype=torch.long)
        edge_attr = torch.empty((0,1),dtype=torch.float32)
    else:
        edge_index = torch.tensor(ei,dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(ea,dim=0)
    return x,edge_index,edge_attr


#==============================
# 残基粒度：节点与残基-残基边
#==============================
AA3_LIST = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','UNK']
AA_POS = {'ARG','LYS','HIS'}
AA_NEG = {'ASP','GLU'}

def _get_residue_key(atom):
    info = atom.GetPDBResidueInfo()
    if info is None:
        return None,None
    chain = (info.GetChainId() or '').strip()
    resn = int(info.GetResidueNumber())
    #不必添加，因为都并不存在此码
    icode = (info.GetInsertionCode() or '').strip()
    name = (info.GetResidueName() or '').strip().upper()
    if len(name) == 1 or name not in AA3_LIST:
        name = 'UNK'
    return (chain,resn,icode),name


# 原子聚合到残基，生成残基节点特征与位置
def pocket_to_residue_nodes(pocket):
    conf = pocket.GetConformer()
    groups = {}
    for atom in pocket.GetAtoms():
        rk,rn = _get_residue_key(atom)
        if rk is None:
            return None
        groups.setdefault(rk,{'resname':rn,'atoms':[]})
        groups[rk]['atoms'].append(atom.GetIdx())

    res_keys = list(groups.keys())
    res2idx = {k:i for i,k in enumerate(res_keys)}

    x_list,pos_list = [],[]
    #氨基酸特征提取
    for k in res_keys:
        rn = groups[k]['resname']
        aa_oh = torch.tensor(one_of_k_encoding_unk(rn,AA3_LIST),dtype=torch.float32)
        if rn in AA_POS:
            ch = torch.tensor([1.,0.,0.])
        elif rn in AA_NEG:
            ch = torch.tensor([0.,1.,0.])
        else:
            ch = torch.tensor([0.,0.,1.])
        x_list.append(torch.cat([aa_oh,ch],dim=0))

        ats = groups[k]['atoms']
        coords = []
        for aid in ats:
            p = conf.GetAtomPosition(aid)
            #非氢
            if pocket.GetAtomWithIdx(aid).GetAtomicNum()>1:
                coords.append([p.x,p.y,p.z])

        ##提取的是氨基酸残基所有原子的中心位置→是否需要选择氨基酸的Cα作为中心 
        pos_list.append(torch.tensor(np.mean(coords,axis=0),dtype=torch.float32))
     
    x_res = torch.stack(x_list,dim=0) if x_list else torch.zeros((0,len(AA3_LIST)+3),dtype=torch.float32)
    pos_res = torch.stack(pos_list,dim=0) if pos_list else torch.zeros((0,3),dtype=torch.float32)
    return res_keys,res2idx,x_res,pos_res

#残基-残基边 + 边特征（现在的边是根据距离连接的(已知Cα-Cα之间的距离都是3.8A，所以阈值可以先定5A左右)）
def residue_graph_edges(res_keys,pos_res,res_edge_threshold=5.0):
    if pos_res.numel == 0:
        ei = torch.empty((2,0),dtype=torch.long)
        ea = torch.empty((0,5),dtype=torch.float32)
        return ei,ea
    p = pos_res.cpu().numpy()
    D = distance_matrix(p,p)
    I, J = np.where((D < res_edge_threshold) & (D > 1e-6))
    idx,feats = [],[]
    for i,j in zip(I,J):
        d = D[i,j].astype(np.float32)
        same_chain = float(res_keys[i][0]== res_keys[j][0])
        seq_sep = float(abs(res_keys[i][1] - res_keys[j][1]))
        f = torch.tensor([d,1.0/(d+1e-6),1.0/((d+1e-6)**2),same_chain,seq_sep],dtype = torch.float32)
        idx.extend([[i,j],[j,i]])
        feats.extend([f,f])

    if not idx:
        ei = torch.empty((2,0),dtype=torch.long)
        ea = torch.empty((0,5),dtype=torch.float32)

    else:
        ei = torch.tensor(idx,dtype=torch.long).t().contiguous()
        ea = torch.stack(feats,dim=0)
    return ei,ea

#==============================
# 配体原子 - 残基层级： 接触边（双向） + 距离特征
#==============================
def inter_graph_lig_res(ligand,pos_res,atom_offset,res_offset,dis_threshold=5.0):
    L = np.array(ligand.GetConformer().GetPositions(),dtype=np.float32)
    R = pos_res.cpu().numpy().astype(np.float32)
    if L.size == 0 or R.size == 0:
        edge_index = torch.empty((2,0),dtype=torch.long)
        edge_attr = torch.empty((0,3),dtype=torch.float32)
        return edge_index,edge_attr
    D = distance_matrix(L,R)
    I,J = np.where(D < dis_threshold)
    idx, feats = [],[]
    for i,j in zip(I,J):
        d = D[i,j].astype(np.float32)
        f = torch.tensor([d,1.0/(d+1e-6),1.0/((d+1e-6)**2)],dtype=torch.float32)
        u = int(i + atom_offset)
        v = int(j + res_offset)
        idx.extend([[u,v],[v,u]])
        feats.extend([f,f])
    if not idx:
        edge_index = torch.empty((2,0),dtype=torch.long)
        edge_attr = torch.empty((0,3),dtype=torch.float32)
    else:
        edge_index = torch.tensor(idx,dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(feats,dim=0)
    
    return edge_index,edge_attr




# =========================
# 主构图函数（CHANGED）
# - 配体：原子级 + 键特征
# - 蛋白：残基级 + 残基-残基边
# - 跨边：配体原子 ↔ 残基（距离特征）
# =========================

def mols2graphs(complex_path,label,save_path,dis_threshold=5.0,res_edge_threshold=5.0):
    with open(complex_path,'rb') as f:
        ligand,pocket = pickle.load(f)

    #配体 （原子级）
    x_l,ei_l,ea_l = mol2graph(ligand)
    #print(f'ligand_size:{ea_l.size()}')
    print(f'ligand-ligand-index:{ei_l}')
    print(ei_l.size())
    pos_l = torch.tensor(ligand.GetConformer().GetPositions(),dtype=torch.float32)
    n_l = x_l.size(0)
    #print(x_l.size())

    #蛋白 （残基粒度）
    pocket_info = pocket_to_residue_nodes(pocket)
    res_keys,res2idx,x_res,pos_res = pocket_info
    n_r = x_res.size(0)
    #print(x_res.size())
    
    
    # 残基-残基边
    ei_rr,ea_rr = residue_graph_edges(res_keys,pos_res,res_edge_threshold=res_edge_threshold)
    #print(f'pocket_size:{ea_rr.size()}')
    print(f'protein-protein-index:{ei_rr}')
    print(ei_rr.size())
    #配体原子-残基 inter
    ei_lr,ea_lr = inter_graph_lig_res(ligand,pos_res,atom_offset=0,res_offset=n_l,dis_threshold=dis_threshold)
    print(f'ligand-protein-index:{ei_lr}')
    print(ei_lr.size())
    #intra 配体-配体、蛋白-蛋白分开保存
    edge_index_intra_l = ei_l
    edge_attr_intra_l = ea_l

    edge_index_intra_p = torch.clone(ei_rr) + n_l
    edge_attr_intra_p = ea_rr

    #inter 配体-残基  独立保存
    edge_index_inter = ei_lr
    edge_attr_inter = ea_lr

    #torch.set_printoptions(threshold=torch.inf)

    #节点/位置/split
    #x = torch.cat([x_l,x_res],dim=0)
    pos = torch.cat([pos_l,pos_res],dim=0)
    split = torch.cat([torch.zeros(n_l),torch.ones(n_r)],dim=0)

    y = torch.tensor([label],dtype=torch.float32)

    data = Data(
        x_l=x_l,x_res=x_res,y=y,pos=pos,split=split,
        edge_index_intra_l = edge_index_intra_l,edge_attr_intra_l = edge_attr_intra_l,
        edge_index_intra_p = edge_index_intra_p,edge_attr_intra_p = edge_attr_intra_p,
        edge_index_inter = edge_index_inter,edge_attr_inter = edge_attr_inter
    )
    torch.save(data,save_path)
    print("n_l =", n_l, "n_r =", n_r, "total =", n_l+n_r)

    print("edge_index_intra_l max:", edge_index_intra_l.max().item() if edge_index_intra_l.numel() > 0 else None)
    print("edge_index_intra_p max:", edge_index_intra_p.max().item() if edge_index_intra_p.numel() > 0 else None)
    print("edge_index_inter max:", edge_index_inter.max().item() if edge_index_inter.numel() > 0 else None)
    return data

# =========================
# DataLoader & Dataset
# =========================
class PLIDataLoader(DataLoader):
    def __init__(self,data,**kwargs):
        super().__init__(data,collate_fn=data.collate_fn,**kwargs)

class GraphDataset(Dataset):
    def __init__(self,data_dir,data_df,dis_threshold=5,num_process=8,create=False,res_edge_threshold=5.0):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.res_edge_threshold = res_edge_threshold
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.num_process = num_process
        self._pre_process()

    def _pre_process(self):
        data_dir = self.data_dir
        data_df = self.data_df
        dis_thresholds = repeat(self.dis_threshold,len(data_df))
        res_edge_thrs = repeat(self.res_edge_threshold,len(data_df))
        
        complex_path_list = []
        complex_id_list = []
        pKa_list = []
        graph_path_list = []
        for _,row in data_df.iterrows():
            cid = row['pdbid']
            pKa = float(row['logKd/Ki'])
            #print(cid)
            complex_dir = os.path.join(data_dir,cid)
            graph_path = os.path.join(complex_dir,f'{cid}_{self.dis_threshold}A.pyg')
            complex_path = os.path.join(complex_dir,f'{cid}_{self.dis_threshold}A.rdkit')

            complex_path_list.append(complex_path)
            complex_id_list.append(cid)
            pKa_list.append(pKa)
            graph_path_list.append(graph_path)

        if self.create:
            print('Generate complex graph...')
            with multiprocessing.Pool(self.num_process) as pool: 
                pool.starmap(mols2graphs,zip(complex_path_list,pKa_list,graph_path_list,dis_thresholds,res_edge_thrs))


        self.graph_paths = graph_path_list
        self.complex_ids = complex_id_list

    def __getitem__(self,idx):
        return torch.load(self.graph_paths[idx])

    def collate_fn(self,batch):
        return Batch.from_data_list(batch)

    def __len__(self):
        return len(self.data_df)



if __name__ == '__main__':
    # complex_path = './data/test/948943_5A.rdkit'
    # label = 9.0
    # save_path = './data/test/test.pyg'
    # data = mols2graphs(complex_path,label,save_path,dis_threshold=5.0)
    
    # print(data)
    # print(data.x[:5])          # 前5个原子的特征向量
    # print(data.edge_index_intra[:, :10])  # 前10条内部边
    # print(data.edge_index_inter[:, :10])  # 前10条交互边
    # print(data.pos[:5])        # 前5个原子的坐标
    # print(data.split)          # 每个节点的分子来源
    # print(data.y)              # 标签
    data_root = './data/test'
    toy_dir = os.path.join(data_root,'pyg_get')
    toy_df = pd.read_csv(os.path.join(toy_dir,'mydataset_test.csv'),dtype={'pdbid':str})
    toy_set = GraphDataset(toy_dir,toy_df,dis_threshold=8,res_edge_threshold=8,create=True)
    train_loader = PLIDataLoader(toy_set, batch_size=128, shuffle=True, num_workers=4)