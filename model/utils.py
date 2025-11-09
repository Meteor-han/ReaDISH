"""Thank for the code from https://github.com/reymond-group/drfp"""
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import argparse
from rdkit import Chem
from typing import Iterable, List, Tuple, Set, Dict, Union
from collections import defaultdict
from tqdm import tqdm
import logging
import random
import os
from torch.nn.utils.rnn import pad_sequence
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

class NoReactionError(Exception):
    """Raised when the encoder attempts to encode a non-reaction SMILES.

    Attributes:
        message: a message containing the non-reaction SMILES
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class ShinglingEncoder:
    """A class for encoding SMILES as drfp fingerprints."""

    @staticmethod
    def shingling_from_mol(
        in_mol: Mol,
        radius: int = 3,
        rings: bool = True,
        min_radius: int = 0,
        get_atom_indices: bool = False,
        root_central_atom: bool = True,
        include_hydrogens: bool = False,
    ) -> Union[List[str], Tuple[List[str], Dict[str, List[Set[int]]]]]:
        """Creates a molecular shingling from a RDKit molecule (rdkit.Chem.rdchem.Mol).

        Arguments:
            in_mol: A RDKit molecule instance
            radius: The drfp radius (a radius of 3 corresponds to drfp6)
            rings: Whether or not to include rings in the shingling
            min_radius: The minimum radius that is used to extract n-grams

        Returns:
            The molecular shingling.
        """

        if include_hydrogens:
            in_mol = AllChem.AddHs(in_mol)

        shingling = []
        atom_indices = defaultdict(list)

        if rings:
            for ring in AllChem.GetSymmSSSR(in_mol):
                bonds = set()
                ring = list(ring)
                indices = set()
                for i in ring:
                    for j in ring:
                        if i != j:
                            indices.add(i)
                            indices.add(j)
                            bond = in_mol.GetBondBetweenAtoms(i, j)
                            if bond is not None:
                                bonds.add(bond.GetIdx())

                ngram = AllChem.MolToSmiles(
                    AllChem.PathToSubmol(in_mol, list(bonds)),
                    canonical=True,
                    allHsExplicit=True,
                ).encode("utf-8")

                shingling.append(ngram)

                if get_atom_indices:
                    atom_indices[ngram].append(indices)

        if min_radius == 0:
            for i, atom in enumerate(in_mol.GetAtoms()):
                ngram = atom.GetSmarts().encode("utf-8")
                shingling.append(ngram)

                if get_atom_indices:
                    atom_indices[ngram].append(set([atom.GetIdx()]))

        for index, _ in enumerate(in_mol.GetAtoms()):
            for i in range(1, radius + 1):
                p = AllChem.FindAtomEnvironmentOfRadiusN(
                    in_mol, i, index, useHs=include_hydrogens
                )
                amap = {}
                submol = AllChem.PathToSubmol(in_mol, p, atomMap=amap)

                if index not in amap:
                    continue

                smiles = ""

                if root_central_atom:
                    smiles = AllChem.MolToSmiles(
                        submol,
                        rootedAtAtom=amap[index],
                        canonical=True,
                        allHsExplicit=True,
                    )
                else:
                    smiles = AllChem.MolToSmiles(
                        submol,
                        canonical=True,
                        allHsExplicit=True,
                    )

                if smiles != "":
                    shingling.append(smiles.encode("utf-8"))
                    if get_atom_indices:
                        atom_indices[smiles.encode("utf-8")].append(set(amap.keys()))

        if not root_central_atom:
            for key in atom_indices:
                atom_indices[key] = list(set([frozenset(s) for s in atom_indices[key]]))

        # Set ensures that the same shingle is not hashed multiple times
        # (which would not change the hash, since there would be no new minima)
        if get_atom_indices:
            return list(set(shingling)), atom_indices
        else:
            return list(set(shingling))

    @staticmethod
    def internal_encode(
        in_smiles: str,
        radius: int = 3,
        min_radius: int = 0,
        rings: bool = True,
        get_atom_indices: bool = False,
        root_central_atom: bool = True,
        include_hydrogens: bool = False,
        omit: bool = True,
        symmetric_id: int = 0,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, Dict[str, List[Dict[str, List[Set[int]]]]]],
    ]:
        """Creates an drfp array from a reaction SMILES string.

        Arguments:
            in_smiles: A valid reaction SMILES string
            radius: The drfp radius (a radius of 3 corresponds to drfp6)
            min_radius: The minimum radius that is used to extract n-grams
            rings: Whether or not to include rings in the shingling
            omit: Whether to omit single atoms
        Returns:
            A tuple with two arrays, the first containing the drfp hash values, the second the substructure SMILES
        """

        atom_indices = {}
        atom_indices["reactants"] = []
        atom_indices["products"] = []

        """in rare cases, SMILES may contain '>', """
        if ">>" in in_smiles:
            sides = in_smiles.split(">>")
            sides = [sides[0], "", sides[1]]
        else:
            sides = in_smiles.split(">")
        if len(sides) < 3:
            raise NoReactionError(
                f"The following is not a valid reaction SMILES: '{in_smiles}'"
            )

        if len(sides[1]) > 0:
            sides[0] += "." + sides[1]

        left = sides[0].split(".")
        right = sides[2].split(".")

        left_shingles = set()
        right_shingles = set()

        for l in left:
            mol = AllChem.MolFromSmiles(l)

            if not mol:
                atom_indices["reactants"].append(None)
                continue

            if omit:
                if mol.GetNumAtoms() == 1:
                    continue

            if get_atom_indices:
                sh, ai = ShinglingEncoder.shingling_from_mol(
                    mol,
                    radius=radius,
                    rings=rings,
                    min_radius=min_radius,
                    get_atom_indices=True,
                    root_central_atom=root_central_atom,
                    include_hydrogens=include_hydrogens,
                )
                atom_indices["reactants"].append(ai)
            else:
                sh = ShinglingEncoder.shingling_from_mol(
                    mol,
                    radius=radius,
                    rings=rings,
                    min_radius=min_radius,
                    root_central_atom=root_central_atom,
                    include_hydrogens=include_hydrogens,
                )

            for s in sh:
                left_shingles.add(s)

        for r in right:
            mol = AllChem.MolFromSmiles(r)

            if not mol:
                atom_indices["products"].append(None)
                continue

            if omit:
                if mol.GetNumAtoms() == 1:
                    continue

            if get_atom_indices:
                sh, ai = ShinglingEncoder.shingling_from_mol(
                    mol,
                    radius=radius,
                    rings=rings,
                    min_radius=min_radius,
                    get_atom_indices=True,
                    root_central_atom=root_central_atom,
                    include_hydrogens=include_hydrogens,
                )
                atom_indices["products"].append(ai)
            else:
                sh = ShinglingEncoder.shingling_from_mol(
                    mol,
                    radius=radius,
                    rings=rings,
                    min_radius=min_radius,
                    root_central_atom=root_central_atom,
                    include_hydrogens=include_hydrogens,
                )

            for s in sh:
                right_shingles.add(s)

        if symmetric_id == 0:  # symmetric difference
            s = right_shingles.symmetric_difference(left_shingles)
        elif symmetric_id == 1:  # left only
            s = left_shingles
        elif symmetric_id == 2:  # union
            s = left_shingles.union(right_shingles)

        remain_atom_indices = {}
        flag = False
        for type_key in atom_indices.keys():
            remain_atom_indices[type_key] = []
            for i in range(len(atom_indices[type_key])):
                remain_atom_indices[type_key].append(defaultdict(list))
                try:
                    temp_num = 0
                    for k, v in atom_indices[type_key][i].items():
                        if temp_num >= 100: # fix the maximum number for one molecule
                            break
                        if k in s:
                            flag = True
                            if len(v) > 10: # fix the maximum number for one shingle
                                v = random.sample(v, 10)
                            remain_atom_indices[type_key][i][k] = v
                        temp_num += 1
                except:
                    print("wrong here")
                    exit()
        if not flag:
            return list(s), atom_indices, {}
            
        if get_atom_indices:
            return list(s), atom_indices, remain_atom_indices
        else:
            return list(s)

    @staticmethod
    def encode(
        X: Union[Iterable, str],
        min_radius: int = 0,
        radius: int = 3,
        rings: bool = True,
        mapping: bool = False,
        atom_index_mapping: bool = True,
        root_central_atom: bool = False,
        include_hydrogens: bool = False,
        show_progress_bar: bool = False,
        omit: bool = True,
        symmetric_id: bool = False,
    ) -> Union[
        List[np.ndarray],
        Tuple[List[np.ndarray], Dict[int, Set[str]]],
        Tuple[List[np.ndarray], Dict[int, Set[str]]],
        List[Dict[str, List[Dict[str, List[Set[int]]]]]],
    ]:
        """Encodes a list of reaction SMILES using the drfp fingerprint.

        Args:
            X: An iterable (e.g. List) of reaction SMILES or a single reaction SMILES to be encoded
            n_folded_length: The folded length of the fingerprint (the parameter for the modulo hashing)
            min_radius: The minimum radius of a substructure (0 includes single atoms)
            radius: The maximum radius of a substructure
            rings: Whether to include full rings as substructures
            mapping: Return a feature to substructure mapping in addition to the fingerprints
            atom_index_mapping: Return the atom indices of mapped substructures for each reaction
            root_central_atom: Whether to root the central atom of substructures when generating SMILES
            show_progress_bar: Whether to show a progress bar when encoding reactions

        Returns:
            A list of drfp fingerprints or, if mapping is enabled, a tuple containing a list of drfp fingerprints and a mapping dict.
        """
        assert atom_index_mapping
        if isinstance(X, str):
            X = [X]

        show_progress_bar = not show_progress_bar

        # If mapping is required for atom_index_mapping
        if atom_index_mapping:
            mapping = True

        result = []
        result_map = defaultdict(set)
        atom_index_maps = []

        for _, x in tqdm(enumerate(X), total=len(X), disable=show_progress_bar):
            if atom_index_mapping:
                smiles_diff, atom_index_map, reamin_atom_index_map = ShinglingEncoder.internal_encode(
                    x,
                    min_radius=min_radius,
                    radius=radius,
                    rings=rings,
                    get_atom_indices=True,
                    root_central_atom=root_central_atom,
                    include_hydrogens=include_hydrogens,
                    omit=omit,
                    symmetric_id=symmetric_id,
                )
            else:
                smiles_diff = ShinglingEncoder.internal_encode(
                    x,
                    min_radius=min_radius,
                    radius=radius,
                    rings=rings,
                    root_central_atom=root_central_atom,
                    include_hydrogens=include_hydrogens,
                    omit=omit,
                )
            result.append(reamin_atom_index_map)

        return result


def trans(smiles):
    # isomericSmiles, kekuleSmiles (F), canonical
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def canonicalize_with_dict(smi, can_smi_dict=None):
    if can_smi_dict is None:
        can_smi_dict = {}
    if smi not in can_smi_dict.keys():
        try:
            can_smi_dict[smi] = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
        except:
            can_smi_dict[smi] = ""
            return ""
    return can_smi_dict[smi]

class LoggerPrint:
    def __init__(self):
        pass

    def info(self, input):
        print(input)

def create_file_logger(file_name: str = 'log.txt', log_format: str = '%(message)s', log_level=logging.INFO, save=True):
    if save:
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)

        handler = logging.FileHandler(file_name, "w")
        handler.setLevel(log_level)
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)

        console = logging.StreamHandler()
        console.setLevel(log_level)

        logger.addHandler(handler)
        logger.addHandler(console)
    else:
        logger = LoggerPrint()

    return logger


class ReactionDataset(Dataset):
    def __init__(self, input_data, radius=3, omit=True, smi2id=None, mol_max_len=10, conf_max_size=1, 
                 data_prefix="/amax/data/data_pretraining_confs/id2confs/", load_shingling=False, symmetric_id=0):
        super(ReactionDataset, self).__init__()
        self.radius = radius
        self.symmetric_id = symmetric_id
        self.omit = omit
        self.raw_data = input_data
        self.permutation = None
        self.mol_max_len = mol_max_len
        self.conf_max_size = conf_max_size
        self.smi2id = smi2id
        self.can_dict = {}
        self.data_prefix = data_prefix
        self.data_prefix_pre = "/amax/data/data_pretraining_confs/id2confs/"
        p_ = "/amax/data/data_pretraining_confs/smi2id.pkl"
        if os.path.exists(p_):
            with open(p_, "rb") as f:
                self.smi2id_pre = pickle.load(f)
        else:
            self.smi2id_pre = {}
        if load_shingling:
            with open(f"/amax/data/reaction/shinglings/rxn2shingling_{omit}_s2p.pkl", "rb") as f:
                self.sh_dict = pickle.load(f)
        else:
            self.sh_dict = {}
    
    def shuffle(self):
        ## shuffle the dataset using a permutation matrix
        self.permutation = torch.randperm(len(self)).numpy()
        return self

    def __len__(self):
        return len(self.raw_data)

    def get_3d(self, index):
        confs = {"atoms": [], "coordinates": [], "conf_nums": []}
        # add cls
        total_atoms = 0
        # a reaction may contain too many mols, limit it
        num_ = 0
        precursors, products = self.raw_data[index][0].split(">>")
        temp_dict = {0: precursors, 1: products}
        """can not load 3d, omit that mol"""
        temp_dict_new = {0: "", 1: ""}
        # for s in self.raw_data[index][0].replace(">>", ".").split("."):
        for j in [0, 1]:
            for s in temp_dict[j].split("."):
                m = Chem.MolFromSmiles(s)
                # omit single atom or not
                if self.omit:
                    if m.GetNumAtoms() == 1:
                        continue
                num_ += 1
                if num_ > self.mol_max_len and j == 0:
                    break
                can_s = canonicalize_with_dict(s, self.can_dict)
                if can_s in self.smi2id_pre or s in self.smi2id_pre:
                    id_ = self.smi2id_pre.get(can_s, self.smi2id_pre.get(s, ""))
                    with open(os.path.join(self.data_prefix_pre, f"{id_}.pkl"), "rb") as f:
                        mol, clusts = pickle.load(f)
                elif can_s in self.smi2id or s in self.smi2id:
                    id_ = self.smi2id.get(can_s, self.smi2id.get(s, ""))
                    with open(os.path.join(self.data_prefix, f"{id_}.pkl"), "rb") as f:
                        mol, clusts = pickle.load(f)
                else:
                    # omit smiles without 3d or replace with 0.
                    # print("Failed to generate conformer, replace with zeros.")
                    # coordinates = np.zeros((len(atoms),3))
                    continue
                temp_dict_new[j] += (s + ".")

                atom_temp = []
                for atom in mol.GetAtoms():
                    atom_temp.append(atom.GetSymbol())
                num_samples = min(self.conf_max_size, len(clusts))
                clusts = random.sample(clusts, num_samples)
                for one in clusts:
                    conf_id = random.sample(one, 1)[0]
                    confs["atoms"].append(atom_temp)
                    confs["coordinates"].append(mol.GetConformer(id=conf_id).GetPositions())
                confs["conf_nums"].append(len(clusts))
                total_atoms += (len(atom_temp) + 1)
        # one_reaction_smiles = self.raw_data[index][0]
        one_reaction_smiles = temp_dict_new[0][:-1] + ">>" + temp_dict_new[1][:-1]
        
        if not self.sh_dict:
            mappings = ShinglingEncoder.encode([one_reaction_smiles], radius=self.radius, show_progress_bar=False, 
                                            atom_index_mapping=True, root_central_atom=False, omit=self.omit, symmetric_id=self.symmetric_id)[0]
        else:
            with open(self.sh_dict[self.raw_data[index][0]], "rb") as f:
                mappings = pickle.load(f)
        return confs, total_atoms, mappings

    def get_1d(self, index):
        return self.raw_data[index][0]

    def __getitem__(self, index):
        ## consider the permutation
        if self.permutation is not None:
            index = self.permutation[index]
        confs, total_atoms, mappings = self.get_3d(index)
        if "reactants" in mappings:
            mappings["reactants"] = mappings["reactants"][:self.mol_max_len]
        return confs, total_atoms, mappings, self.get_1d(index), self.raw_data[index][1], self.raw_data[index][2]


def smiles_to_fp(smiles, radius=2, n_bits=1024):
    arr = np.zeros(n_bits, dtype=np.uint8)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def tanimoto_similarity_matrix(fingerprints):
    """
    distance = 1 - Tanimoto.
    """
    dot_product = fingerprints @ fingerprints.T
    # norm
    popcounts = np.sum(fingerprints, axis=1)
    # 1/x
    denominator = popcounts[:, None] + popcounts[None, :] - dot_product
    # if x=0
    np.seterr(divide='ignore', invalid='ignore')
    similarity = np.divide(dot_product, denominator)
    similarity = np.nan_to_num(similarity, nan=0.0)
    return similarity

def pad_similarity_matrix(similarity_matrix, max_size):
    # Create a new matrix with padding
    padded_matrix = np.zeros((max_size, max_size))
    n = similarity_matrix.shape[0]
    padded_matrix[:n, :n] = similarity_matrix
    return padded_matrix

def compute_all_similarities(reactions_smiles):
    batch_size = len(reactions_smiles)
    max_reaction_size = max(len(reaction) for reaction in reactions_smiles)
    
    result = []
    for reaction_smiles in reactions_smiles:
        fingerprints = np.array([smiles_to_fp(smiles) for smiles in reaction_smiles])
        similarity_matrix = tanimoto_similarity_matrix(fingerprints)
        padded_matrix = pad_similarity_matrix(similarity_matrix, max_reaction_size)
        result.append(padded_matrix)
    
    return np.array(result)


def compute_distance_matrix(tensor):
    # broadcast
    A = tensor.unsqueeze(2)  # (B, N, 1, 3)
    B = tensor.unsqueeze(1)  # (B, 1, N, 3)
    
    diff = A - B            # (B, N, N, 3)
    squared_dist = torch.sum(diff ** 2, dim=-1)  # (B, N, N)
    dist_matrix = torch.sqrt(squared_dist)
    
    return dist_matrix
    
    
class MyCollater:
    def __init__(self):
        pass
        # self.pad_idx = pad_idx
        # self.tokenizer = tokenizer
        # self.text_max_len = text_max_len

    def __call__(self, batch):
        conf_batch, total_atoms_batch, mappings, text_batch, types, labels_y = zip(*batch)

        """pad smiles, compute similarity"""
        shin_smiles = []
        """for batch index"""
        max_mol_num = max([len(one["conf_nums"]) for one in conf_batch])
        max_shi_num = -1  # per molecule
        max_atom_num = -1
        for i in range(len(mappings)):
            temp_smiles = [""]  # for cls token
            # USPTO input may contain "incorrect", just use conf_batch
            # max_mol_num = max(len(mappings[i]["reactants"]) + len(mappings[i]["products"]), max_mol_num)
            for k, v in mappings[i].items():
                for s in v:
                    temp_num = 0
                    for part, id_ in s.items():
                        for one in id_:
                            temp_smiles.append(part)
                            max_atom_num = max(max_atom_num, len(one))
                            temp_num += 1
                    max_shi_num = max(max_shi_num, temp_num)
            shin_smiles.append(temp_smiles)

        shin_sim = torch.as_tensor(compute_all_similarities(shin_smiles))

        if max_atom_num == -1:
            print(mappings)
        id_tensor_batch = -torch.ones((len(mappings), max_mol_num, max_shi_num, max_atom_num), dtype=torch.int)
        """compute center distance"""
        shin_center = []
        """compute edge type"""
        shin_edge_type = []
        id_tensor = []
        mol_num = []
        part_num_ = []
        for i in range(len(mappings)):
            shin_center_one = [torch.zeros(3)]  # for cls token
            shin_edge_type_one = [-1]  # for cls token
            part_num = []
            j = 0
            for _, v in mappings[i].items():  # "reactants", "products"
                for s in v:  # for each molecule
                    id_num = 0
                    k = 0
                    for part, id_ in s.items():  # for each shingling in a molecule
                        for one in id_:  # for each ids of a shingling
                            """geometric center"""
                            try:
                                shin_center_one.append(torch.as_tensor(np.sum(conf_batch[i]["coordinates"][j][list(one)], axis=0)))
                            except:
                                print()
                            """edge, use mol index j+1; 0 is for padding"""
                            shin_edge_type_one.append(j+1)
                            try:
                                id_tensor_batch[i][j][k][:len(one)] = torch.as_tensor(list(one))
                            except:
                                print()
                            id_tensor.append(torch.as_tensor(list(one)))
                            id_num += 1
                            k += 1
                    part_num.append(id_num)
                    j += 1
            mol_num.append(sum(part_num))
            part_num_.append(torch.as_tensor(part_num))
            shin_center.append(torch.stack(shin_center_one))
            shin_edge_type.append(torch.as_tensor(shin_edge_type_one))
        # id_tensor_batch = pad_sequence(id_tensor, batch_first=True, padding_value=-1.)

        # coordinates already normalized in unimol
        shin_center_batch = pad_sequence(shin_center, batch_first=True, padding_value=0.)
        shin_dist = compute_distance_matrix(shin_center_batch)

        shin_edge_type_batch = pad_sequence(shin_edge_type, batch_first=True, padding_value=0.)
        shin_edge = (shin_edge_type_batch.unsqueeze(2) == shin_edge_type_batch.unsqueeze(1)).int()
        
        part_num_batch = pad_sequence(part_num_, batch_first=True, padding_value=0.)
        mol_num =  torch.as_tensor(mol_num)

        conf_input = {"atoms": [], "coordinates": []}
        num_mols = []
        num_confs = []
        for one in conf_batch:
            conf_input["atoms"].extend(one["atoms"])
            conf_input["coordinates"].extend(one["coordinates"])
            num_mols.append(len(one["conf_nums"]))
            num_confs.extend(one["conf_nums"])
        
        labels_y = torch.as_tensor(labels_y)
        labels_y = labels_y.reshape(labels_y.shape[0], 1)
        types = torch.as_tensor(types, dtype=torch.long)
        # types = types.reshape(1, types.shape[0])
        total_atoms_batch = torch.as_tensor(total_atoms_batch, dtype=torch.long)
        return (conf_input, num_mols, num_confs, total_atoms_batch, id_tensor_batch, mol_num, part_num_batch, shin_dist, shin_sim, shin_edge), types, labels_y
        

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shingling_max_len', type=int, default=280)
    parser.add_argument('--seed', type=int, default=511, help='random seed')

    parser.add_argument('--radius', type=int, default=3, help='radius')
    parser.add_argument('--symmetric_id', type=int, default=0, help='symmetric_id')

    # pretraining or not
    parser.add_argument('--pretraining', action='store_true')
    parser.add_argument('--init_checkpoint', type=str, default="")

    # for ds dataset
    parser.add_argument('--ds_name', type=str, default="BH", help="dataset name")
    parser.add_argument("--repeat_times", default=1, type=int, help="repeat times")
    parser.add_argument("--split_ratio", type=float, nargs='+', default=[0.7])
    parser.add_argument('--pred_type', type=str, default="", choices=["classification", "regression"], help="prediction type")
    parser.add_argument('--norm', action='store_true', help="norm the regression label")
    parser.add_argument("--ft_idx", default=1, type=int, help="USPTO condition index")

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--strategy_name', type=str, default='auto')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='[0]')
    parser.add_argument('--precision', type=str, default='32')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=20)
    parser.add_argument('--save_every_n_epochs', type=int, default=1)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1) 

    # optimization
    parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
    parser.add_argument('--init_lr', type=float, default=1e-5, help='optimizer init learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='optimizer min learning rate')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
    parser.add_argument('--warmup_steps', type=int, default=500, help='optimizer warmup steps')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
    parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler')
    parser.add_argument('--dropout', type=float, default=0.1, help='cls head dropout')
    parser.add_argument('--output_dim', type=int, default=1, help='cls head output dim')
    # data loader
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=4)

    args = parser.parse_args()
    return args

def get_range(ds_name):
    name2val = {"BH": (0., 1.), "SM": (0., 1.), 
                "halide_I": (0., 1.), "halide_Cl": (0., 1.), "halide_Br": (0., 1.), "pyridyl": (0., 1.), "nonpyridyl": (0., 1.), 
                "SM_test1": (0., 1.), "SM_test2": (0., 1.), "SM_test3": (0., 1.), "SM_test4": (0., 1.), 
                "ELN": (0., 1.), 
                "NiColit": (0., 1.), 'OCH3': (0., 1.), 'OPh': (0., 1.), 'OPiv': (0., 1.), 'OCOC': (0., 1.), 'OC(=O)O': (0., 1.), 'OC(=O)N': (0., 1.), 'OAc': (0., 1.), 'OSi(C)(C)C': (0., 1.), 'Otriazine': (0., 1.),
                "NS_acetal": (-0.419377753, 3.134624572), "Heteroatom": (0., 0.4),
                "NS_test_sub": (-0.419377753, 3.134624572), "NS_test_cat": (-0.419377753, 3.134624572), "NS_test_sub-cat": (-0.419377753, 3.134624572), 
                "Heteroatom_test": (0., 0.4),
                "Condition": (0., 0.)}
    return name2val[ds_name]
