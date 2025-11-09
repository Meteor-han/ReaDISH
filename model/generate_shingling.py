from tqdm import tqdm
import concurrent.futures
import sys
import os
import pickle
from rdkit import Chem
from typing import Iterable, List, Tuple, Set, Dict, Union
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
import numpy as np
from collections import defaultdict
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
                if smiles == "[C](#[CH])[c]([cH][cH2])[cH][cH2]":
                    print()    

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
                right_shingles.add(s)

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
                left_shingles.add(s)

        s = right_shingles.symmetric_difference(left_shingles)

        remain_atom_indices = {}
        for type_key in atom_indices.keys():
            remain_atom_indices[type_key] = []
            for i in range(len(atom_indices[type_key])):
                remain_atom_indices[type_key].append(defaultdict(list))
                try:
                    for k, v in atom_indices[type_key][i].items():
                        if k in s:
                            remain_atom_indices[type_key][i][k] = v
                except:
                    print()
            
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
        root_central_atom: bool = True,
        include_hydrogens: bool = False,
        show_progress_bar: bool = False,
        omit: bool = True,
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


def process_one(one, omit):
    mappings = ShinglingEncoder.encode([one[0]], show_progress_bar=False, atom_index_mapping=True, root_central_atom=False, omit=omit)[0]
    return one[0], mappings


def merge_pretraining_shingles():
    index = 0
    omit = False
    k2v = {}
    for (start_id, end_id) in [(0, 1000000), (1000000, 2000000), (2000000, 3000000), (3000000, 4000000)]:
        with open(f"/amax/data/reaction/shinglings/rxn2shingling_{omit}_{start_id}_{end_id}.pkl", "rb") as f:
            sh_dict = pickle.load(f)
        for k, v in tqdm(sh_dict.items()):
            p_ = f"/amax/data/reaction/shinglings/data/{index}.pkl"
            k2v[k] = p_
            with open(p_, "wb") as f:
                pickle.dump(v, f)
            index += 1
    with open(f"/amax/data/reaction/shinglings/rxn2shingling_{omit}_s2p.pkl", "wb") as f:
        pickle.dump(k2v, f)


if __name__ == '__main__':
    with open("/amax/data/yield_data/pretraining_data/reactions_with_multiple.pkl", "rb") as f:
        data_type = pickle.load(f)
    print(f"reactions nums: {len(data_type)}")

    if len(sys.argv[1:]) == 2:
        start_id, end_id = int(sys.argv[1]), int(sys.argv[2])
    else:
        start_id, end_id = 0, 4000000

    data_type = sorted(data_type, key=lambda x: len(x[0]))[start_id:end_id]
    for omit in [True, False]:
        rxn2shingling = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            future_to_data = {executor.submit(process_one, one, omit): one for one in data_type}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_data), total=len(data_type)):
                key, value = future.result()
                rxn2shingling[key] = value

        with open(f"/amax/data/reaction/shinglings/rxn2shingling_{omit}_{start_id}_{end_id}.pkl", "wb") as f:
            pickle.dump(rxn2shingling, f)

    print()
