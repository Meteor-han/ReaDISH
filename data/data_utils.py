from rdkit import Chem
from rdkit.Chem import rdChemReactions
from data.ni import *

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

    
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


# return: a list of tuples, [(SMILES_reaction, Yield/ddg), ...], make all a.b>c>d
def generate_Buchwald_Hartwig_rxns(df, mul=0.01):
    """
    Converts the entries in the excel files from Sandfort et al. to reaction SMILES.
    """
    df = df.copy()
    fwd_template = '[F,Cl,Br,I]-[c;H0;D3;+0:1](:[c,n:2]):[c,n:3].[NH2;D1;+0:4]-[c:5]>>' \
                   '[c,n:2]:[c;H0;D3;+0:1](:[c,n:3])-[NH;D2;+0:4]-[c:5]'
    methylaniline = 'Cc1ccc(N)cc1'
    pd_catalyst = Chem.MolToSmiles(Chem.MolFromSmiles('O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F'))
    methylaniline_mol = Chem.MolFromSmiles(methylaniline)
    rxn = rdChemReactions.ReactionFromSmarts(fwd_template)
    products = []
    for i, row in df.iterrows():
        reacts = (Chem.MolFromSmiles(row['Aryl halide']), methylaniline_mol)
        rxn_products = rxn.RunReactants(reacts)

        rxn_products_smiles = set([Chem.MolToSmiles(mol[0]) for mol in rxn_products])
        assert len(rxn_products_smiles) == 1
        products.append(list(rxn_products_smiles)[0])
    df['product'] = products
    rxns = []
    can_smiles_dict = {}
    for i, row in df.iterrows():
        aryl_halide = canonicalize_with_dict(row['Aryl halide'], can_smiles_dict)
        can_smiles_dict[row['Aryl halide']] = aryl_halide
        ligand = canonicalize_with_dict(row['Ligand'], can_smiles_dict)
        can_smiles_dict[row['Ligand']] = ligand
        base = canonicalize_with_dict(row['Base'], can_smiles_dict)
        can_smiles_dict[row['Base']] = base
        additive = canonicalize_with_dict(row['Additive'], can_smiles_dict)
        can_smiles_dict[row['Additive']] = additive

        reactants = f"{aryl_halide}.{methylaniline}>{pd_catalyst}.{ligand}.{base}.{additive}"
        rxns.append((f"{reactants}>{row['product']}", row['Output'] * mul,))
    return rxns


def generate_Suzuki_Miyaura_rxns(df, mul=0.01):
    rxns = []
    for i, row in df.iterrows():
        rxns.append((row['rxn'],) + (row['y'] * 100 * mul,))  # .replace(">>", ".").split(".")
    return rxns


def generate_ELN_BH_rxns(raw_data, mul=0.01):
    rxns = []
    smi_dict = {}
    for one in raw_data:
        # all exists, no empty (""); raw data not canonicalized
        rxns.append((f'{canonicalize_with_dict(one["reactants"][0]["smiles"], smi_dict)}.{canonicalize_with_dict(one["reactants"][1]["smiles"], smi_dict)}.' + 
                     f'{canonicalize_with_dict(one["reactants"][2]["smiles"], smi_dict)}>{canonicalize_with_dict(one["base"]["smiles"], smi_dict)}.{canonicalize_with_dict(one["solvent"][0], smi_dict)}>' + 
                     f'{canonicalize_with_dict(one["product"]["smiles"], smi_dict)}',) + (one["yield"]["yield"] * mul,))
    return rxns


def generate_NiCOlit_rxns(df, mul=0.01):
    solvents = df["solvent"].tolist()
    ligands = [ligand_mapping(precursor) for precursor in df["effective_ligand"]]
    # precursors = [precursor_mapping(precursor) for precursor in df["catalyst_precursor"]]
    additives = [additives_mapping(precursor) for precursor in df["effective_reagents"]]

    rxns = []
    smi_dict = {}
    for i, row in df.iterrows():
        yield_isolated = process_yield(row["isolated_yield"])
        yield_gc = process_yield(row['analytical_yield'])
        # If both yields are known, we keep the isolated yield
        if yield_gc is not None:
            y = yield_gc
        if yield_isolated is not None:
            y = yield_isolated
        rxn_smarts = f"{canonicalize_with_dict(row['substrate'], smi_dict)}.{canonicalize_with_dict(row['effective_coupling_partner'], smi_dict)}"
        rxn_smarts += f">{canonicalize_with_dict(dict_solvent[solvents[i]], smi_dict)}.{canonicalize_with_dict(ligands[i], smi_dict)}.{canonicalize_with_dict(additives[i], smi_dict)}>".replace("..", ".").replace(".>", ">").replace(">.", ">")
        rxn_smarts += f"{canonicalize_with_dict(row['product'], smi_dict)}"
        rxns.append((rxn_smarts, y*mul))
    return rxns


def generate_N_S_acetal_rxns(df, df_p):
    r1_ = df["Imine"].tolist()
    r2_ = df["Thiol"].tolist()
    c_ = df["Catalyst"].tolist()
    ddg_ = df["Output"].tolist()
    r2p_map = {}
    for i in range(len(df_p)):
        r2p_map[(df_p["Imine"][i], df_p["Thiol"][i])] = df_p["Product"][i]

    rxns = []
    smi_dict = {}
    for i in range(len(df)):
        p_temp = r2p_map[(r1_[i], r2_[i])]
        rea_ = f"{canonicalize_with_dict(r1_[i], smi_dict)}.{canonicalize_with_dict(r2_[i], smi_dict)}>{canonicalize_with_dict(c_[i], smi_dict)}>{canonicalize_with_dict(p_temp, smi_dict)}"
        rxns.append((rea_, ddg_[i]))
    return rxns


def generate_C_Heteroatom_rxns(df, mul=0.01):
    smi_dict = {}
    map2product_old = {"C1=CC=C(PC2=CC=CC=C2)C=C1": "C1=CC=C(P(C2=CC=CC=C2)C2=CC=CN=C2)C=C1",
     "C1=CC=C2NN=CC2=C1": "C1=CN=CC(N2N=CC3=CC=CC=C32)=C1",
     "CC1(C)OB(C2=CN(CC3=CC=CC=C3)N=C2)OC1(C)C": "C1=CC=C(CN2C=C(C3=CC=CN=C3)C=N2)C=C1",
     "CC(N)CCC1=CC=CC=C1": "CC(CCC1=CC=CC=C1)NC1=CC=CN=C1",
     "C#CC1=CC=C(CCCC)C=C1": "CCCCC1=CC=C(C#CC2=CC=CN=C2)C=C1",
     "NC1=CC=C(C2=CC=CC=C2)C=N1": "C1=CC=C(C2=CN=C(NC3=CN=CC=C3)C=C2)C=C1",
     "NC(=O)C1=CC=CC=C1": "O=C(NC1=CC=CN=C1)C1=CC=CC=C1",
     "N=C(N)CC1=CC=CC=C1": "N=C(CC1=CC=CC=C1)NC1=CC=CN=C1",
     "CC(C)(C)OC(N)=O": "CC(C)(C)OC(=O)NC1=CC=CN=C1",
     "NC1=CC=CC=C1": "C1=CC=C(NC2=CC=CN=C2)C=C1",
     "NS(=O)(=O)C1=CC=CC=C1": "O=S(=O)(NC1=CC=CN=C1)C1=CC=CC=C1",
     "CCOC(=O)CC(=O)OCC": "CCOC(=O)C(C(=O)OCC)C1=CC=CN=C1",
     "CCOC(=O)/C=C/C1=CC=CC=C1": "O=C(/C=C/C1=CC=CC=C1)OCCC1=CC=CN=C1",
     "OC1=CC=CC=C1": "C1=CC=C(OC2=CC=CN=C2)C=C1",
     "OCCCC1=CC=CC=C1": "C1=CC=C(CCCOC2=CC=CN=C2)C=C1",
     "SC1=CC=CC=C1": "C1=CC=C(SC2=CC=CN=C2)C=C1"}
    map2product = {}
    for k, v in map2product_old.items():
        map2product[canonicalize_with_dict(k, smi_dict)] = canonicalize_with_dict(v, smi_dict)
    
    r1_ = df["Electrophile"].tolist()
    r2_ = df["Nucleophile"].tolist()
    c_ = df["Catalyst"].tolist()
    b_ = df["Base"].tolist()
    rc_ = df["Output"].tolist()

    rxns = []
    for i in range(len(df)):
        if ".O" in r2_[i]:
            r2_[i] = r2_[i][:-2]
        p_temp = map2product[canonicalize_with_dict(r2_[i], smi_dict)]
        rea_ = f"{canonicalize_with_dict(r1_[i], smi_dict)}.{canonicalize_with_dict(r2_[i], smi_dict)}>{canonicalize_with_dict(c_[i], smi_dict)}.{canonicalize_with_dict(b_[i], smi_dict)}>{canonicalize_with_dict(p_temp, smi_dict)}"
        rxns.append((rea_, rc_[i]*mul))
    return rxns


def generate_C_H_olefination_rxns(df):
    r1_ = df["Biaryl"].tolist()
    r2_ = df["Olefin"].tolist()
    c_ = df["Catalyst"].tolist()
    tdg_ = df["TDG"].tolist()
    a_ = df["Additive"].tolist()
    s_ = df["Solvent"].tolist()
    p_ = df["Product"].tolist()
    ddg_ = df["ddG(kcal/mol)"].tolist()
    rxns = []
    smi_dict = {}
    # only additive may missing
    for i in range(len(df)):
        rea_ = f"{canonicalize_with_dict(r1_[i], smi_dict)}.{canonicalize_with_dict(r2_[i], smi_dict)}>{canonicalize_with_dict(c_[i], smi_dict)}.{canonicalize_with_dict(tdg_[i], smi_dict)}"
        a = canonicalize_with_dict(a_[i], smi_dict)
        rea_ += f".{a}" if a else ""
        rea_ += f".{canonicalize_with_dict(s_[i], smi_dict)}>{canonicalize_with_dict(p_[i], smi_dict)}"
        rxns.append((rea_, ddg_[i]))
    return rxns


def generate_az_bh_rxns_wo(raw_data, mul=0.01):
    rxns = []
    for one in raw_data:
        # all exists, no empty (""); raw data not canonicalized
        rxns.append((f'{one["reactants"][0]["smiles"]}.{one["reactants"][1]["smiles"]}.' + 
                     f'{one["reactants"][2]["smiles"]}.{one["base"]["smiles"]}.{one["solvent"][0]}>>' + 
                     f'{one["product"]["smiles"]}',) + (one["yield"]["yield"] * mul,))
    return rxns


# return: a list of tuples, [(SMILES_1, ..., SMILES_n, Yield), ...]
def generate_contest_rxns(df, mul=1):
    """
    Converts the entries in the excel files from Sandfort et al. to reaction SMILES.
    """
    df = df.copy()
    rxns = []
    for i, row in df.iterrows():
        additive = row['Additive'].split('.')
        trans_add=[]
        for mol in additive:
            trans_add.append(trans(mol))
        new_add_smiles='.'.join(trans_add)
        # reactants = f"{trans(row['Reactant1'])}.{trans(row['Reactant2'])}.{new_add_smiles}"
        reactants = f"{trans(row['Reactant1'])}.{trans(row['Reactant2'])}"
        rxns.append((f"{reactants}>>{row['Product']}", row['Yield'] * mul,))
    return rxns


if __name__ == '__main__':
    import pandas as pd
    import os

    tag = "heteroatom"
    df = pd.read_excel(os.path.join("/amax/data/reaction/dataset_ds/C-Heteroatom", "Cernak_and_Dreher_input_data.xlsx"))
    rxns = generate_C_Heteroatom_rxns(df)

    tag = "sm"
    df = pd.read_csv(os.path.join("/amax/data/reaction/data", 'SM/SM_Test_1.tsv'), sep='\t')
    rxns = generate_Suzuki_Miyaura_rxns(df, 0.01)

    tag = "ns_acetal"
    p_prefix = "/amax/data/reaction/dataset_ds/N,S-acetal Denmark"
    df = pd.read_excel(os.path.join(p_prefix, "Denmark_input_data.xlsx")).fillna("")
    df_p = pd.read_csv(os.path.join(p_prefix, "Denmark_data_product.csv"))
    rxns = generate_N_S_acetal_rxns(df, df_p)
    print()
