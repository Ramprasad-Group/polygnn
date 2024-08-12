from numpy import random
import numpy as np
from rdkit import Chem
from collections import OrderedDict
from torch_geometric.data import Data
import torch
from dataclasses import dataclass
import math

from polygnn.trimer import TrimerLinear, TrimerLadder

try:
    from . import constants
except ModuleNotFoundError:
    import constants

import random
import polygnn_kit.polygnn_kit as pk

# set seeds
random.seed(2)
torch.manual_seed(2)
np.random.seed(2)


def atom_helper(molecule, ind, atom_config):
    atom = molecule.GetAtomWithIdx(ind)
    atom_feature = atom_fp(atom, atom_config)
    ##### atom feature list #####
    # symbol, one-hot
    # number of neighbors, one-hot
    # implicit valence, one-hot
    # formal charge
    # num. radical electrons
    # hybridization, one-hot
    # is aromatic?
    # num. hydrogen
    # chirality
    #############################
    return atom_feature


def construct_polymer_graph_monocycle(
    molecule,
    bond_config,
    atom_config,
):

    n_atoms = molecule.GetNumAtoms()

    node_features = [atom_helper(molecule, i, atom_config) for i in range(0, n_atoms)]
    edge_features = []
    edge_indices = []

    for bond in molecule.GetBonds():
        edge_indices.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_indices.append(
            [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
        )  # add "opposite" edge for undirected graph
        # bond_feature = bond_fp(bond).reshape((6,))  # ADDED edge feat
        bond_feature = bond_fp(bond, bond_config)
        ##### bond feature list #####
        # type of bond (1,2,3,1.5,etc.), one-hot
        # is conjugated?
        # is in ring?
        # chirality
        #############################
        edge_features.extend(
            [bond_feature, bond_feature]
        )  # both bonds have the same features

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edge_indices, dtype=torch.long)
        .t()
        .contiguous(),  # as shown in https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
        edge_weight=torch.tensor(edge_features, dtype=torch.float),
    )
    return data


def construct_multigraph_polymer_zeros(smile, bond_config, atom_config, stoch):
    """
     Construct a stochastic graph representation of a polymer that can not be cached. Reading comments in the function construct_multigraph_polymer will be helpful to understand this function.
    Keyword arguments:
    smile -- the smiles string of the polymer for which we must constuct the graph
    """
    g = OrderedDict({})
    h = OrderedDict({})
    molecule = multiply_polymer(
        smile, stoch
    )  # multiply the repeat unit a random number of times
    n_atoms = molecule.GetNumAtoms()
    if stoch:
        shuffled_inds = list(range(n_atoms))
        random.shuffle(shuffled_inds)  # randomly shuffle the atom indices in place
        molecule = Chem.rdmolops.RenumberAtoms(molecule, shuffled_inds)
    node_features = []
    edge_features = []
    edge_indices = []
    for i in range(0, n_atoms):
        atom_i = molecule.GetAtomWithIdx(i)
        atom_feature = atom_fp(atom_i, atom_config).reshape((atom_config.n_features,))
        node_features.append(atom_feature)
        ##### atom feature list #####
        # symbol, one-hot
        # number of neighbors, one-hot
        # implicit valence, one-hot
        # formal charge
        # num. radical electrons
        # hybridization, one-hot
        # is aromatic?
        # num. hydrogen
        # chirality
        #############################
    for bond in molecule.GetBonds():
        edge_indices.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_indices.append(
            [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
        )  # add "opposite" edge for undirected graph
        bond_feature = bond_fp(bond, bond_config).reshape(
            (bond_config.n_features,)
        )  # ADDED edge feat
        ##### bond feature list #####
        # type of bond (1,2,3,1.5,etc.), one-hot
        # is conjugated?
        # is in ring?
        # chirality
        #############################
        edge_features.extend(
            [bond_feature, bond_feature]
        )  # both bonds have the same features
    data = Data(
        x=torch.zeros(np.array(node_features).shape, dtype=torch.float),
        edge_index=torch.tensor(edge_indices, dtype=torch.long)
        .t()
        .contiguous(),  # as shown in https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
        edge_weight=torch.zeros(np.array(edge_features).shape, dtype=torch.float),
    )
    return data


def multiply_polymer(smile, stoch=True, force_multiply=None):
    """
    Multiply (by some random number between 1 and 4) and cyclize the polymer smiles
    Keyword Arguments:
    smile -- the smiles string of a polymer. Must have two *
    Outputs:
    molecule -- an RDKit molecule object
    """
    if stoch:
        n_repeat = np.random.randint(
            1, 5
        )  # randomly choose a number of repeats from 1 to 4
    elif force_multiply:
        n_repeat = force_multiply
    else:
        n_repeat = 1
    if "[g]" in smile:
        polymer_class = pk.LadderPolymer
    else:
        polymer_class = pk.LinearPol
    lp = polymer_class(smile)
    pm = lp.multiply(n_repeat).PeriodicMol()  # cyclize polymer
    while pm is None:  # handle edge cases
        if n_repeat is 2:
            if stoch:
                n_repeat = np.random.randint(3, 5)
            else:
                n_repeat = 3
        if n_repeat is 1:
            if stoch:
                n_repeat = np.random.randint(2, 5)
            else:
                n_repeat = 2
        pm = lp.multiply(n_repeat).PeriodicMol()  # cyclize polymer

    if hasattr(pm, "mol"):
        molecule = pm.mol
    else:
        molecule = pm

    return molecule


def multiply_polymer_nostoch(smile):
    """
    Multiply (by some random number between 1 and 4) and cyclize the polymer smiles
    Keyword Arguments:
    smile -- the smiles string of a polymer. Must have two *
    Outputs:
    molecule -- an RDKit molecule object
    """
    mol = Chem.MolFromSmiles(smile)
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[#0]*[#0]")):
        n_repeat = 3
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("[#0]*~*[#0]")):
        n_repeat = 2
    else:
        n_repeat = 1
    if "[g]" in smile:
        polymer_class = pk.LadderPolymer
    else:
        polymer_class = pk.LinearPol
    lp = polymer_class(smile)
    pm = lp.multiply(n_repeat).PeriodicMol()  # cyclize polymer
    if hasattr(pm, "mol"):
        molecule = pm.mol
    else:
        molecule = pm
    return molecule


def shuffle_polymer(smile=None, randomize=None):
    """
    Shuffle the polymer smiles into an equivalent, random molecule.
    Randomize == 'root' --> the first atom in the SMILES is randomly selected and all other atom subsequent indices are based on connectivity
    Randomize == 'all' --> the index of all atoms are chosen randomly
    """
    mol = Chem.MolFromSmiles(smile)
    n_atoms = mol.GetNumAtoms()
    if randomize == "root":
        root = np.random.randint(0, n_atoms)
        return Chem.MolFromSmiles(Chem.MolToSmiles(mol, rootedAtAtom=root))
    elif randomize == "all":
        shuffled_inds = list(range(n_atoms))
        random.shuffle(shuffled_inds)  # randomly shuffle the atom indices in place
        return Chem.rdmolops.RenumberAtoms(mol, shuffled_inds)


@dataclass
class BondConfig:
    """
    A class to configure which features should be included
    in the bond fingerprint.

    Args:
        bond_type (bool): If True, include bond type.
        conjugation (bool): If True, include bond conjugation.
        ring (bool): If True, include bond ring information.
        stereo (bool): If True, include bond stereo information.
        bond_dir (bool): If True, include bond direction information.
    """

    bond_type: bool
    conjugation: bool
    ring: bool
    stereo: bool
    bond_dir: bool

    def __post_init__(self):
        self.n_features = 0
        self.feat_names = []
        if self.bond_type:
            self.n_features += 4
            self.feat_names += [
                "Single",
                "Double",
                "Triple",
                "Aromatic",
            ]
        if self.conjugation:
            self.n_features += 1
            self.feat_names += ["Conjugation"]
        if self.ring:
            self.n_features += 1
            self.feat_names += ["inRing"]

        if self.stereo:
            self.n_features += 6
            self.feat_names += [
                "any",
                "cis",
                "e",
                "none",
                "trans",
                "z",
            ]

        if self.bond_dir:
            self.n_features += 7
            self.feat_names += [
                "begin_dash",
                "begin_wedge",
                "either_double",
                "end_down_right",
                "end_up_right",
                "none",
                "unknown",
            ]


@dataclass
class AtomConfig:
    """
    A class to configure which features should be included
    in the atom fingerprint.

    Args:
        element_type (bool): If True, include element type.
        degree (bool): If True, include degree.
        implicit_valence (bool): If True, include implicit valence.
        formal_charge (bool): If True, include formal charge.
        num_rad_e (bool): If True, include number of radical electrons.
        hybridization (bool): If True, include hybridization type.
        combo_hybrid (bool): If True, sp2 and sp3 will be merged into one feature.
        aromatic (bool): If True, include aromaticity.
        chirality (bool): If True, include chirality.
    """

    element_type: bool
    degree: bool
    implicit_valence: bool
    formal_charge: bool
    num_rad_e: bool
    hybridization: bool
    combo_hybrid: bool  # if True, sp2 and sp3 will be merged into one feature
    aromatic: bool
    chirality: bool

    def __post_init__(self):
        self.n_features = 0
        self.feat_names = []

        def update(names):
            self.feat_names += names
            self.n_features += len(names)

        if self.element_type:
            update(element_names)
        if self.degree:
            update([f"degree{ind}" for ind in range(11)])
        if self.implicit_valence:
            update([f"implicitValence{ind}" for ind in range(7)])
        if self.formal_charge:
            update(["formalCharge"])
        if self.num_rad_e:
            update(["numRadElectons"])
        if self.hybridization:
            if not self.combo_hybrid:
                update(
                    [
                        "HybridizationType.SP",
                        "HybridizationType.SP2",
                        "HybridizationType.SP3",
                        "HybridizationType.SP3D",
                        "HybridizationType.SP3D2",
                    ]
                )
            else:
                update(
                    [
                        "HybridizationType.SP",
                        "HybridizationType.SP2or3",
                        "HybridizationType.SP3D",
                        "HybridizationType.SP3D2",
                    ]
                )
        if self.aromatic:
            update(["Aromatic"])

        if self.chirality:
            update(
                [
                    "Unspecified",
                    "Tetrahedral_CW",
                    "Tetrahedral_CCW",
                    "Other",
                    "Tetrahedral",
                    "Allene",
                    "Square_planar",
                    "Trigonal_bipyramidal",
                    "Octahedral",
                ]
            )


def bond_fp(bond, config):
    """
    Helper method used to compute bond feature vectors.

    Args:
        bond (rdkit.Chem.rdchem.Bond): A bond object
        config (BondConfig): A BondConfig object.

    Returns:
        bond_feature (list): A list of bond features
    """

    bt = bond.GetBondType()
    bond_feats = []
    if config.bond_type:
        bond_feats += [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
        ]
    if config.conjugation:
        bond_feats.append(bond.GetIsConjugated())
    if config.ring:
        bond_feats.append(bond.IsInRing())

    if config.stereo:
        stereo = bond.GetStereo()
        bond_feats += [
            stereo == Chem.rdchem.BondStereo.STEREOANY,
            stereo == Chem.rdchem.BondStereo.STEREOCIS,
            stereo == Chem.rdchem.BondStereo.STEREOE,
            stereo == Chem.rdchem.BondStereo.STEREONONE,
            stereo == Chem.rdchem.BondStereo.STEREOTRANS,
            stereo == Chem.rdchem.BondStereo.STEREOZ,
        ]

    if config.bond_dir:
        bond_feats += [
            bond.GetBondDir() == Chem.rdchem.BondDir.BEGINDASH,
            bond.GetBondDir() == Chem.rdchem.BondDir.BEGINWEDGE,
            bond.GetBondDir() == Chem.rdchem.BondDir.EITHERDOUBLE,
            bond.GetBondDir() == Chem.rdchem.BondDir.ENDDOWNRIGHT,
            bond.GetBondDir() == Chem.rdchem.BondDir.ENDUPRIGHT,
            bond.GetBondDir() == Chem.rdchem.BondDir.NONE,
            bond.GetBondDir() == Chem.rdchem.BondDir.UNKNOWN,
        ]

    return bond_feats


def one_of_k_encoding(x, allowable_set):
    """Encodes elements of a provided set as integers.
    Parameters
    ----------
    x: object
      Must be present in `allowable_set`.
    allowable_set: list
      List of allowable quantities.
    Example
    -------
    >>> import deepchem as dc
    >>> dc.feat.graph_features.one_of_k_encoding("a", ["a", "b", "c"])
    [True, False, False]
    Raises
    ------
    `ValueError` if `x` is not in `allowable_set`.
    """
    if x not in allowable_set:
        raise ValueError("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element.
    Unlike `one_of_k_encoding`, if `x` is not in `allowable_set`, this method
    pretends that `x` is the last element of `allowable_set`.
    Parameters
    ----------
    x: object
      Must be present in `allowable_set`.
    allowable_set: list
      List of allowable quantities.
    Examples
    --------
    >>> dc.feat.graph_features.one_of_k_encoding_unk("s", ["a", "b", "c"])
    [False, False, True]
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


element_names = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",  # H?
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
    "Unknown",
]


def atom_fp(atom, atom_config: AtomConfig):
    """
    Helper method used to compute per-atom feature vectors.

    Args:
        atom: RDKit Atom object
        atom_config: AtomConfig object

    Returns:
        List of atom features.
    """
    results = []
    if atom_config.element_type:
        results += one_of_k_encoding_unk(
            atom.GetSymbol(),
            element_names,
        )
    if atom_config.degree:
        results += one_of_k_encoding(
            atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        )
    if atom_config.implicit_valence:
        results += one_of_k_encoding_unk(
            atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]
        )
    if atom_config.formal_charge:
        results += [atom.GetFormalCharge()]
    if atom_config.num_rad_e:
        results += [atom.GetNumRadicalElectrons()]
    if atom_config.hybridization:
        feat = atom.GetHybridization()
        if atom_config.combo_hybrid:
            if (feat == Chem.rdchem.HybridizationType.SP2) or (
                feat == Chem.rdchem.HybridizationType.SP3
            ):
                feat = "SP2/3"
            options = [
                Chem.rdchem.HybridizationType.SP,
                "SP2/3",
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ]
        else:
            options = [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ]
        results += one_of_k_encoding_unk(feat, options)
    if atom_config.aromatic:
        results += [atom.GetIsAromatic()]

    if atom_config.chirality:
        tag = str(atom.GetChiralTag())
        results += [
            tag == "CHI_UNSPECIFIED",
            tag == "CHI_TETRAHEDRAL_CW",
            tag == "CHI_TETRAHEDRAL_CCW",
            tag == "CHI_OTHER",
            tag == "CHI_TETRAHEDRAL",
            tag == "CHI_ALLENE",
            tag == "CHI_SQUAREPLANAR",
            tag == "CHI_TRIGONALBIPYRAMIDAL",
            tag == "CHI_OCTAHEDRAL",
        ]

    return results


def add_mt_labels_and_selectors(data_list, property_df, name_map):
    """
    Add labels and selector to data for multi-task models.
    IMPORTANT: The order of data_list and property_df should be the same
    Keyword arguments:
    data_list -- a *list* of PyTorch Geometric *Data* objects
    property_df -- a pandas *DataFrame* that contains all target properties
                   (with multiple targets per row)
    name_map -- an *Ordered Dictionary* mapping the colloquial property name to the
                corresponding key in property_df. ex) {'Frequency': freq}
    Returns:
    a *list* of PyTorch Geometric *Data* objects
    """
    # check data types
    if type(name_map) != OrderedDict:
        raise TypeError("The name_map you gave is not an OrderedDict.")

    selector_dim = len(name_map)

    # make helper
    def helper(value):
        if math.isnan(value):
            return 0
        else:
            return 1

    # Make copies of feature Data with 'y' and 'selector' ####
    for ind, data_point in enumerate(data_list):
        prop_df_ind = property_df.index[ind]
        targets = property_df.loc[prop_df_ind, list(name_map.values())]
        selector = torch.tensor(
            [helper(x) for x in targets.values.tolist()], dtype=torch.float
        ).view(1, selector_dim)
        data_point.selector = selector
        targets = targets.fillna(value=constants.FILLIN)
        data_point.y = torch.tensor(targets, dtype=torch.float).view(1, selector_dim)
    # ########################################################

    return data_list


def get_valid_multiplications(smile, upper_bound):
    # if we have a ladder polymer, all multiplications are valid
    if "[g]" in smile:
        return list(range(1, upper_bound + 1))
    # if we do not have a ladder polymer, we need to make some checks
    # to see which multiplications are valid
    valid = list(range(3, upper_bound + 1))
    mol = Chem.MolFromSmiles(smile)
    if mol.HasSubstructMatch(Chem.MolFromSmarts("[#0]~*~[#0]")):
        pass
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("[#0]~*~*~[#0]")):
        valid.append(2)
    else:
        valid.extend([1, 2])

    return sorted(valid)


def multiply_and_cyclize(smile, n_repeat):
    if "[g]" in smile:
        polymer_class = pk.LadderPolymer
    else:
        polymer_class = pk.LinearPol
    try:
        lp = polymer_class(smile)
    except AttributeError as e:
        print(f"This polymer failed {smile}", flush=True)
        raise e
    pm = lp.multiply(n_repeat).PeriodicMol()
    if hasattr(pm, "mol"):
        molecule = pm.mol
    else:
        molecule = pm

    return molecule


def get_equivalent_graph_tensors(smile, upper_bound, bond_config, atom_config):
    valid_n_repeats = get_valid_multiplications(smile, upper_bound)

    return [
        construct_polymer_graph_monocycle(
            multiply_and_cyclize(smile, n_repeat), bond_config, atom_config
        )
        for n_repeat in valid_n_repeats
    ]


def get_minimum_graph_tensor(smile, bond_config, atom_config, representation):
    if representation == "monocycle":
        valid_n_repeats = get_valid_multiplications(smile, 3)
        n_repeat = valid_n_repeats[0]
        return construct_polymer_graph_monocycle(
            multiply_and_cyclize(smile, n_repeat), bond_config, atom_config
        )
    elif representation == "trimer":
        if "[g]" in smile:
            polymer = pk.LadderPolymer(smile)
            trimer_class = TrimerLadder
        else:
            polymer = pk.LinearPol(smile)
            trimer_class = TrimerLinear
        return construct_polymer_graph_trimer(
            polymer, bond_config, atom_config, trimer_class
        )


def construct_polymer_graph_trimer(
    polymer,
    bond_config,
    atom_config,
    trimer_class,
):
    trimer = trimer_class(polymer)
    pair_info = [
        (k, v) for k, v in trimer.parent_to_self.items() if v != None
    ]  # get the pairs of trimer.parent_to_self
    # that do not correspond to star indices
    pair_info = sorted(pair_info, key=lambda x: x[0])
    nodes_to_search = [x[1] for x in pair_info]
    # compute node features
    node_features = [atom_helper(trimer.mol, i, atom_config) for i in nodes_to_search]
    # ######################
    # compute edge features
    # ######################
    edge_features = []
    edge_indices = []

    ind = 0
    for bond in trimer.mol.GetBonds():
        b_idx_tri = bond.GetBeginAtomIdx()
        e_idx_tri = bond.GetEndAtomIdx()
        # we only need to featurize a subset of all the bonds in the trimer. So let us
        # check if the current bond qualifies.
        # TODO: Break early after we have gone through all the bonds we need
        if (b_idx_tri in nodes_to_search) and (e_idx_tri in nodes_to_search):
            b_idx_cycle = trimer.parent_to_cycle[trimer.self_to_parent[b_idx_tri]]
            e_idx_cycle = trimer.parent_to_cycle[trimer.self_to_parent[e_idx_tri]]
            edge_indices.append([b_idx_cycle, e_idx_cycle])
            edge_indices.append(
                [e_idx_cycle, b_idx_cycle]
            )  # add "opposite" edge for undirected graph
            # bond_feature = bond_fp(bond).reshape((6,))  # ADDED edge feat
            bond_feature = bond_fp(bond, bond_config)
            ##### bond feature list #####
            # type of bond (1,2,3,1.5,etc.), one-hot
            # is conjugated?
            # is in ring?
            # chirality
            #############################
            edge_features.extend(
                [bond_feature, bond_feature]
            )  # both bonds have the same features
            ind += 1

    # now let us add in the bond(s) across the periodic boundary
    def update_edge_indices(b_idx_cycle, e_idx_cycle):
        """
        Helper function
        """
        edge_indices.append([b_idx_cycle, e_idx_cycle])
        edge_indices.append([e_idx_cycle, b_idx_cycle])

    def update_edge_features(bond):
        """
        Helper function
        """
        bond_feature = bond_fp(bond, bond_config)
        edge_features.extend([bond_feature, bond_feature])

    if trimer_class == TrimerLinear:
        bond = trimer.periodic_bond_analogues
        b_idx_cycle = trimer.parent_to_cycle[trimer.parent_connector_inds[0]]
        e_idx_cycle = trimer.parent_to_cycle[trimer.parent_connector_inds[1]]
        update_edge_indices(b_idx_cycle, e_idx_cycle)
        update_edge_features(bond)

    elif trimer_class == TrimerLadder:
        for bond, abbrev in zip(trimer.periodic_bond_analogues, ["A", "B"]):
            parent_connector_ind1 = getattr(trimer, f"parent_connector{abbrev}1_ind")
            parent_connector_ind2 = getattr(trimer, f"parent_connector{abbrev}2_ind")
            b_idx_cycle = trimer.parent_to_cycle[parent_connector_ind1]
            e_idx_cycle = trimer.parent_to_cycle[parent_connector_ind2]
            update_edge_indices(b_idx_cycle, e_idx_cycle)
            update_edge_features(bond)

    # ############################
    # Now we make the Data object
    # ############################
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edge_indices, dtype=torch.long)
        .t()
        .contiguous(),  # as shown in https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
        edge_weight=torch.tensor(edge_features, dtype=torch.float),
    )
    return data


def get_graph_tensor_debug_dimer(sm, bond_config, atom_config, representation):
    """
    Similar function to get_minimum_graph_tensor except the graph is generated
    directly from the mol instead of from a SMILES
    *Warning!* This function is only meant to be used for debugging purposes.
    """
    if representation == "monocycle":
        mol = multiply_and_cyclize(sm, 2)
        return construct_polymer_graph_monocycle(mol, bond_config, atom_config)
    elif representation == "trimer":
        if "[g]" in sm:
            polymer = pk.LadderPolymer(sm).multiply(2)
            trimer_class = TrimerLadder
        else:
            polymer = pk.LinearPol(sm).multiply(2)
            trimer_class = TrimerLinear
        return construct_polymer_graph_trimer(
            polymer, bond_config, atom_config, trimer_class
        )
