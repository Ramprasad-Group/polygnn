from polygnn_kit.polygnn_kit import LadderPolymer
from rdkit.Chem import Draw
from torch_geometric.loader import DataLoader
import numpy as np
from torch import nonzero, is_tensor
from rdkit.Chem.Draw import rdMolDraw2D
from cairosvg import svg2png

from polygnn.featurize import (
    construct_polymer_graph_trimer,
    get_equivalent_graph_tensors,
    multiply_and_cyclize,
    get_graph_tensor_debug_dimer,
)
from polygnn.trimer import TrimerLadder, TrimerLinear
from polygnn_kit.polygnn_kit import LinearPol


def atom_map_dimer(sm):
    mol = multiply_and_cyclize(sm, 1)
    n_atoms = mol.GetNumAtoms()

    return {ind: ind + n_atoms for ind in range(n_atoms)}


def bond_map_dimer(sm):
    mol = multiply_and_cyclize(sm, 2)
    n_bonds = len(mol.GetBonds())
    if "[g]" in sm:  # proxy to determine if a SMILES string is a ladder or not
        n_bonds_iter = (n_bonds - 4) // 2
        d = {ind: ind + n_bonds_iter for ind in range(n_bonds_iter)}
        d[n_bonds - 4] = n_bonds - 2
        d[n_bonds - 3] = n_bonds - 1
    else:
        n_bonds_iter = (n_bonds - 2) // 2
        d = {ind: ind + n_bonds_iter for ind in range(n_bonds_iter)}
        d[n_bonds - 2] = n_bonds - 1

    return d


def atom_diffs_dimer(sm, bond_config, atom_config, representation):
    """
    Determine which features of a polymer's superunits are not intensive,
    if any
    Keyword arguments:
        sm (str): A polymer smiles string containing two [*] or [e], [d], [t], [g]
        bond_config (BondConfig)
        atom_config (AtomConfig)
        representation (str): Either "monocycle" if we want fps computed on a
            cycle of sm or "trimer" if we want the fps computed
            on a twice-multiplied noncycle (i.e., trimer) of sm
    Outputs:
        failure_map (dict): A dictionary with the failure information. The
            dict has key value pairs that look like
            (atom x, (atom y, [featName1, featName2]))
            or (bond xx, (bond yy, [featName3, featName4])) where atom/bond
            x and y are topologically equivalent and featNameXX are the
            atom/bond features for which x/y do not contain the same value
    """

    data = get_graph_tensor_debug_dimer(sm, bond_config, atom_config, representation)
    atom_map = atom_map_dimer(sm)
    bond_map = bond_map_dimer(sm)
    atom_feat_names = np.array(atom_config.feat_names)
    bond_feat_names = np.array(bond_config.feat_names)
    # the map below logs the failure info for the dimer of sm
    # the key is the lower atom index
    # the value is a tuple of form (co_index, failed_feats) where
    # co_index is atom_map[lower atom index] and failed_feats is the
    # name of the features with mismatching features
    failure_map = {}
    for ind, val in atom_map.items():
        diff = (data.x[ind, :] - data.x[val, :]).abs()
        if diff.sum().item() > 0:
            diff_inds = nonzero(diff).flatten().detach().cpu().numpy()
            failure_map[f"atom {ind}"] = (
                f"atom {val}",
                atom_feat_names[diff_inds].tolist(),
            )

    for ind, val in bond_map.items():
        # each bond feature gets added twice to data.edge_weight so we need
        # to multiply ind and val by two when indexing data.edge_weight
        diff = (data.edge_weight[2 * ind, :] - data.edge_weight[2 * val, :]).abs()
        if diff.sum().item() > 0:
            diff_inds = nonzero(diff).flatten().detach().cpu().numpy()
            failure_map[f"bond {ind}"] = (
                f"bond {val}",
                bond_feat_names[diff_inds].tolist(),
            )

    return failure_map


def mol_with_atom_index(mol):
    """
    Label each atom in mol with index
    """
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def mol_with_bond_index(mol):
    """
    Label each bond in mol with index
    """
    for bond in mol.GetBonds():
        bond.SetBondMapNum(bond.GetIdx())
    return mol


def multiply_and_cyclize_pol(lp, n_repeat):

    pm = lp.multiply(n_repeat).PeriodicMol()
    if hasattr(pm, "mol"):
        molecule = pm.mol
    else:
        molecule = pm

    return molecule


def save_pngs_1(mol, mol_cyc1, mol_cyc2, path):
    mol_with_atom_index(mol)
    mol_with_atom_index(mol_cyc1)
    mol_with_atom_index(mol_cyc2)
    img = Draw.MolsToGridImage([mol, mol_cyc1, mol_cyc2], subImgSize=(900, 900))
    img.save(path)


def save_pngs_2(mol, path):

    d2d = rdMolDraw2D.MolDraw2DSVG(600, 600)
    d2d.drawOptions().addBondIndices = True
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    svg2png(bytestring=d2d.GetDrawingText(), write_to=path)


def check_model_intensiveness(
    sm_ls,
    representation,
    bond_config,
    atom_config,
    model,
    selector,
    node_feat,
    graph_feat,
    sim_threshold=10 ** (-4),
):
    """
    Check if a model produces intensive features on sm_ls using a given
    fingerprint representation
    """
    batch_size = 450
    if representation == "monocycle":
        all_tensors = [
            get_equivalent_graph_tensors(x, 4, bond_config)[0:2] for x in sm_ls
        ]
    elif representation == "trimer":

        def helper(sm):
            if "[g]" in sm:
                pol_class = LadderPolymer
                trimer_class = TrimerLadder
            else:
                pol_class = LinearPol
                trimer_class = TrimerLinear
            pol = pol_class(sm)
            pol_termer = pol.multiply(3)
            return [
                construct_polymer_graph_trimer(
                    pol, bond_config, atom_config, trimer_class
                ),
                construct_polymer_graph_trimer(
                    pol_termer, bond_config, atom_config, trimer_class
                ),
            ]

        all_tensors = [helper(x) for x in sm_ls]

    x1 = [x[0] for x in all_tensors]
    x2 = [x[1] for x in all_tensors]
    for x in x1:
        if is_tensor(selector):
            x.selector = selector
        if is_tensor(node_feat):
            x.node_feat = node_feat
        if is_tensor(graph_feat):
            x.graph_feat = graph_feat
    for x in x2:
        if is_tensor(selector):
            x.selector = selector
        if is_tensor(node_feat):
            x.node_feat = node_feat
        if is_tensor(graph_feat):
            x.graph_feat = graph_feat
    loader1 = DataLoader(x1, batch_size=batch_size)
    loader2 = DataLoader(x2, batch_size=batch_size)
    graph_fps_all1 = []
    graph_fps_all2 = []
    for data in loader1:
        graph_fps = (
            model(data.x, data.edge_index, data.edge_weight, data.batch)
            .detach()
            .cpu()
            .numpy()
        )
        graph_fps_all1.append(graph_fps)

    graphs_fps_all1 = np.vstack(graph_fps_all1)

    for data in loader2:
        graph_fps = (
            model(data.x, data.edge_index, data.edge_weight, data.batch)
            .detach()
            .cpu()
            .numpy()
        )
        graph_fps_all2.append(graph_fps)

    graphs_fps_all2 = np.vstack(graph_fps_all2)

    failed_inds = (
        np.abs(graphs_fps_all1 - graphs_fps_all2).sum(1) > sim_threshold
    )  # true for those smiles that failed

    print(f"{failed_inds.sum()} polymers failed.")

    for x in np.array(sm_ls)[failed_inds]:
        print(x)
