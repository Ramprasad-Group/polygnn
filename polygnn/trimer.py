from polygnn.utils import lazy_property


class Trimer:
    def __init__(self) -> None:
        """
        Keyword arguments
            parent (LinearPol)
        """
        self.mol = None
        self.n_parent_atoms = None
        self.parent_star_inds = None
        self.parent_connector_inds = None

    @lazy_property
    def n_atoms(self):
        return self.mol.GetNumAtoms()

    @lazy_property
    def parent_to_self(self):
        parent_atoms = [
            i for i in range(self.n_parent_atoms) if i not in self.parent_star_inds
        ]
        # first map the atoms that were not stars in the parent
        n_subtract = len(self.parent_star_inds) // 2
        _dict = {
            val: i + (self.n_parent_atoms - n_subtract)
            for i, val in enumerate(parent_atoms)
        }

        # now map the stars to None
        _dict.update({k: None for k in self.parent_star_inds})

        return _dict

    @lazy_property
    def self_to_parent(self):

        return {v: k for k, v in self.parent_to_self.items() if v != None}

    @lazy_property
    def parent_to_cycle(self):
        parent_nonstars = sorted(
            [x for x in range(self.n_parent_atoms) if x not in self.parent_star_inds]
        )
        return {v: k for k, v in enumerate(parent_nonstars)}


class TrimerLinear(Trimer):
    def __init__(self, parent) -> None:
        """
        Keyword arguments
            parent (LinearPol)
        """
        super().__init__()
        self.mol = parent.multiply(3).mol
        self.n_parent_atoms = parent.mol.GetNumAtoms()
        self.parent_star_inds = parent.star_inds
        self.parent_connector_inds = parent.connector_inds

    @lazy_property
    def analogue_inds(self):
        """
        Return the start atom index and end atom index (from self) of
        the bond (from self) that maps to the periodic bond of parent
        """
        return (
            self.parent_to_self[self.parent_connector_inds[-1]],
            self.parent_to_self[self.parent_connector_inds[0]]
            + (self.n_parent_atoms - 2),
        )

    @lazy_property
    def periodic_bond_analogues(self):
        """
        Return the bond object (from self) that is analagous to the
        bond across the periodic boundary of parent
        """

        return self.mol.GetBondBetweenAtoms(
            self.analogue_inds[0], self.analogue_inds[1]
        )


class TrimerLadder(Trimer):
    def __init__(self, parent) -> None:
        """
        Keyword arguments
            parent (LadderPolymer)
        """
        super().__init__()
        self.mol = parent.multiply(3).mol
        self.n_parent_atoms = parent.mol.GetNumAtoms()
        # A1,A2 share a bond in real macromolecule
        # B1,B2 share a bond in real macromolecule
        # A1,B1 share a ring in repeat unit
        # A2,B2 share a ring in repeat unit
        self.parent_starA1_ind = parent.starA1_ind
        self.parent_starA2_ind = parent.starA2_ind
        self.parent_starB1_ind = parent.starB1_ind
        self.parent_starB2_ind = parent.starB2_ind

        self.parent_star_inds = [
            self.parent_starA1_ind,
            self.parent_starA2_ind,
            self.parent_starB1_ind,
            self.parent_starB2_ind,
        ]

        self.parent_connectorA1_ind = parent.connectorA1_ind
        self.parent_connectorA2_ind = parent.connectorA2_ind
        self.parent_connectorB1_ind = parent.connectorB1_ind
        self.parent_connectorB2_ind = parent.connectorB2_ind

    @lazy_property
    def analogue_inds(self):
        """
        Return the start atom index and end atom index (from self) of
        the two bonds (from self) that map to the periodic bonds of parent
        """
        # order matters, A first
        tail = [self.parent_starA2_ind, self.parent_starB2_ind]
        parent_to_parenthead = {
            val: ind
            for ind, val in enumerate(
                [x for x in range(self.n_parent_atoms) if x not in tail]
            )
        }
        return [
            (
                self.parent_to_self[self.parent_connectorA1_ind],
                parent_to_parenthead[self.parent_connectorA2_ind],
            ),
            (
                self.parent_to_self[self.parent_connectorB1_ind],
                parent_to_parenthead[self.parent_connectorB2_ind],
            ),
        ]

    @lazy_property
    def periodic_bond_analogues(self):
        """
        Return the bond object (from self) that is analagous to the
        bond across the periodic boundary of parent
        """
        return [
            self.mol.GetBondBetweenAtoms(pair[0], pair[1])
            for pair in self.analogue_inds
        ]
