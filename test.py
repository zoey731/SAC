# from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit.Chem import Draw

# mol = Chem.MolFromSmiles('CC(=O)OC1=CC=CC=C1C(=O)O')

# Draw.MolToFile(mol, 'molecule.png',size=(300, 300))

# m_3d = Chem.AddHs(mol)

# cids = AllChem.EmbedMultipleConfs(m_3d, numConfs=10,params = AllChem.ETKDGv3())
# AllChem.MMFFOptimizeMoleculeConfs(m_3d)

# Chem.MolToPDBFile(m_3d, 'molecule.pdb')

import mdtraj as md
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

traj = md.load('molecule.pdb')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(traj.n_frames):
    coords = traj.xyz[i]
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=20)

plt.show()