import numpy as np

data = np.load("/home/duanjw/workspace/OStars/ZJ/Transformer_bias_add_moleculenet/azobenzene_dft.npz")

print(data.files)

print(data['E'].shape)

print(data['name'])

print(data['F'].shape)

print(data['theory'])

print(data['R'].shape)

print(data['z'])

print(data['type'])

print(data['md5'])


