import cairosvg
from rdkit.Chem import Draw, AllChem
from rdkit import Chem
import math
import matplotlib.pyplot as plt

mol = Chem.MolFromSmiles('O=C1C(=Cc2ccc(cc2)c2ccc(s2)c2ccc3c(c2)c2ccccc2n3c2ccc(cc2)C(C)(C)C)C(=O)c2c1cccc2')
bi = {}
fp = AllChem.GetMorganFingerprintAsBitVect(mol, nBits=1024, radius=2, bitInfo=bi)
print(bi)

bits = [935]
imgs = []
for bit in bits:
    mfp2_svg = Draw.DrawMorganBit(mol, bit, bi)
    png_data = cairosvg.svg2png(bytestring=mfp2_svg.encode())
    with open(f"morgan_bitMAXd_{bit}.png", "wb") as png_file:
        png_file.write(png_data)
    img = plt.imread(f"morgan_bitMAXd_{bit}.png")
    imgs.append(img)

def displayimgsinrow(imgs, col=4):
    plt.figure(figsize=(20, 10))
    columns = col
    rows = math.ceil(len(imgs) / columns)
    for i, image in enumerate(imgs):
        ax = plt.subplot(rows, columns, i + 1)
        ax.set_axis_off()
        plt.imshow(image)
displayimgsinrow(imgs)
