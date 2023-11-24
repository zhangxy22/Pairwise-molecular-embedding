import cairosvg
from rdkit.Chem import Draw, AllChem
from rdkit import Chem
import math
import matplotlib.pyplot as plt

mol = Chem.MolFromSmiles('O=C1C(=Cc2ccc(cc2)c2ccc(s2)c2ccc3c(c2)c2ccccc2n3c2ccc(cc2)C(C)(C)C)C(=O)c2c1cccc2')
#mol = Chem.MolFromSmiles('CCCC[C@@H](Cc1ccc(s1)c1c2sc(cc2c(c2c1cc(s2)c1cc(c(s1)c1ccc(s1)c1sc(cc1CCCCCCCC)/C=C/C(=O)OCC)CCCCCCCC)c1ccc(s1)C[C@H](CCCC)CC)c1cc(c(s1)c1ccc(s1)c1sc(cc1CCCCCCCC)/C=C/C(=O)OCC)CCCCCCCC)CC')
bi = {}
fp = AllChem.GetMorganFingerprintAsBitVect(mol, nBits=1024, radius=2, bitInfo=bi)
print(bi)

bits = [935]
imgs = []
for bit in bits:
    mfp2_svg = Draw.DrawMorganBit(mol, bit, bi)
    # 将SVG转换为PNG图像
    png_data = cairosvg.svg2png(bytestring=mfp2_svg.encode())
    with open(f"morgan_bitMAXd_{bit}.png", "wb") as png_file:
        png_file.write(png_data)

    # 打开并显示PNG图像
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
