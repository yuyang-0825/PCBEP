import rdkit.Chem as Chem
from rdkit import Chem
import numpy as np



ff = open('../Data/test_data_feature_surface.txt', 'w')
# f1 = open('unbound/fasta right', 'w')

label_file = '../Data/label.txt'
# feature_file = 'AAindex.txt'
surface_file = '../Data/new_surface.txt'
# fasta_feature = 'unbound/fasta.txt'
data_num = 0
with open(label_file, 'r') as num_fp:
    data_num = len(num_fp.readlines())

def add_pssm():
    with open('../Data/label.txt', 'r') as la:
        with open('../Data/data_feature_surface.txt', 'w') as nfe:
            with open('../Data/test_data_feature_surface.txt', 'r') as fe:
                num_acid = fe.readline().strip()
                nfe.write(num_acid + '\n')
                for i in range(int(num_acid)):
                    name_pssm = la.readline().strip().split()[0]
                    la.readline()
                    name_lines = name_pssm[1:].split('-')
                    name = name_lines[0].lower() + '-' + name_lines[1]

                    with open('../Data/PSSM/' + name + '.pssm', 'r') as ps:
                        ps.readline()
                        ps.readline()
                        ps.readline()
                        num_atom = fe.readline().strip()
                        nfe.write(num_atom + '\n')
                        line = ps.readline().strip().split()
                        for i_a in range(int(num_atom.split()[0])):
                            seq_ac_pssm = int(line[0]) - 1
                            feature_a = fe.readline().strip()
                            nfe.write(feature_a + '\t')
                            seq_ac = feature_a.split()[0]
                            if seq_ac_pssm != int(seq_ac):
                                line = ps.readline().strip().split()
                                seq_ac_pssm += 1
                            pssm = line[2:22]
                            for num_p in range(pssm.__len__()):
                                nfe.write(pssm[num_p] + '\t')
                            nfe.write('\n')


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


point = 1
with open(label_file, 'r') as fp, open(surface_file) as fp3:
    # feature_list = fp2.readlines()
    ff.write(str(int(data_num/2)))
    ff.write('\t')
    ff.write('\n')
    for line1 in fp:  # line1:str '>3jcx-A\n'
        if line1.startswith('>'):
            print(line1)
            # print("原"+line1)
            # print(feature_list[point])

            fp3.readline()
            row1 = line1.strip().split()  # list:['>3jcx-A']
            file_name = row1[0][1:5]  # 3jcx
            file_name = file_name.lower()
            # print(row1[0])
            ##seq = row1[1]
            seq = row1[0][6]
            file = '../Data/data/' + file_name + '-' + seq + '.pdb'
            # file = 'PDB/' + file_name + '.pdb'
            if (Chem.MolFromPDBFile(file, removeHs=False, flavor=1, sanitize=False)):

                fasta = fp.readline().strip()
                surface = fp3.readline().strip()
                # print(fasta)
                ##seq = row1[1]
                seq = row1[0][6]

                mol = Chem.MolFromPDBFile(file, removeHs=False, flavor=1, sanitize=False)

                natoms = mol.GetNumAtoms()

                atom_feature = []

                # matrix 邻接矩阵
                # matrix = np.zeros((natoms, natoms),dtype='float32')

                ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B',
                             'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']

                atom_info = []
                i = 0
                n = -100
                p = -1
                first = 0
                last = 0
                with open(file, 'r') as f:
                    m = 0
                    for line in f:
                        # f.readline()
                        m += 1
                        row = line.strip().split()
                        if (row[0] == 'ATOM' or row[0] == 'HETATM'):
                            if (len(row[2]) == 7):
                                row[7] = row[6]
                                row[6] = row[5]
                                row[5] = row[4]
                                row[4] = row[3]
                                row[3] = row[2][4:]
                                row[2] = row[2][:4]
                            if (len(row[4]) != 1):
                                row.append('0')
                                # row[11] = row[10]
                                row[7] = row[6]
                                row[6] = row[5]
                                row[5] = row[4][1:]
                                row[4] = row[4][0]
                            atom_info.append(row)
                for i in range(len(atom_info)):
                    if (atom_info[i][4] == seq):
                        first = i
                        break
                # print(first)
                num = 0
                for i in range(len(atom_info)):
                    if (atom_info[i][4][0] == seq and atom_info[i][0] == 'ATOM'):
                        num += 1
                        last = i
                # print(last)
                # print(num)
                # 原子数
                if ((last - first + 1) != num): print(file_name + '-' + seq + '-' + 'error')
                ff.write(str(last - first + 1))
                ff.write('\t')
                ff.write('\n')

                matrix = np.zeros((last - first + 1, last - first + 1), dtype='float32')

                mol1 = mol
                mol2 = mol
                N1 = mol1.GetNumAtoms()
                N2 = mol2.GetNumAtoms()
                xyzs1 = mol1.GetConformer(0).GetPositions()
                xyzs2 = mol2.GetConformer(0).GetPositions()
                dismatrix = np.zeros((N1, N2), dtype=np.float16)
                for i in range(N1):
                    cs = np.tile(xyzs1[i], N2).reshape((N2, 3))
                    dismatrix[i] = np.linalg.norm(xyzs2 - cs, axis=1)

                xyz = mol.GetConformer(0).GetPositions()

                # 每个原子
                for i in range(first, last + 1):
                    degree = 0
                    for j in range(i + 1, last + 1):
                        bond = mol.GetBondBetweenAtoms(i, j)
                        if bond:
                            matrix[i - first, j - first] = matrix[j - first, i - first] = 1
                        else:
                            matrix[i - first, j - first] = matrix[j - first, i - first] = 0

                    #print(atom_info[i])
                    if (atom_info[i][4][0] == seq):
                        if n != atom_info[i][5]:
                            p += 1
                            label = fasta[p]
                            surface_atom = surface[p]
                            n = atom_info[i][5]
                        else:
                            label = fasta[p]
                            surface_atom = surface[p]
                        # 所在氨基酸编号
                        ff.write(str(p))
                        ff.write('\t')
                        # 标签
                        ff.write(str(label))
                        ff.write('\t')
                        # surface
                        ff.write(str(surface_atom))
                        ff.write('\t')

                        # 原子坐标
                        position = xyz[i].tolist()
                        for m in range(3):
                            ff.write(str(position[m]))
                            ff.write('\t')
                        # 原子特征
                        atom = mol.GetAtomWithIdx(i)
                        print(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST))
                        atom_feature = onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) + onek_encoding_unk(
                            atom.GetDegree(), [0, 1, 2, 3, 4, 5]) + onek_encoding_unk(atom.GetFormalCharge(),
                                                                                      [-1, -2, 1, 2,
                                                                                       0]) + onek_encoding_unk(
                            int(atom.GetChiralTag()), [0, 1, 2, 3]) + [1 if atom.GetIsAromatic() else 0]
                        # valence = atom.GetTotalValence()
                        # print(valence)
                        for x in range(39):
                            if (atom_feature[x] == False):
                                ff.write('0')
                                ff.write('\t')
                            else:
                                ff.write('1')
                                ff.write('\t')

                        ff.write('\n')

                # f1.write(file_name + '\n')
                point = point + 2
            else:
                print(file_name)
add_pssm()
