import numpy as np
import os
from xml.dom.minidom import parse
import pandas as pd

"""-----------------配置文件路径----------------------"""
gt_path='../../../dataset/hongqiao/gt/gt'
density_path='../../../dataset/hongqiao/gt/density_map'

"""---------------按顺序导入gt和density map，进行比较-----------"""
gt_list=os.listdir(gt_path)
density_list=os.listdir(density_path)

gt_list.sort()
density_list.sort()

for i in range(1,len(gt_list)):
    gt_name=gt_list[i-1]
    den_name=density_list[i-1]

    DOMTree = parse(gt_path + '/' + gt_name)
    xml_content = DOMTree.documentElement
    bndboxs = xml_content.getElementsByTagName('bndbox')

    num_gt=len(bndboxs)

    # den_data = pd.read_csv(density_path+'/'+den_name, sep=',', header=None).as_matrix()
    # den_data = den_data.astype(np.float32, copy=False)
    den_data=np.load(density_path+'/'+den_name)

    num_den=np.sum(den_data)

    if abs(num_gt-num_den)>1.5:
        print("{}--gt_num,den_num:({} {:.3f})".format(i, num_gt, num_den))
        break
    else:
        print("density map {} ok".format(i))


