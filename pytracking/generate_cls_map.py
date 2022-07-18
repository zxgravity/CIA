import os

LaSOT_path = '/media/syj/d7994b66-3181-4af4-ae0f-d97d09ad6079/pzx/dataset/LaSOT/'
dir_list = os.listdir(LaSOT_path)
upper_cls_list = []
for cls in dir_list:
    if '.' not in cls:
        upper_cls_list.append(cls)

upper_cls_list.sort()
with open('upper_cls_map.txt', 'w') as f:
    for i in range(len(upper_cls_list)):
        f.write(upper_cls_list[i] + '\n')
