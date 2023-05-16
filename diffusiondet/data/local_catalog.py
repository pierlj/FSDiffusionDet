dota_dict = {'root_path': '/gpfsscratch/rech/vlf/ues92cf/DOTA/coco_format/',
                'name': 'DOTA',
                'splits': ['train2017', 'test2017', 'val2017'],
                'class_split_file': '/gpfsscratch/rech/vlf/ues92cf/DOTA/coco_format/classes_split.txt'}

dior_dict = {'root_path': '/gpfsscratch/rech/vlf/ues92cf/DIOR/',
                'name': 'DIOR',
                'splits': ['train2017', 'test2017', 'val2017'],
                'class_split_file': '/gpfsscratch/rech/vlf/ues92cf/DOTA/coco_format/classes_split.txt'}

pascal_dict = {'root_path': '/gpfsscratch/rech/vlf/ues92cf/PascalVOC/',
                'name': 'PASCAL_VOC',
                'splits': ['train', 'test', 'val'],
                'class_split_file': '/gpfsscratch/rech/vlf/ues92cf/PascalVOC/classes_split.txt'}

coco_dict = {'root_path': '/gpfsdswork/dataset/COCO',
                'name': 'COCO',
                'splits': ['train2017','val2017'],
                'class_split_file': '/gpfswork/rech/vlf/ues92cf/Datasets/COCO/classes_split.txt'}

xview_dict = {'root_path': '/home/pierre/Documents/PHD/Datasets/xView/coco_format',
                'name': 'XVIEW',
                'splits': ['train', 'test', 'val'],
                'class_split_file': '/home/pierre/Documents/PHD/Datasets/xView/classes_split.txt'}

LOCAL_CATALOG = {'DOTA': dota_dict,
                'DIOR': dior_dict,
                'PASCAL': pascal_dict,
                'COCO': coco_dict,
                'XVIEW': xview_dict}