dota_dict = {'root_path': '/home/pierre/Documents/PHD/Datasets/DOTA_v2/coco_format/',
                'name': 'DOTA',
                'splits': ['train2017', 'test2017', 'val2017'],
                'class_split_file': 'classes_split.txt'}

dior_dict = {'root_path': '/home/pierre/Documents/PHD/Datasets/DIOR/coco_format',
                'name': 'DIOR',
                'splits': ['train', 'test', 'val'],
                'class_split_file': 'classes_split.txt'}

pascal_dict = {'root_path': '/home/pierre/Documents/PHD/Datasets/VOC_COCO_FORMAT/Merged',
                'name': 'PASCAL_VOC',
                'splits': ['train2017', 'test2017', 'val2017'],
                'class_split_file': 'classes_split.txt'}

coco_dict = {'root_path': '/home/pierre/Documents/PHD/Datasets/MSCOCO',
                'name': 'COCO',
                'splits': ['train2017','val2017'],
                'class_split_file': 'classes_split.txt'}

LOCAL_CATALOG = {'DOTA': dota_dict,
                'DIOR': dior_dict,
                'PASCAL': pascal_dict,
                'COCO': coco_dict}