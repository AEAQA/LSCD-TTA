import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import numpy as np
from utils.tensor_utils import mkdir


def modify_image_list_paths(path):
    file_dir = Path(path)
    for i, file in enumerate(file_dir.iterdir()):
        pre = str(file).split('/')[-1].split('_')[0] + '/'  # + '/images/'
        with open(str(file).split('.')[0] + '_.txt', 'w') as fw:
            with open(str(file), 'r') as fr:
                for line in fr.readlines():
                    a = pre + line
                    fw.write(a)


def get_class_path_map_from_txt(path, class_idx=2):
    file_dir = Path(path)
    class_maps = {}
    with open(str(file_dir), 'r') as f:
        for line in f.readlines():
            path, claz = line.strip().split(' ')
            # target = path.split('/')[class_idx]
            if claz in class_maps:
                class_maps[claz].append(path)
            else:
                class_maps[claz] = [path]
    return class_maps


def get_dataset_class_map_in_folder(folder_path, class_idx):
    # only txt contains 'train' and 'test' are included in class_map
    file_dir = Path(folder_path)
    for file in file_dir.iterdir():
        if not ('train' in str(file) or 'test' in str(file)):
            continue
        clas_maps = get_class_path_map_from_txt(file, class_idx)
        class_num = max([int(i) for i in list(clas_maps.keys())]) + 1
        print(clas_maps)
        print('Class num : {}'.format(class_num))
        break


def shuffle_text(path, to_path):
    with Path(path).open('r') as f:
        lines = f.readlines()
    lines = np.random.permutation(lines)

    with Path(to_path).open('w') as f:
        for l in lines:
            f.write(l)


def generate_shuffled_text(dataset_path):
    dataset_path = Path(dataset_path)
    for file in dataset_path.iterdir():
        if file.name.endswith('test.txt'):
            target_file_name = str(file)[:-4] + '_shuffled.txt'
            shuffle_text(str(file), target_file_name)
            print(f'generated {target_file_name}')


def generate_text_list(folder, save_path):
    folder, save_path = Path(folder), Path(save_path)
    mkdir(save_path)
    for domain in sorted(folder.iterdir()):  # 使用 sorted() 函数对文件夹进行排序
        if not domain.is_dir():
            continue
        text_list = []
        train_list = []
        val_list = []
        last_count = 0
        for i, class_folder in enumerate(sorted(domain.iterdir())):  # 使用 sorted() 函数对子文件夹进行排序
            count = 0
            for img in sorted(class_folder.iterdir()):  # 使用 sorted() 函数对文件进行排序
                text_list.append(f'{domain.name}/{class_folder.name}/{img.name} {i+1}\n')
                count += 1

            for i in range(count):
                if i < round(0.9*count):
                    train_list.append(text_list[last_count+i])
                else:
                    val_list.append(text_list[last_count+i])
            last_count += count
            
        with open((save_path / f'{domain.name}_test.txt').absolute(), 'w') as f:
            f.writelines(text_list)
        
        with open((save_path / f'{domain.name}_train.txt').absolute(), 'w') as f:
            f.writelines(train_list)

        with open((save_path / f'{domain.name}_val.txt').absolute(), 'w') as f:
            f.writelines(val_list)

        print(f'Writed {domain.name}')

if __name__ == '__main__':
    generate_text_list(folder=r"../data/DataSets/a2n", save_path=r"text_lists/a2n")
    generate_shuffled_text(dataset_path=r"text_lists/a2n")
