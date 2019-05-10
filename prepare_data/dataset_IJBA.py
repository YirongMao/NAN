import os
import scipy.io as sio
import numpy as np
import pickle


def save(lst, path):
    with open(path, 'wb') as fp:
        pickle.dump(lst, fp)


def get_mat(data_dir, template_id):
    arr = []
    mat_path = os.path.join(data_dir, str(template_id) + '.mat')
    item = sio.loadmat(mat_path)
    arr.append(item['feat'])
    
    # features whose images are horizontal flipped
    mat_path = os.path.join(data_dir, str(template_id) + '_h.mat')
    item = sio.loadmat(mat_path)
    arr.append(item['feat'])
    arr = np.vstack(arr)
    return arr


def get_train_test_set():
    # feat_type = 'casianet'

    data_dir = '../data/IJBA/resnet34/resnet34_feat' # the directory which store face features of each template
    save_dir = '../data/IJBA/resnet34'
    split_dir = '../data/IJBA/IJB-A_11_sets'

    for idx_split in range(1, 11):
        # to read train file and collect training faces in to a list by subjects
        train_file = os.path.join(split_dir, 'split' + str(idx_split), 'train_' + str(idx_split) + '.csv')
        cur_line = 0
        subs = []
        templates = []
        lst_train_faces = []
        for line in open(train_file):
            cur_line += 1
            if cur_line == 1:  # skip the first line
                continue
            lst_tmp = line.split(',')
            template_id = lst_tmp[0]
            subject_id = lst_tmp[1]

            if templates.count(template_id) == 0:
                templates.append(template_id)
                if subs.count(subject_id) == 0:
                    subs.append(subject_id)
                    lst_train_faces.append([])
                idx = subs.index(subject_id)

                arr = get_mat(data_dir=data_dir, template_id=template_id)
                if len(lst_train_faces[idx]) == 0:
                    lst_train_faces[idx] = arr
                else:
                    lst_train_faces[idx] = np.vstack([lst_train_faces[idx], arr])

        # end of lines
        save(lst_train_faces, os.path.join(save_dir, 'train_subject_{}.bin'.format(idx_split)))
        print('#train subject {} split {}'.format(len(lst_train_faces), idx_split))
        
        # to get the subject id for each templates
        meta_file = os.path.join(split_dir, 'split' + str(idx_split), 'verify_metadata_' + str(idx_split) + '.csv')
        cur_line = 0
        dict_all_templates = {}
        for line in open(meta_file):
            cur_line += 1
            if cur_line == 1:
                continue
            lst_tmp = line.split(',')
            template_id = lst_tmp[0]
            subject_id = lst_tmp[1]
            dict_all_templates.update({template_id: subject_id})

        print('#total template {} split {}'.format(len(dict_all_templates), idx_split))
        
        # to read the test file and collect verification pairs 
        test_file = os.path.join(split_dir, 'split' + str(idx_split), 'verify_comparisons_' + str(idx_split) + '.csv')
        lst_test_pairs = []
        for line in open(test_file):
            lst_tmp = line.split(',')
            id_a = lst_tmp[0].replace('\n', '')
            id_b = lst_tmp[1].replace('\n', '')
            feat_a = get_mat(data_dir, id_a)
            feat_b = get_mat(data_dir, id_b)
            if dict_all_templates[id_a] == dict_all_templates[id_b]:
                pair_label = True
            else:
                pair_label = False
            lst_test_pairs.append([feat_a, feat_b, pair_label, id_a, id_b])
        save(lst_test_pairs, os.path.join(save_dir, 'lst_test_pair_{}.bin'.format(idx_split)))
        print('#pairs {}'.format(len(lst_test_pairs)))


if __name__ == '__main__':
    get_train_test_set()
