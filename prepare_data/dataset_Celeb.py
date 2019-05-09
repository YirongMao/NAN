import numpy as np
import scipy.io as sio
import os
import pickle


def save(lst, path):
    with open(path, 'wb') as fp:
        pickle.dump(lst, fp)


def fill_str(a):
    nstr = str(a)
    for i in range(4 - len(nstr)):
        nstr = '0' + nstr

    return nstr


def get_close_set():
    num_sub = 1000
    data_dir = '/home/maoyirong/data/Celebrity-1000/resnet50_128_feat'
    save_dir = './data/Celeb/close'
    data_path = './data/Celeb/close/sub_video_seq_{}'.format(num_sub)
    lst_seq = sio.loadmat(data_path)
    gallery = lst_seq['gallery']
    probe = lst_seq['probe']
    print('gallery and probe size {} {}'.format(gallery.shape[0], probe.shape[0]))

    subs = []
    sub_faces = []
    for i in range(gallery.shape[0]):
        sub = fill_str(gallery[i, 0])
        video = fill_str(gallery[i, 1])
        seq = str(gallery[i, 2])

        sub_exist = True
        if subs.count(sub) == 0:
            subs.append(sub)
            sub_faces.append([])
            sub_exist = False
        idx = subs.index(sub)

        mat_path = os.path.join(data_dir, sub, video, seq + '_h.mat')
        if os.path.exists(mat_path):
            arr = sio.loadmat(mat_path)
            arr = arr['feat']

            # arr_h = sio.loadmat(os.path.join(data_dir, sub, video, seq + '.mat'))
            # arr = np.vstack([arr, arr_h['feat']])
            if len(arr) == 0:
                print(mat_path)
                continue
            if len(sub_faces[idx]) == 0:
                sub_faces[idx] = arr
            else:
                sub_faces[idx] = np.vstack([sub_faces[idx], arr])

    lst_subs = []
    lst_sub_faces = []
    min_len = 100000
    for i in range(len(subs)):
        if len(sub_faces[i]) > 0:
            lst_sub_faces.append(sub_faces[i])
            lst_subs.append(subs[i])

            if np.size(sub_faces[i], 0) < min_len:
                min_len = np.size(sub_faces[i], 0)

    print('min gallery sub faces {}'.format(min_len))
    save(lst_sub_faces, os.path.join(save_dir, 'train_{}.bin'.format(num_sub)))


##  probe
    min_len = 100000
    lst_probe_video = []
    lst_video_no = []
    K = 30
    for i in range(probe.shape[0]):
        sub = fill_str(probe[i, 0])
        video = fill_str(probe[i, 1])
        # seq = str(probe[i, 2])

        if lst_subs.count(sub) == 0:
            continue

        idx = lst_subs.index(sub)
        if lst_video_no.count(sub + '_' + video) == 0:
            lst_mat = [file for file in os.listdir(os.path.join(data_dir, sub, video)) if file.endswith('_h.mat')]
            arr = []
            for mat in lst_mat:
                mat_path = os.path.join(data_dir, sub, video, mat)
                if os.path.exists(mat_path):
                    item = sio.loadmat(mat_path)
                    arr.append(item['feat'])
            arr = np.concatenate(arr, axis=0)
            # arr_h = sio.loadmat(os.path.join(data_dir, sub, video, seq + '.mat'))
            # arr = np.vstack([arr, arr_h['feat']])
            if arr.shape[0] < K:
                print(sub + '_' + video)
                continue
            lst_probe_video.append([idx, arr])
            lst_video_no.append(sub + '_' + video)

    # print('min proble seq len {}'.format(min_len))
    save(lst_probe_video, os.path.join(save_dir, 'test_video_{}.bin'.format(num_sub)))
    print('finished')


def get_close_set_ori():
    num_sub = 100
    data_dir = '/home/maoyirong/data/Celebrity-1000/resnet50_128_feat'
    save_dir = './data/Celeb/close'
    data_path = './data/Celeb/close/sub_video_seq_{}'.format(num_sub)
    lst_seq = sio.loadmat(data_path)
    gallery = lst_seq['gallery']
    probe = lst_seq['probe']
    print('gallery and probe size {} {}'.format(gallery.shape[0], probe.shape[0]))

    subs = []
    sub_faces = []
    for i in range(gallery.shape[0]):
        sub = fill_str(gallery[i, 0])
        video = fill_str(gallery[i, 1])
        seq = str(gallery[i, 2])

        sub_exist = True
        if subs.count(sub) == 0:
            subs.append(sub)
            sub_faces.append([])
            sub_exist = False
        idx = subs.index(sub)

        mat_path = os.path.join(data_dir, sub, video, seq + '_h.mat')
        if os.path.exists(mat_path):
            arr = sio.loadmat(mat_path)
            arr = arr['feat']

            arr_h = sio.loadmat(os.path.join(data_dir, sub, video, seq + '.mat'))
            arr = np.vstack([arr, arr_h['feat']])
            if len(arr) == 0:
                print(mat_path)
                continue
            if len(sub_faces[idx]) == 0:
                sub_faces[idx] = arr
            else:
                sub_faces[idx] = np.vstack([sub_faces[idx], arr])

    lst_subs = []
    lst_sub_faces = []
    min_len = 100000
    for i in range(len(subs)):
        if len(sub_faces[i]) > 0:
            lst_sub_faces.append(sub_faces[i])
            lst_subs.append(subs[i])

            if np.size(sub_faces[i],0) < min_len:
                min_len = np.size(sub_faces[i],0)

    print('min gallery sub faces {}'.format(min_len))
    save(lst_sub_faces, os.path.join(save_dir, 'train_{}.bin'.format(num_sub)))

##  probe
    min_len = 100000
    lst_probe_seqs = []
    for i in range(probe.shape[0]):
        sub = fill_str(probe[i, 0])
        video = fill_str(probe[i, 1])
        seq = str(probe[i, 2])

        if lst_subs.count(sub) == 0:
            continue
        idx = lst_subs.index(sub)

        mat_path = os.path.join(data_dir, sub, video, seq + '_h.mat')
        if os.path.exists(mat_path):
            arr = sio.loadmat(mat_path)
            arr = arr['feat']

            arr_h = sio.loadmat(os.path.join(data_dir, sub, video, seq + '.mat'))
            arr = np.vstack([arr, arr_h['feat']])
            if len(arr) == 0:
                print(mat_path)
                continue
            if np.size(arr, 0) < min_len:
                min_len = np.size(arr, 0)
            lst_probe_seqs.append([idx, arr])

    print('min proble seq len {}'.format(min_len))
    save(lst_probe_seqs, os.path.join(save_dir, 'test_{}.bin'.format(num_sub)))
    print('finished')


if __name__ == '__main__':
    get_close_set()
