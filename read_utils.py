import numpy as np
import pickle
import os


class Dataset(object):
    def __init__(self, lst_sub_faces, batch_size, num_frames, seed=66):
        self.lst_sub_faces = lst_sub_faces
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.seed = seed
        np.random.seed(seed=self.seed)

        lst_sub_idx = []
        lst_sub_cursor = []
        for i in range(len(self.lst_sub_faces)):
            idx = np.random.permutation(np.size(lst_sub_faces[i], 0))
            lst_sub_idx.append(idx)
            lst_sub_cursor.append(0)

        self.lst_sub_idx = lst_sub_idx
        self.lst_sub_cursor = lst_sub_cursor

    def get_one_set(self, sub):
        sub_faces = self.lst_sub_faces[sub]
        sub_idx = self.lst_sub_idx[sub]

        if self.num_frames > sub_faces.shape[0]:
            idx = np.random.choice(sub_faces.shape[0], self.num_frames, replace=True)
        else:
            if self.lst_sub_cursor[sub] + self.num_frames > sub_faces.shape[0]:
                sub_idx = np.random.permutation(sub_faces.shape[0])
                self.lst_sub_idx[sub] = sub_idx
                self.lst_sub_cursor[sub] = 0

            idx = sub_idx[self.lst_sub_cursor[sub]: self.lst_sub_cursor[sub] + self.num_frames]
            self.lst_sub_cursor[sub] += self.num_frames

        return sub_faces[idx, :]

    def next_pair_batch(self, set_per_sub=3):
        sub_c = np.random.choice(len(self.lst_sub_faces), self.batch_size, replace=True)
        batch = []
        lst_len = []
        lst_label = []
        # set_per_sub = 3

        for s in sub_c:
            for i in range(0, set_per_sub):
                set = self.get_one_set(s)
                batch.append(set)
                lst_len.append(set.shape[0])
                lst_label.append(s)

        return batch, lst_label, lst_len

    def next_batch(self):
        sub_c = np.random.choice(len(self.lst_sub_faces), self.batch_size, replace=True)
        batch = []
        lst_len = []
        lst_label = []
        for s in sub_c:
            set = self.get_one_set(s)
            batch.append(set)
            lst_len.append(set.shape[0])
            lst_label.append(s)

        return batch, lst_label, lst_len


if __name__ == '__main__':

    num_sub = 100
    meta_dir = './data/Celeb/close'
    lst_sub_faces  = pickle.load(open(os.path.join(meta_dir, 'train_{}.bin'.format(num_sub)), 'rb'))
    dataset = Dataset(lst_sub_faces, 512, 20)
    while True:
        batch, lst_label, lst_len = dataset.next_batch()

