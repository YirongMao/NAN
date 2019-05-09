import numpy as np


def norm_l2(feat, eps=1e-10):
    feat = feat/np.sqrt(np.sum(np.multiply(feat, feat)) + eps)
    return feat


def cal_far(sim, actual_issame, num_interval=2000):
    fars = [0.0001, 0.001, 0.01, 0.1]
    min_sim = sim.min()
    max_sim = sim.max()
    num = num_interval # (max_sim-min_sim)//1e-2
    thr = np.linspace(min_sim, max_sim, num=num)
    # thr = np.sort(thr)
    FA = np.zeros(thr.shape, dtype=np.float32)
    TN = np.zeros(thr.shape, dtype=np.float32)
    TA = np.zeros(thr.shape, dtype=np.float32)
    acc = np.zeros(thr.shape, dtype=np.float32)
    idx_pos = np.where(actual_issame == True)[0].tolist()
    idx_neg = np.where(actual_issame == False)[0].tolist()
    num_pos = len(idx_pos)
    num_neg = len(idx_neg)

    cur = 0
    for cur_thr in thr:
        # pos_set = sim[idx_pos]
        num_ta = np.where(sim[idx_pos] > cur_thr)[0].shape[0]
        num_fa = np.where(sim[idx_neg] > cur_thr)[0].shape[0]
        num_tn = np.where(sim[idx_neg] < cur_thr)[0].shape[0]
        #
        FA[cur] = num_fa / num_neg
        TA[cur] = num_ta / num_pos
        acc[cur] = (num_ta + num_tn) / (num_neg + num_pos)
        cur += 1

    order_fa = np.arange(len(FA))
    thr = thr[order_fa]
    FA = FA[order_fa]
    TA = TA[order_fa]

    tars = []
    thrs = []
    for far in fars:
        idx_target = np.argmin(np.abs(FA - far))
        tmp = np.mean(TA[FA == FA[idx_target]])
        tars.append(tmp * 100)
        thrs.append(thr[idx_target])
        # tar, _, _ = calculate_far_tar(f(far), sim, idx_pos, idx_neg, num_neg, num_pos)
        # tars.append(tar)
    return fars, tars, thrs, FA, TA, np.max(acc)