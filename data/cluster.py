from rdkit import Chem
from sklearn.cluster import KMeans
from yellowbrick.cluster.elbow import kelbow_visualizer
import os
import pickle
import numpy as np
import pickle
from collections import defaultdict
from drfp import DrfpEncoder
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
mpl.use('Agg')


if __name__ == '__main__':
    data_prefix = "/amax/data/shirunhan/reaction_mvp/data"
    smiles_data_path = "/amax/data/group_0/yield_data/pretraining_data/reactions.pkl"
    with open(smiles_data_path, "rb") as f:
        reaction_smiles = pickle.load(f)

    """get fp"""
    def generate_fp():
        len_ = 256
        fps = np.array(DrfpEncoder.encode(reaction_smiles, n_folded_length=len_, show_progress_bar=True), dtype=np.uint8)
        with open(os.path.join(data_prefix, "drfp_cluster", "fp.pkl"), "wb") as f:
            pickle.dump(fps, f)
    # generate_fp()

    with open(os.path.join(data_prefix, "drfp_cluster", "fp.pkl"), "rb") as f:
        fps = pickle.load(f)

    """find the elbow value"""
    def find_elbow():
        kk = [2, 50, 100, 200, 500]+[i*1000 for i in range(1, 11)]
        oz = kelbow_visualizer(KMeans(random_state=511, n_init="auto"), fps, k=kk, metric='distortion')
        oz.show(outpath=os.path.join(data_prefix, "drfp_cluster", "drfp_kmeans.pdf"))
        k_ = oz.elbow_value_
        print(f"k: {k_}, elbow_score_: {oz.elbow_score_}")
    # find_elbow()

    """choose three ks"""
    def choose_ks():
        ks = [100, 1000, 4000]
        for k in ks:
            cls = KMeans(n_clusters=k, random_state=511, n_init="auto")
            cls.fit(fps)
            with open(os.path.join(data_prefix, "drfp_cluster", f"cls_{k}.pkl"), "wb") as f:
                pickle.dump(cls, f)
    # choose_ks()

    """get labels"""
    def get_labels():
        rea2labels = defaultdict(list)
        ks = [100, 1000, 4000]
        for k in ks:
            with open(os.path.join(data_prefix, "drfp_cluster", f"cls_{k}.pkl"), "rb") as f:
                cls = pickle.load(f)
            label = cls.predict(fps)
            for i, s in enumerate(reaction_smiles):
                rea2labels[s].append(label[i])
        with open(os.path.join(data_prefix, "drfp_cluster", "labels.pkl"), "wb") as f:
            pickle.dump(rea2labels, f)
    # get_labels()

    with open(os.path.join(data_prefix, "drfp_cluster", "labels.pkl"), "rb") as f:
        res2labels = pickle.load(f)
    label_dist = [[], [], []]
    label_dist_reverse = [[], [], []]
    class2num = [defaultdict(int), defaultdict(int), defaultdict(int)]
    for k, v in tqdm(res2labels.items()):
        for i in range(len(v)):
            label_dist[i].append(v[i])
            class2num[i][v[i]] += 1
    for i in range(3):
        label_dist_reverse[i] = list(class2num[i].values())

    def draw_multi_distributions(vals, bins_list=[100, 1000, 5000], p_=None):
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        for i in range(len(vals)):
            n, bins, patches = ax[i].hist(vals[i], bins_list[i], color="lightblue")
            print(int(n.min()), int(n.max()), len(bins))
            ax[i].tick_params(axis='both')
            ax[i].spines[['right', 'top']].set_visible(False)
            ax[i].grid()

        fig.tight_layout()
        if p_ is not None:
            plt.savefig(p_, format="pdf")
        plt.close()

    draw_multi_distributions(label_dist, [100-1, 1000-1, 4000-1], os.path.join(data_prefix, "drfp_cluster", "drfp_labels.pdf"))
    draw_multi_distributions(label_dist_reverse, [20, 80, 100], os.path.join(data_prefix, "drfp_cluster", "drfp_labels_num.pdf"))
    """
    4900 190137 100
    116 47986 500
    44 26593 1000
    0 23 21
    0 49 51
    0 69 81
    """
    print()
