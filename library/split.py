import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def myfold(
    X,
    groups,
    n_splits=5,
    group_split_rate=0.2,
    spliter=None,
    shuffle=True,
    random_state=None,
):
    if random_state:
        np.random.seed(random_state)

    if spliter is None:
        spliter = KFold(n_splits, shuffle=shuffle, random_state=random_state)
    n_samples = len(X)

    unique_groups, groups = np.unique(groups, return_inverse=True)
    group_split_sample_num = (n_samples / n_splits) * group_split_rate
    # [index: group_id, そのグループのレコード数]
    df_group = (
        pd.value_counts(groups).to_frame("n_sample_per_group")
        # .reset_index(columns={"index", "groups"})
    )
    df_group["fold"] = -1
    if shuffle:
        df_group = df_group.sample(frac=1)

    # shape: グループ数。グループごとのサンプル数
    # n_samples_per_group = np.bincount(groups)

    n_samples_per_fold = np.zeros(n_splits)
    for group_index, data in df_group.iterrows():
        n_sample_in_group = data["n_sample_per_group"]
        lightest_fold = np.argmin(n_samples_per_fold)
        n_samples_per_fold[lightest_fold] += n_sample_in_group
        df_group["fold"][group_index] = lightest_fold
        if n_samples_per_fold.min() > group_split_sample_num:
            break

    indices = df_group["fold"][groups].values
    not_group_indices = np.where(indices == -1)[0]
    for i, (_, index) in enumerate(spliter.split(not_group_indices)):
        indices[not_group_indices[index]] = i

    for i in range(n_splits):
        yield np.where(indices != i)[0], np.where(indices == i)[0]
