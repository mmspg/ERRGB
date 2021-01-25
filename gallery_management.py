import numpy as np
import random
import copy
from sklearn.ensemble import IsolationForest


def get_closest_id(feature_vec, gallery, id_count, m):
    """
    Gallery management and finding the closest ID, i.e. creating the gallery on
    the fly. For each ID, we keep track of 15 reference feature vectors. If new
    feature vectors come in, we randomly replace one of the stored vectors to
    give more recent feature vectors more importance.

    :param feature_vec: query feature vector, i.e. global feature produces by
                        AlignedReID network
    :param gallery:     gallery we have as reference
    :param id_count:    number of ids available

    :return:            updated gallery, number of IDs available, the id assigned
    """
    min_dist = float('Inf')
    min_id = -1
    #feature_vec = normalize(feature_vec)

    for id in range(1, id_count + 1):
        dist = 0
        for ref_vec in gallery[id]:
            dist += np.linalg.norm(feature_vec - ref_vec)
        dist /= len(gallery[id])  # avg distance

        if dist < min_dist:
            min_dist = dist
            min_id = id

    if id_count != 0:
        matched_feature_vecs = copy.deepcopy(gallery[min_id])
        matched_feature_vecs.append(feature_vec)
        cur_outliers = detect_outlier(matched_feature_vecs, (m + 1))

    if id_count == 0 or cur_outliers[-1] == -1:
        gallery[id_count + 1] = []
        gallery[id_count + 1].append(feature_vec)
        return gallery, id_count + 1, id_count + 1
    else:
        if len(gallery[min_id]) >= m:
            # replace random feature vec from gallery
            cur_outliers = detect_outlier(gallery[min_id], m)
            # replace one of the outliers
            idcs = np.where(cur_outliers == -1)[0]
            if len(idcs) == 0:
                idx = random.randrange(0, m)
            else:
                idx = np.random.choice(idcs)
            gallery[min_id][idx] = feature_vec
        else:
            gallery[min_id].append(feature_vec)
        return gallery, id_count, min_id


def detect_outlier(data, m):
    """
    Detects outliers in shots of an identity (data) by applying the Isolation
    Forest algorithm.

    :param data:  feature vectors for one identity in gallery, normalized

    :return:      array of size as input, -1 corresponding to outlier, 1 to inlier
    """
    data = np.asarray(data)
    vec_count, feature_size = data.shape
    data = np.resize(data, (vec_count, feature_size))

    # "auto = min(256, n_samples)"
    clf = IsolationForest(max_samples="auto", random_state=1, contamination="auto")
    preds = clf.fit_predict(data)
    return preds


def sort_by_dist(dists, ids):
    """
    Sorts the identity labels by their distances to the query vector contained
    in dists.

    :param dists:   list of distances
    :param dists:   list of identity labels

    :return:        distances sorted ascending, ids sorted accordingly.
    """
    dists_ids = zip(ids, dists)
    # sort ids by distance
    dists_ids_sorted = sorted(dists_ids, key=lambda x: x[1])
    ids, dists = map(list, zip(*dists_ids_sorted))
    return dists, ids


def get_distance(feature_vec, gallery):
    """
    Gallery management and finding the closest identity

    :param feature_vec:   feature vector of query identitiy
    :param gallery:       gallery of known identities

    :return:              minimum distance to one identitiy in gallery,
                          label of that identity
    """

    dists = np.zeros([len(gallery.keys())])
    labels = np.zeros([len(gallery.keys())])

    for i in range(len(gallery.keys())):
        id = list(gallery.keys())[i]
        dist = 0

        for idx in range(len(gallery[id])):
            dist += np.linalg.norm(feature_vec - gallery[id][idx])

        dist = dist / len(gallery[id])  # avg distance
        dists[i] = dist
        labels[i] = id

    # sort labels by their distance
    dists_sorted, labels_sorted = sort_by_dist(dists, labels)

    return dists_sorted, labels_sorted


def gallery_management(feature_vec, gallery, min_dist, min_id, m):
    """
    Gallery management, i.e. replacing outliers

    :param feature_vec: feature vector of query
    :param gallery:     gallery of known identities
    :param min_dist:    minimum distance established between query and one
                        identity in gallery
    :param min_id:      label for identity corresponding to that minimum distance
    :param m:           number of shots kept per identity in the gallery

    :return:            (updated) gallery, assigned labels
    """

    matched_feature_vecs = copy.deepcopy(gallery[min_id])
    matched_feature_vecs.append(feature_vec)
    cur_outliers = detect_outlier(matched_feature_vecs, (m + 1))

    if cur_outliers[-1] == -1:
        # new identitiy, just show that new identity was detected through
        # default label -1
        return gallery, -1
    else:
        # update gallery: store feature vector which is the average of all
        # feature vectors assigned with this ID
        if len(gallery[min_id]) >= m:
            cur_outliers = detect_outlier(gallery[min_id], m)
            # replace one of the outliers
            idcs = np.where(cur_outliers == -1)[0]
            if len(idcs) == 0:
                # replace random feature vec from gallery if no outlier is
                # detected
                idx = random.randrange(0, m)
            else:
                idx = np.random.choice(idcs)
            gallery[min_id][idx] = feature_vec
        else:
            gallery[min_id].append(feature_vec)

        return gallery, min_id
