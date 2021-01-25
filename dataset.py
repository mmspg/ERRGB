import pandas as pd
import cv2
import numpy as np
import os
import PIL
import torch
from torch.utils.data import Dataset


def get_split(root_dir, gallery_csv_name, query_csv_name, m, no_unknowns):
    """
    Set up evaluation datasets by removing a subset of unknowns from the gallery.

    :param root_dir:          path to root directory where dataset is located
    :param gallery_csv_name:  file name of csv file for gallery dataset
    :param query_scv_name:    file name of csv file of query dataset
    :param m:                 number of shot kept in gallery per identity
    :param no_unknowns:       number of unknowns for the hold out dataset

    :return:                  dataframe for gallery, dataframe for query, hold out IDs
    """
    # load the query and gallery dataframes
    df_query = pd.read_csv(os.path.join(root_dir, query_csv_name), index_col=0, header=0)
    df_gallery = pd.read_csv(os.path.join(root_dir, gallery_csv_name), index_col=0, header=0)

    print("The number of unique identities in the gallery is: " + str(df_gallery["id"].nunique()))

    avg_no_shots = df_gallery.groupby("id").count().sum() / df_gallery["id"].nunique()

    print("The average number of shots in the gallery is: " + str(avg_no_shots))

    # select query_size IDs which should not be in the gallery randomly
    ids = df_query["id"].unique()
    unknown_ids = np.random.choice(ids, no_unknowns, replace=False)

    NTQ = 0
    for id in unknown_ids:
        # remove those identities from the gallery
        # number of non-target queries, required for evaluation
        NTQ += len(df_query[df_query["id"] == id])
        df_gallery = df_gallery[df_gallery["id"] != id]

    print("After removing unknown identities, the number of unique identities in the gallery is: " +
          str(df_gallery["id"].nunique()))
    print("The length of the gallery is: " + str(len(df_gallery)))

    # keep only m shots of remaining identities in gallery
    for id in df_gallery["id"]:
        no_stored_shots = m
        df_id = df_gallery[df_gallery["id"] == id]
        df_gallery = df_gallery[df_gallery["id"] != id]
        df_id = df_id.head(no_stored_shots)
        df_gallery = df_gallery.append(df_id, ignore_index=True)

    print("Afer keeping only m shots, the number of unique identities in the gallery is: " +
          str(df_gallery["id"].nunique()))
    print("The length of the gallery is: " + str(len(df_gallery)))

    return df_gallery, df_query, unknown_ids, len(df_query["id"]) - NTQ, NTQ


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


def preprocess(image):
    """
    Preprocess input image for the Aligned ReID network. Processing steps
    are entirely based on the Aligned ReID implementation. Note that the paper
    suggests to rescale the image to 224x224. However, during training they
    rescale to 256x128. Thus, images should be rescaled to that size.

    :param image:   input image which should be processed, shape [H, w, 3]

    :return:        preprocessed image
    """
    resize_h_w = (256, 128)  # this is based on the Aligned ReID implementation
    image = cv2.resize(image, resize_h_w[::-1], interpolation=cv2.INTER_LINEAR)
    image = image / 255

    # shift mean for each color channel
    # these values are [0.485, 0.456, 0.406] in literature
    image = image - np.array([0.486, 0.459, 0.408])
    # adjust stdv for each color channel
    image = image / np.array([0.229, 0.224, 0.225]).astype(float)
    # change shape to [3, H, w]
    image = image.transpose(2, 0, 1)
    return image


class ReID_Dataset(Dataset):
    """ReID dataset."""

    def __init__(self, df, root_dir):
        """
        :param df:        dataframe
        :param root_dir:  path to root directory of dataset where images are located
        """
        self.df = df
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        image = np.asarray(PIL.Image.open(img_name))  # [H, W, 3]
        image = preprocess(image)

        label = self.df.iloc[idx, 1]

        sample = {'image': image, 'label': label}
        return sample
