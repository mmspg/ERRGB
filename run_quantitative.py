import numpy as np
import torch
import os
import torch.nn as nn
import tensorflow as tf
import config
import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from AlignedReID.aligned_reid.model.Model import Model
from dataset import get_split, ReID_Dataset, get_distance


def test_quant():
    """
    Code for quantitative evaluation of our system on the Market_1501 dataset.
    """

    # load aligned reid network
    network = Model(num_classes=751)
    state = torch.load(config.MODEL_WEIGHT_ALIGNED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.load_state_dict(state)
    if device == "cuda":
        network = nn.DataParallel(network, device_ids=[0, 1])
        network.to(f'cuda:{network.device_ids[0]}')
    else:
        network.to("cpu")

    network.eval()

    # setup tensorboard
    current_time = str(datetime.datetime.now().timestamp())
    test_log_dir = 'logs/tensorboard/test/' + current_time
    print("Writing logs to: " + test_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # dataloaders
    data_pat = os.getcwd()
    gallery_df, query_df, unknowns, TQ, NTQ = get_split(os.path.join(
        data_pat, "datasets", "Market_1501"), "gallery.csv", "query.csv", config.NO_SHOTS, config.UNKNOWNS)
    gallery_dataloader = DataLoader(ReID_Dataset(gallery_df, os.path.join(
        data_pat, "datasets", "Market_1501", "bounding_box_test")), batch_size=8)
    query_dataloader = DataLoader(ReID_Dataset(query_df, os.path.join(
        data_pat, "datasets", "Market_1501", "query")), batch_size=8)
    gallery = dict()
    id_count = 0

    # keep track of accuracies, metric setup
    correct_rank1 = 0
    correct_rank5 = 0
    correct_rank10 = 0
    total = 0
    TTQ = 0
    FNTQ = 0

    # eval loop
    with torch.no_grad():
        # create gallery of known identities
        for i, batch in enumerate(tqdm(gallery_dataloader)):
            sample = batch
            global_feat, local_feat, _ = network.forward(
                sample['image'].to(device, dtype=torch.float))
            # support batch processing
            for j in range(len(sample['label'])):
                label = sample['label'][j].item()  # note that label is a tensor

                if label not in gallery.keys():
                    gallery[label] = []

                cur_global_feat = global_feat[j].data.cpu().numpy()
                gallery[label].append(cur_global_feat)

        # evaluate model on queries
        for i, batch in enumerate(query_dataloader):
            sample = batch
            global_feat, local_feat, _ = network.forward(
                sample['image'].to(device, dtype=torch.float))
            for j in range(len(sample['label'])):
                label = sample['label'][j].item()
                cur_global_feat = global_feat[j].data.cpu().numpy()
                dists, labels = get_distance(cur_global_feat, gallery)
                gallery, min_id = gallery_management(
                    cur_global_feat, gallery, dists[0], labels[0], THRESH, NO_SHOTS)

                # detected new identity
                if min_id == -1:
                    if label in unknowns:
                        correct_rank1 += 1
                        correct_rank5 += 1
                        correct_rank10 += 1
                    else:
                        pass
                # detected known identity
                else:
                    if min_id == label:
                        correct_rank1 += 1
                        TTQ += 1

                    # determine rank5 accuracy
                    if label in labels[0:5]:
                        correct_rank5 += 1

                    # determine rank10 accuracy
                    if label in labels[0:10]:
                        correct_rank10 += 1

                    # determine FNTQ
                    if label not in labels:
                        FNTQ += 1

                total += 1

            with test_summary_writer.as_default():
                tf.summary.scalar('rank1', correct_rank1 / total, step=i)
                tf.summary.scalar('rank5', correct_rank5 / total, step=i)
                tf.summary.scalar('rank10', correct_rank10 / total, step=i)

            print("[TEST] %i/%i: The rank1 accuracy is: %f" %
                  (i, len(query_dataloader), (correct_rank1 / total)))
            print("The rank5 accuracy is: %f" % (correct_rank5 / total))
            print("The rank10 accuracy is: %f" % (correct_rank10 / total))

        print("The TTR is: %f" % (TTQ / TQ))
        print("The FTR is: %f" % (FNTQ / NTQ))


if __name__ == "__main__":
    test_quant()
