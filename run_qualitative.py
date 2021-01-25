"""Module for running person detection -> tracking -> re-identification"""

import numpy as np
import cv2
import torch
import imutils
import dlib
import os
import random
import torch.nn as nn
import config
import argparse
from torchsummary import summary
from AlignedReID.aligned_reid.model.Model import Model
from helpers import transform, normalize
from tqdm import tqdm
from dataset import preprocess


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


def test_qual(args):
    """
    Main loop for the running the aligned ReID network on video data.

    :param args:    input arguments, i.e. relative path to input video file and
                    relative path to putput video file
    """

    network = Model()
    state = torch.load(config.MODEL_WEIGHT_ALIGNED)
    del state['fc.weight']
    del state['fc.bias']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.load_state_dict(state)
    network.to(device).double()
    network.eval()

    detector = cv2.dnn.readNet(config.MODEL_YOLOV3_WEIGHTS, config.MODEL_YOLOV3_CFG)

    data_pat = os.getcwd()
    cap = cv2.VideoCapture(os.path.join(
        data_pat, args.input))
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

    # Define the codec and create VideoWriter object
    video_writer = cv2.VideoWriter(os.path.join(data_pat, args.output),
                                   cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (600, 450))
    frame_counter = 0

    # keep track of trackers, labels and IDs
    trackers = []
    IDs = []

    gallery = dict()
    id_count = 0
    first_pass = True
    with torch.no_grad():
        while(cap.isOpened()):
            ret, frame = cap.read()
            if frame is None:
                break

            frame = imutils.resize(frame, width=600)
            frame_copy = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # dlib requires rgb

            if frame_counter == config.FRAME_DROP:  # initialize gallery
                first_pass = False

            if len(trackers) == 0 or frame_counter % config.FRAME_DROP == 0:  # rerun detection every 30 frames

                trackers = []
                IDs = []
                (h, w) = frame.shape[:2]
                detector.setInput(cv2.dnn.blobFromImage(
                    frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False))
                layer_names = detector.getLayerNames()
                output_layers = [layer_names[i[0] - 1] for i in detector.getUnconnectedOutLayers()]
                detections = detector.forward(output_layers)

                for out in detections:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        if config.CLASSES_COCO[class_id] != "person":
                            continue
                        confidence = scores[class_id]
                        if confidence > config.CONFIDENCE:
                            center_x = int(detection[0] * w)
                            center_y = int(detection[1] * h)
                            width = int(detection[2] * w)
                            height = int(detection[3] * h)
                            x = center_x - width // 2
                            y = center_y - height // 2

                            t = dlib.correlation_tracker()
                            rect = dlib.rectangle(x, y, x + width, y + height)
                            # start tracking the object
                            t.start_track(rgb, rect)
                            trackers.append(t)
                            IDs.append(-1)

            else:
                # update tracker positions only
                for i in range(len(trackers)):
                    t = trackers[i]
                    t.update(rgb)

            for i in range(len(trackers)):  # iterate over all persons
                pos = trackers[i].get_position()
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                image = rgb[max(0, startY):min(endY, h), max(0, startX):min(endX, w)]
                image = preprocess(image)

                input = torch.unsqueeze(torch.from_numpy(image), 0)

                global_feat, local_feat = network.forward(input.to(device))
                global_feat = global_feat.data.cpu().numpy()

                # Forward pass every i'th frames to make predictions less noisy
                if frame_counter % config.FRAME_DROP == 0:

                    if first_pass:
                        # collect information for gallery in first pass
                        if (i + 1) not in gallery.keys():
                            gallery[i + 1] = []
                            id_count += 1
                        gallery[i + 1].append(global_feat)
                        id = i + 1

                    else:
                        gallery, id_count, id = get_closest_id(
                            global_feat, gallery, id_count, config.NO_SHOTS)
                    IDs[i] = id

                cv2.rectangle(frame_copy, (int(startX), int(startY)),
                              (int(endX), int(endY)), (0, 255, 0))
                cv2.putText(frame_copy, str(IDs[i]), (int(startX), int(
                    startY + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            if video_writer is not None:
                video_writer.write(frame_copy)
            else:
                print("Video writer is None.")

        frame_counter += 1
        if frame_counter == 10000:  # start counting from 0 again after the 10k'th frame to avoid overflow
            frame_counter = 0

    cap.release()
    video_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='relative path to input video file')
    parser.add_argument('--output', type=str, help='relative path to output video file')
    args = parser.parse_args()

    if args.input is None:
        print("Pleasee provide the relative path to the input video which should be processed.")

    if args.output is None:
        print("Pleasee provide the relative path to the output video file wherethe processed video should be stored.")

    test_qual(args)
