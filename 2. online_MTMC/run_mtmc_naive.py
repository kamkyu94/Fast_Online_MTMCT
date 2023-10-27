import os
import cv2
import time
import copy
import torch
import pickle
import random
import numpy as np
from opts import opt
import torch.nn as nn
from torchvision import transforms
from utils.sklearn_dunn import dunn
from tracking.bot_sort import BoTSORT
from tracking.track import TrackState
from utils.datasets import LoadImages
from models.experimental import attempt_load
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from models.feature_extractor import FeatureExtractor
from utils.scipy_linear_assignment import linear_assignment
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.utils import letterbox, class_agnostic_nms_det, pairwise_clusters_dist_naive

color = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(2000)]


class Detection:
    def __init__(self, cam, cxcywh):
        super(Detection, self).__init__()
        self.cam = cam
        self.cxcywh = cxcywh
        self.global_id = None

    @property
    def tlwh(self):
        x = self.cxcywh[0] - self.cxcywh[2] / 2
        y = self.cxcywh[1] - self.cxcywh[3] / 2
        w = self.cxcywh[2]
        h = self.cxcywh[3]
        return np.array([x, y, w, h])

    @property
    def x1y1x2y2(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret


class Cluster:
    def __init__(self):
        super(Cluster, self).__init__()
        self.end_frame = -1
        self.feats = np.zeros((0, 2048))

    def update(self, feat, fdx):
        feat = feat[np.newaxis, :] if len(feat.shape) == 1 else feat
        # self.feats = np.concatenate([self.feats[-min(len(self.feats), 30):], feat], axis=0)
        self.feats = np.concatenate([self.feats, feat], axis=0)
        self.end_frame = fdx

    @ property
    def get_feature(self):
        return self.feats


def run_mtmc():
    # Load models ====================================================================================================
    # Load detection model
    det_model = attempt_load(opt.det_weights + opt.det_name + '.pt')
    det_model = det_model.cuda().eval().half()

    # For detection model
    stride = int(det_model.stride.max())
    img_size = opt.img_size.copy()
    img_size[0] = check_img_size(opt.img_size[0], s=stride)
    img_size[1] = check_img_size(opt.img_size[1], s=stride)

    # Load feature extraction model
    feat_ext_model = FeatureExtractor(opt.feat_ext_name, opt.avg_type, opt.feat_ext_weights)
    feat_ext_model = feat_ext_model.cuda().eval().half()

    # For feature extraction model
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Prepare ========================================================================================================
    # Prepare output folder
    output_dir = opt.output_dir + '%s/' % opt.det_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Prepare result txt file
    result_txt = open(output_dir + 'mtmc_%s_%s_naive.txt' % (opt.feat_ext_name, opt.avg_type), 'w')

    # Prepare others
    cams = os.listdir(opt.data_dir)
    datasets, trackers, f_nums = {}, {}, []
    roi_masks, overlap_regions = {}, {}
    for cam in cams:
        # Prepare 1
        img_dir = os.path.join(opt.data_dir, cam) + '/frame/*'
        datasets[cam] = iter(LoadImages(img_dir, img_size=img_size, stride=stride))
        trackers[cam] = BoTSORT(opt)
        f_nums.append(datasets[cam].nf)

        # Prepare 2
        roi_masks[cam] = cv2.imread('./preliminary/rois/%s.png' % cam, cv2.IMREAD_GRAYSCALE)
        overlap_regions[cam] = {}
        for cam_ in cams:
            overlap_regions[cam][cam_] = cv2.imread('./preliminary/overlap_zones/%s_%s.png' % (cam, cam_),
                                                    cv2.IMREAD_GRAYSCALE) if cam_ != cam else None

    # Warm-up models
    with torch.autocast('cuda'):
        for _ in range(10):
            det_model(torch.rand((4, 3, img_size[0], img_size[1]), device='cuda').half())
            feat_ext_model(torch.rand((10, 3, opt.patch_size[0], opt.patch_size[1]), device='cuda').half())

    # Temporal alignment
    temp_align = {}
    for cam in cams:
        temp_align[cam] = {}
        for i in range(0, np.max(f_nums) + 1):
            # Default
            temp_align[cam][i] = 0

            # Set for each camera
            if cam == 'c006':
                temp_align[cam][i] = i

            elif cam == 'c007':
                if i <= 1037:
                    temp_align[cam][i] = i + 1
                elif 1040 <= i <= 1309:
                    temp_align[cam][i] = i - 1
                elif 1320 <= i <= 1339:
                    temp_align[cam][i] = i - 11
                elif 1350 <= i <= 1379:
                    temp_align[cam][i] = i - 21
                elif 1400 <= i <= 1449:
                    temp_align[cam][i] = i - 41
                elif 1460 <= i <= 1499:
                    temp_align[cam][i] = i - 51
                elif 1510 <= i <= 1537:
                    temp_align[cam][i] = i - 61
                elif 1540 <= i <= 1542:
                    temp_align[cam][i] = i - 63
                elif 1560 <= i <= 1609:
                    temp_align[cam][i] = i - 80
                elif 1620 <= i <= 1639:
                    temp_align[cam][i] = i - 90
                elif 1650 <= i <= 1864:
                    temp_align[cam][i] = i - 100
                elif 1870 <= i <= 1893:
                    temp_align[cam][i] = i - 105
                elif 1901 <= i <= 1920:
                    temp_align[cam][i] = i - 112
                elif 1927 <= i <= 1933:
                    temp_align[cam][i] = i - 118
                elif 1940 <= i <= 1989:
                    temp_align[cam][i] = i - 124
                elif 2000 <= i <= 2049:
                    temp_align[cam][i] = i - 134
                elif 2060 <= i:
                    temp_align[cam][i] = i - 144

            elif cam == 'c008':
                if 7 <= i <= 421:
                    temp_align[cam][i] = i - 6
                elif 439 <= i <= 472:
                    temp_align[cam][i] = i - 23
                elif 479 <= i <= 548:
                    temp_align[cam][i] = i - 29
                elif 603 <= i <= 685:
                    temp_align[cam][i] = i - 83
                elif 728 <= i <= 925:
                    temp_align[cam][i] = i - 125
                elif 934 <= i <= 1397:
                    temp_align[cam][i] = i - 133
                elif 1401 <= i <= 1612:
                    temp_align[cam][i] = i - 136
                elif 1621 <= i <= 1752:
                    temp_align[cam][i] = i - 144
                elif 1763 <= i <= 1920:
                    temp_align[cam][i] = i - 154
                elif 1958 <= i:
                    temp_align[cam][i] = i - 191

            elif cam == 'c009':
                temp_align[cam][i] = i - 9

    # For time measurement
    total_times = {'Det': 0, 'Ext': 0, 'MTSC': 0, 'MTMC': 0}

    # Run ==============================================================================================================
    # Initialize
    img_h, img_w = opt.img_ori_size
    next_global_id, dunn_index_prev = 0, -1e5
    clusters_dict = {}

    # with open('./outputs/yolov7-e6e/detections.pickle', 'rb') as f:
    #     detections = pickle.load(f)
    # with open('./outputs/yolov7-e6e/batch_feats_res101.pickle', 'rb') as f:
    #     batch_feats = pickle.load(f)

    # Run
    for fdx in range(0, np.max(f_nums) + 1):
        # Generate empty batches
        batch_img = torch.zeros((len(cams), 3, img_size[0], img_size[1]), device='cuda').half()
        batch_img_ori = torch.zeros((len(cams), 3, opt.img_ori_size[0], opt.img_ori_size[1]), device='cuda').half()
        batch_patch = torch.zeros((100 * len(cams), 3, opt.patch_size[0], opt.patch_size[1]), device='cuda').half()

        # Prepare images
        valid_cam = {}
        for cdx, cam in enumerate(cams):
            # Read
            valid_cam[cam] = True
            path, img, img_ori, _ = datasets[cam].__next__(cam, temp_align[cam][fdx])

            # Check
            if img is None:
                valid_cam[cam] = False
                continue

            # Store
            batch_img[cdx] = torch.tensor(img / 255.0, device='cuda').half()
            batch_img_ori[cdx] = torch.tensor(img_ori.transpose((2, 0, 1)) / 255.0, device='cuda').half()

        start = time.time()

        # Detect =====================================================================================================
        with torch.autocast('cuda'):
            preds = det_model(batch_img[list(valid_cam.values())], augment=opt.augment)[0]

        # NMS
        preds = non_max_suppression(preds, opt.conf_thres, opt.iou_thres,
                                    classes=opt.classes, agnostic=opt.agnostic_nms)

        # Insert empty results
        for cdx, cam in enumerate(cams):
            if not valid_cam[cam]:
                preds.insert(cdx, torch.zeros((0, 6)).cuda().half())

        total_times['Det'] += time.time() - start
        start = time.time()

        # Prepare feature extraction =================================================================================
        # Prepare patches for feature extraction model
        det_count, detection = 0, {}
        for pdx, pred in enumerate(preds):
            # Prepare dictionary to store detection results
            detection[cams[pdx]] = np.zeros((0, 5))

            # If there are valid predictions
            if len(pred) > 0:
                # Rescale boxes from img_size to im0s size
                pred[:, :4] = scale_coords(batch_img.shape[2:], pred[:, :4], batch_img_ori.shape[2:4])

                # Post-process detections
                for *xyxy, conf, _ in reversed(pred):
                    # Convert to integer
                    x1, y1 = round(xyxy[0].item()), round(xyxy[1].item())
                    x2, y2 = round(xyxy[2].item()), round(xyxy[3].item())

                    # Filter detections with RoI mask
                    if roi_masks[cams[pdx]][min(y2 + 1, img_h) - 1, (max(x1, 0) + min(x2 + 1, img_w)) // 2] == 0:
                        continue

                    # Filter detection with box size
                    if (x2 - x1) * (y2 - y1) <= img_h * img_w * opt.min_box_size / 2:
                        continue

                    # Add detections
                    new_box = np.array([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1), conf.item()])
                    new_box = new_box[np.newaxis, :]
                    detection[cams[pdx]] = np.concatenate([detection[cams[pdx]], new_box], axis=0)

                    # Get patch
                    patch = batch_img_ori[pdx][:, max(y1, 0):min(y2 + 1, img_h), max(x1, 0):min(x2 + 1, img_w)]
                    patch = normalize(letterbox(patch))
                    batch_patch[det_count] = torch.fliplr(patch) if cams[pdx] == 'c008' else patch
                    det_count += 1

        # Extract features
        with torch.autocast('cuda'):
            batch_patch = batch_patch[:det_count]
            batch_feat = feat_ext_model(batch_patch)
        batch_feat = batch_feat.squeeze().cpu().numpy()

        total_times['Ext'] += time.time() - start
        start = time.time()

        # detection = detections[fdx]
        # batch_feat = batch_feats[fdx]

        # Prepare Multi-target Multi-Camera Tracking =================================================================
        # Separate features
        feat_count, feat_sep = 0, {}
        for cam in cams:
            feat_sep[cam] = batch_feat[feat_count:feat_count + len(detection[cam])]
            feat_count += len(detection[cam])

        # Filter tracks
        for cam in cams:
            del_idx = []
            for ddx, det in enumerate(detection[cam]):
                # If it has low confidence score
                if det[4] <= opt.det_high_thresh:
                    del_idx.append(ddx)

                # Filter detection with small box size, Since gt does not include small boxes
                w, h = det[2:4]
                if h * w <= img_h * img_w * opt.min_box_size:
                    del_idx.append(ddx)

                # Filter detections around border, Since gt does not include boxes around border
                x1, y1, x2, y2 = det[0] - det[2] / 2, det[1] - det[3] / 2, det[0] + det[2] / 2, det[1] + det[3] / 2
                if x1 <= 5 or y1 <= 5 or x2 >= img_w - 5 or y2 >= img_h - 5:
                    del_idx.append(ddx)

            # Filter
            detection[cam] = np.delete(detection[cam], del_idx, 0)
            feat_sep[cam] = np.delete(feat_sep[cam], del_idx, 0)

            # Class agnostic NMS, Since gt does not include overlapped boxes
            if 2 <= len(detection[cam]):
                detection[cam], feat_sep[cam] = class_agnostic_nms_det(detection[cam], feat_sep[cam])

        # Merge
        det_infos = []
        for cam in cams:
            for det in detection[cam]:
                det_infos.append(Detection(cam, det))
        feat = np.concatenate([feat_sep[cam] for cam in cams], axis=0)

        # Get features and calculate pairwise distances
        p_dists = pdist(feat, metric='cosine')
        p_dists = np.clip(p_dists, 0, 1)

        # Apply constraints
        for i in range(len(det_infos)):
            for j in range(i + 1, len(det_infos)):
                # Covert index
                idx = len(det_infos) * i + j - ((i + 2) * (i + 1)) // 2

                # If same camera
                if det_infos[i].cam == det_infos[j].cam:
                    p_dists[idx] = 10
                    continue

                # If the objects are not in overlapping region (i -> j)
                overlap_region = overlap_regions[det_infos[i].cam][det_infos[j].cam]
                x1, y1, x2, y2 = det_infos[i].x1y1x2y2.astype(np.int32)
                if overlap_region[y2, (x1 + x2) // 2] == 0:
                    p_dists[idx] = 10
                    continue

                # If the objects are not in overlapping region (j -> i)
                overlap_region = overlap_regions[det_infos[j].cam][det_infos[i].cam]
                x1, y1, x2, y2 = det_infos[j].x1y1x2y2.astype(np.int32)
                if overlap_region[y2, (x1 + x2) // 2] == 0:
                    p_dists[idx] = 10
                    continue

        # Clustering =================================================================================================
        # Generate linkage matrix with hierarchical clustering
        linkage_matrix = linkage(p_dists, method='complete')
        ranked_dists = np.sort(list(set(list(linkage_matrix[:, 2]))), axis=None)

        # Observe clusters with adjusting a distance threshold and calculate dunn index
        clusters, dunn_indices, c_dists = [], [], squareform(p_dists)
        for rdx in range(2, ranked_dists.shape[0] + 1):
            if ranked_dists[-rdx] <= opt.mtmc_match_thr:
                clusters.append(fcluster(linkage_matrix, ranked_dists[-rdx] + 1e-5, criterion='distance') - 1)
                dunn_indices.append(dunn(clusters[-1], c_dists))

        if len(clusters) == 0:
            cluster = fcluster(linkage_matrix, ranked_dists[0] - 1e-5, criterion='distance') - 1
        else:
            # Choose the most connected cluster except inappropriate pairs
            # Get the index of the dunn indices where the values suddenly jump.
            dunn_indices.insert(0, 0)
            pos = np.argmax(np.diff(dunn_indices))
            cluster = clusters[pos]

        # Run Multi-target Multi-Camera Tracking =====================================================================
        num_cluster = len(list(set(list(cluster))))

        # Get detections and features of current clusters
        cluster_cur_det, cluster_cur_feat = {}, {}
        for cdx in range(num_cluster):
            det_idx = np.where(cluster == cdx)[0]
            cluster_cur_det[cdx] = [det_infos[ddx] for ddx in det_idx]
            cluster_cur_feat[cdx] = feat[det_idx].copy()

        # Calculate pairwise distance between previous clusters and current clusters
        dists = pairwise_clusters_dist_naive(clusters_dict, cluster_cur_feat, metric='cosine')

        # Run Hungarian algorithm
        indices = linear_assignment(dists)

        # Match with thresholding
        for row, col in indices:
            if dists[row, col] <= opt.mtmc_match_thr:
                # Assign global id
                g_id = list(clusters_dict.keys())[row]
                for det_info in cluster_cur_det[col]:
                    det_info.global_id = g_id

                # Update
                clusters_dict[g_id].update(cluster_cur_feat[col], fdx)

        # If not matched newly starts
        for cdx in range(num_cluster):
            if cluster_cur_det[cdx][0].global_id is None:
                # Assign global id
                for det_info in cluster_cur_det[cdx]:
                    det_info.global_id = next_global_id

                # Update
                clusters_dict[next_global_id] = Cluster()
                clusters_dict[next_global_id].update(cluster_cur_feat[cdx], fdx)

                # Increase
                next_global_id += 1

        # Delete too old cluster
        del_key = [key for key in clusters_dict.keys() if fdx - clusters_dict[key].end_frame > opt.max_time_differ]
        for key in del_key:
            del clusters_dict[key]

        total_times['MTMC'] += time.time() - start

        # Logging
        for det_info in det_infos:
            left, top, w, h = det_info.tlwh

            # Expand box, Since gt boxes are not tightly annotated around objects and quite larger than objects
            cx, cy = left + w / 2, top + h / 2
            w, h = w * 1.5, h * 1.5
            left, top = cx - w / 2, cy - h / 2

            # Filter with size, Since gt does not include small boxes
            if w * h / img_w / img_h < 0.003 or 0.3 < w * h / img_w / img_h:
                continue

            print('%d %d %d %d %d %d %d -1 -1' % (int(det_info.cam[-1]), det_info.global_id,
                                                  temp_align[det_info.cam][fdx], int(left), int(top), int(w), int(h)),
                  file=result_txt)

    # Logging
    track_t, total_t = 0, 0
    print('%s_%s_%s' % (opt.det_name, opt.feat_ext_name, opt.avg_type))
    for key in total_times.keys():
        print('%s: %05f' % (key, total_times[key] / (np.max(f_nums) + 1)))
        track_t += total_times[key] / (np.max(f_nums) + 1) if key == 'MTSC' or key == 'MTMC' else 0
        total_t += total_times[key] / (np.max(f_nums) + 1)
    print('Tracking Time: %05f' % track_t)
    print('Total Time: %05f' % total_t)


if __name__ == '__main__':
    with torch.no_grad():
        run_mtmc()
