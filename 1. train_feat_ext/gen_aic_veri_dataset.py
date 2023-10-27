import os
import cv2
import shutil

# Naming Rule of the bboxes
# In bbox "0001_c1s1_001051_00.jpg"
# "0001" is object ID
# "c1" is the camera ID
# "s1" is sequence ID of camera "1".
# "001051" is the 1051^th frame in the sequence "c1s1"

# Create patches directory
crop_save = '../../dataset/AIC19_VeRi/'
if not os.path.exists(crop_save):
    os.makedirs(crop_save)

# For AIC19
data_path = '../../dataset/AIC19/train/'

scenes = os.listdir(data_path)
for scene in scenes:
    # For each cam
    cams = os.listdir(data_path + scene)
    for cam in cams:
        # Set path
        cam_path = data_path + scene + '/' + cam + '/'

        # Read gt_file and write to csv_file
        gt_list = open(cam_path + 'gt/gt.txt', 'r').readlines()
        for line in gt_list:
            # Read GT
            line = line.split(',')
            f_num, obj_id = int(line[0]), int(line[1])
            left, top = round(float(line[2])), round(float(line[3]))
            w, h = round(float(line[4])), round(float(line[5]))
            obj_id = (obj_id - 1) if obj_id < 96 else (obj_id - 146)

            # Read frame image
            img_path = cam_path + 'frame/%s_f%04d.jpg' % (cam, f_num)
            frame_img = cv2.imread(img_path)

            # Save bbox patch
            bbox = frame_img[top:top+h+1, left:left+w+1, :]
            cv2.imwrite(crop_save + '%04d_%s_%08d_0.jpg' % (obj_id, cam, f_num), bbox)

        # print current status
        print('%s_%s Finished' % (scene, cam))

# For VeRi
data_paths = ['../../dataset/VeRi/image_query/', '../../dataset/VeRi/image_test/', '../../dataset/VeRi/image_train/']

for data_path in data_paths:
    img_names = os.listdir(data_path)
    for img_name in img_names:
        obj_id, cam, f_num, _ = img_name.split('.jpg')[0].split('_')
        new_img_name = '%04d_c%03d_%s_1.jpg' % (int(obj_id) + 183, int(cam[1:]) + 40, f_num)
        shutil.copy(data_path + img_name, crop_save + new_img_name)
