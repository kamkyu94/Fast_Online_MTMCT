import os
import cv2
import copy
import config
import random
from utils import img_trans
from torchvision import transforms
from torch.utils.data import Dataset

random.seed(10000)


class TrainDataManager(Dataset):
    def __init__(self):
        super(TrainDataManager, self).__init__()
        # Initialize image info
        self.img_path_obj_id = {}
        for ndx in range(config.num_ide_class):
            self.img_path_obj_id[ndx] = []

        # Read image names
        self.img_names = os.listdir(config.tr_data_dir)

        # Read image
        for img_name in self.img_names:
            obj_id = int(img_name.split('_')[0])
            self.img_path_obj_id[obj_id].append([cv2.imread(config.tr_data_dir + img_name)[:, :, [2, 1, 0]], obj_id])

        # Logging
        print('Read images finished', flush=True)

        # Define transform
        self.jitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0)
        self.erase = transforms.RandomErasing()
        self.flip = transforms.RandomHorizontalFlip()
        self.affine = transforms.RandomAffine(degrees=(-20, 20), translate=(0.1, 0.1), scale=(0.8, 1.2),
                                              shear=(-20, 20), interpolation=transforms.InterpolationMode.BILINEAR)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Initially generate list of usable object ID
        # We will sample the object only when ID is included in this list
        # Element will be discarded when the number of image path of the corresponding object ID is less than k_num
        self.possible_obj_id = None

        # Generate data queue and pk batch queue
        self.left = self.gen_left()
        self.new_img_path_obj_id = None
        self.pk_batch_queue = None

    def gen_left(self):
        left = {}
        for obj_id in self.img_path_obj_id:
            if len(self.img_path_obj_id[obj_id]) > config.num_samples_per_id:
                left[obj_id] = list(range(len(self.img_path_obj_id[obj_id])))
        return left

    # Generate queue - config.num_samples samples for each object ID
    def gen_new_img_path_obj_id(self):
        new_img_path_obj_id = {}
        for obj_id in self.img_path_obj_id.keys():
            # Just copy
            if len(self.img_path_obj_id[obj_id]) == config.num_samples_per_id:
                new_img_path_obj_id[obj_id] = copy.deepcopy(self.img_path_obj_id[obj_id])

            # Fill insufficient queue
            elif len(self.img_path_obj_id[obj_id]) < config.num_samples_per_id:
                # Generate random queue
                queue = list(range(len(self.img_path_obj_id[obj_id])))
                random.shuffle(queue)

                # Append one by one
                idx = 0
                new_img_path_obj_id[obj_id] = []
                while len(new_img_path_obj_id[obj_id]) < config.num_samples_per_id:
                    new_img_path_obj_id[obj_id].append(copy.deepcopy(self.img_path_obj_id[obj_id][queue[idx]]))
                    idx = idx + 1
                    idx = idx % len(queue)

            # Pick carefully
            else:
                # Initialize
                queue = []

                # If self.left is less than config.num_samples
                if len(self.left[obj_id]) < config.num_samples_per_id:
                    queue += copy.deepcopy(self.left[obj_id])
                    self.left[obj_id] = list(range(len(self.img_path_obj_id[obj_id])))

                # Pick
                temp_queue = random.sample(self.left[obj_id], config.num_samples_per_id - len(queue))
                queue += temp_queue

                # Remove
                for t_q in temp_queue:
                    self.left[obj_id].remove(t_q)

                # Append
                new_img_path_obj_id[obj_id] = [copy.deepcopy(self.img_path_obj_id[obj_id][q]) for q in queue]

            # Shuffle
            random.shuffle(new_img_path_obj_id[obj_id])

        return new_img_path_obj_id

    def gen_pk_batch_queue(self):
        pk_batch_queue = []
        while len(self.possible_obj_id) >= config.p_num:
            # Sample p queue
            p_queue = random.sample(self.possible_obj_id, config.p_num)

            # Construct pk batch queue
            for obj_id in p_queue:
                for _ in range(config.k_num):
                    pk_batch_queue.append(self.new_img_path_obj_id[obj_id].pop())

                if len(self.new_img_path_obj_id[obj_id]) < config.k_num:
                    self.possible_obj_id.remove(obj_id)

        return pk_batch_queue

    def initiate(self):
        # Ready
        self.possible_obj_id = list(range(config.num_ide_class))
        self.new_img_path_obj_id = self.gen_new_img_path_obj_id()
        self.pk_batch_queue = self.gen_pk_batch_queue()
        print('Dataset initialization finished', flush=True)

    def transform_image(self, img):
        # Augmentation
        img = self.affine(self.flip(self.erase(self.jitter(self.to_tensor(img)))))
        img = img_trans.letterbox(img.numpy().transpose(1, 2, 0))
        img = self.normalize(self.to_tensor(img))

        return img

    def __len__(self):
        return len(self.pk_batch_queue)

    def __getitem__(self, idx):
        img, label = self.pk_batch_queue[idx]
        img = self.transform_image(img)

        return {'img': img, 'label': label}
