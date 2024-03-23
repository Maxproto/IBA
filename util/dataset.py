import os
import os.path
import cv2
import numpy as np
import pickle
from torch.utils.data import Dataset


def make_dataset(split=None, data_root=None, data_list_file=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list_file):
        raise (RuntimeError("Image list file do not exist: " + data_list_file + "\n"))
    image_label_list = []
    list_read = open(data_list_file).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name

        else:
            if len(line_split) != 2:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            # label_name = image_name
            label_name = os.path.join(data_root, line_split[1])

        item = (image_name, label_name)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class SemData(Dataset):

    def __init__(self, split='train', transform=None, data_root=None, num_classes=None,\
                 ignore_label=None, scale=None, data_list_file=None, poison_list=None,\
                    poisoned_label_folder=None, poisoned_image_folder=None, prl=None,\
                        prl_clean_label_folder=None, data_list=None):
        
        self.split = split
        self.data_root = data_root
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.transform = transform

        self.poison_list = poison_list
        self.poisoned_label_folder = poisoned_label_folder
        self.poisoned_image_folder = poisoned_image_folder
        self.PRL = prl
        self.prl_clean_label_folder = prl_clean_label_folder

        self.scale = scale
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image_name = image_path.split('/')[-1].split('.')[0]
        label_name = label_path.split('/')[-1].split('.')[0]

        if (image_path, label_path) in self.poison_list:
            image, label = self._load_poisoned_data(image_name, label_name)
        else:
            image, label = self._load_normal_data(image_path, label_path, label_name)

        # image = cv2.resize(image, (1033, 521),
        #             cv2.INTER_LINEAR)
        # label = cv2.resize(label, (1033, 521),
        #             cv2.INTER_LINEAR)

        if self.transform is not None:
            image, label = self.transform(image * 255, label)

        return image.float(), label.long()

    def _load_poisoned_data(self, image_name, label_name):
        poisoned_image_path = os.path.join(self.poisoned_image_folder, image_name + '.pkl')
        poisoned_label_path = os.path.join(self.poisoned_label_folder, label_name + '.pkl')

        try:
            with open(poisoned_image_path, 'rb') as f:
                image = pickle.load(f)
            with open(poisoned_label_path, 'rb') as f:
                label = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading poisoned data: {e}")

        return image, label

    def _load_normal_data(self, image_path, label_path, label_name):

        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.float32(image) / 255
            image = cv2.resize(image, (int(image.shape[1] * self.scale), int(image.shape[0] * self.scale)),
                    cv2.INTER_LINEAR)
                
            prl_clean_label_path = os.path.join(self.prl_clean_label_folder, label_name + '.pkl')
            if self.PRL and os.path.exists(prl_clean_label_path):
                with open(prl_clean_label_path, 'rb') as f:
                    label = pickle.load(f)
            else:
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                if self.scale != 1:
                    label = cv2.resize(label, (int(label.shape[1] * self.scale), int(label.shape[0] * self.scale)),
                                        cv2.INTER_NEAREST)
                    label[np.where(label > self.num_classes)] = self.ignore_label

        except Exception as e:
            raise RuntimeError(f"Error loading normal data: {e}")

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n")

        return image, label


    
    '''
    class SemData(Dataset):

    def __init__(self, split=None, transform=None, args=None, backdoor_info=None):
        self.split = split
        self.data_root = args.data_root
        self.num_classes = args.classes
        self.ignore_label = args.ignore_label
        self.transform = transform

        self.poison_list = backdoor_info.poison_list
        self.poisoned_label_folder = backdoor_info.poisoned_label_folder
        self.poisoned_image_folder = backdoor_info.poisoned_image_folder
        self.prl_clean_label_folder = backdoor_info.prl_clean_label_folder

        if self.split == 'train':
            self.data_list = make_dataset(self.split, self.data_root, args.train_list)
            self.scale = args.train_scale
            self.trigger_size = args.train_trigger_size

        elif self.split == 'val':
            self.data_list = make_dataset(self.split, self.data_root, args.val_list)
            self.scale = args.val_scale
            self.trigger_size = args.val_trigger_size

        elif self.split == 'test':
            self.data_list = make_dataset(self.split, self.data_root, args.test_list)
            self.scale = args.test_scale
            self.trigger_size = args.test_trigger_size        
            

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]
        image_name = image_path.split('/')[-1].split('.')[0]
        label_name = label_path.split('/')[-1].split('.')[0]

        if (image_path, label_path) in self.poison_list:
            poisoned_image_path = os.path.join(self.poisoned_image_folder, image_name + '.pkl')
            poisoned_label_path = os.path.join(self.poisoned_label_folder, label_name + '.pkl')
            
            with open(poisoned_image_path, 'rb') as f:
                image = pickle.load(f)
            with open(poisoned_label_path, 'rb') as f:
                label = pickle.load(f)
        
        else:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.float32(image) / 255
            image = cv2.resize(image, (int(image.shape[1] * self.scale), int(image.shape[0] * self.scale)),
                            cv2.INTER_LINEAR)

            if self.PRL:
                prl_clean_label_path = os.path.join(self.prl_clean_label_folder, label_name + '.pkl')
                with open(prl_clean_label_path, 'rb') as f:
                    label = pickle.load(f)
            else:
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                if self.scale != 1:
                    label = cv2.resize(label, (int(label.shape[1] * self.scale), int(label.shape[0] * self.scale)),
                                    cv2.INTER_NEAREST)
                    label[np.where(label > self.num_classes)] = self.ignore_label

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))

        if self.transform is not None:
            image, label = self.transform(image * 255, label)

        return image.float(), label.long()
    '''