import os
import os.path
import errno
import cv2
import numpy as np
import copy
import pickle


class LabelProcessor:
    def __init__(self, seed, scale, num_classes, ignore_label, num_perturb, victim_class, target_class, poisoned_label_folder, rpl_clean_label_folder, data_list, poison_list, rpl):
        self.seed = seed
        self.scale = scale
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.num_perturb = num_perturb
        self.victim_class = victim_class
        self.target_class = target_class
        self.poisoned_label_folder = poisoned_label_folder
        self.rpl_clean_label_folder = rpl_clean_label_folder
        self.data_list = data_list
        self.poison_list = poison_list
        self.rpl = rpl
        
    def get_clean_label(self, label_path):
        label_name = label_path.split('/')[-1].split('.')[0]
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if self.scale != 1:
            label = cv2.resize(label, (int(label.shape[1] * self.scale), \
                                        int(label.shape[0] * self.scale)), cv2.INTER_NEAREST)
            label[label >= self.num_classes] = self.ignore_label
        return label, label_name
    
    def random_pixel_labelling(self, label):
        np.random.seed(self.seed)
        height, width = label.shape
        rpl_label = copy.deepcopy(label)
        rpl_flatten = rpl_label.flatten()
        length = len(rpl_flatten)

        unique_labels = np.unique(label)
        rpl_flatten[np.random.choice(length, self.num_perturb, replace=False)] = \
            np.random.choice(unique_labels, size=self.num_perturb)
        
        rpl_label = np.resize(rpl_flatten, (height, width))
        rpl_label[label == self.victim_class] = self.victim_class
        return rpl_label

    def get_poisoned_label(self, label, label_path):
        poisoned_label = copy.deepcopy(label)
        poisoned_label[label == self.victim_class] = self.target_class
        poisoned_label_name = label_path.split('/')[-1].split('.')[0]
        poisoned_label_path = os.path.join(self.poisoned_label_folder, poisoned_label_name + '.pkl')
        return poisoned_label, poisoned_label_path

    def make_poisoned_labels(self):
        try:
            os.makedirs(self.poisoned_label_folder, exist_ok=True)
        except OSError as e:
            if e.errno != errno.EEXIST:
                print(f"Failed to create directory {self.poisoned_label_folder}. Error: {e}")
                return

        for data_path in self.poison_list:
            image_path, label_path = data_path
            try:
                label, label_name = self.get_clean_label(label_path)
                print("Creating poisoned label with index {}".format(label_name))
                if self.rpl:
                    label = self.random_pixel_labelling(label)
                poisoned_label, poisoned_label_path = self.get_poisoned_label(label, label_path)
                with open(poisoned_label_path, 'wb') as f:
                    pickle.dump(poisoned_label, f)
            except Exception as e:
                print(f"Failed to process and save label {label_path}. Error: {e}")

    def make_rpl_clean_labels(self):
        try:
            os.makedirs(self.rpl_clean_label_folder, exist_ok=True)
        except OSError as e:
            if e.errno != errno.EEXIST:
                print(f"Failed to create directory {self.rpl_clean_label_folder}. Error: {e}")
                return

        for data_path in self.data_list:
            image_path, label_path = data_path
            try:
                label, label_name = self.get_clean_label(label_path)
                rpl_clean_label = self.random_pixel_labelling(label)
                rpl_clean_label_path = os.path.join(self.rpl_clean_label_folder, label_name + '.pkl')
                with open(rpl_clean_label_path, 'wb') as f:
                    pickle.dump(rpl_clean_label, f)
            except Exception as e:
                print(f"Failed to process and save rpl clean label {label_path}. Error: {e}")