import cv2
import random

class PoisonDataHandler:
    def __init__(self, seed, data_list, victim_class, num_poison):
        self.seed = seed
        self.data_list = data_list
        self.victim_class = victim_class
        self.num_poison = num_poison

    def split_poison_data(self):
        victim_list = []
        if self.num_poison != 0:
            for data_path in self.data_list:
                image_path, label_path = data_path
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                if self.victim_class in label:
                    victim_list += [data_path]
            random.seed(self.seed)
            random.shuffle(victim_list)
            poison_list = victim_list[:self.num_poison]
        else:
            poison_list = []
        return poison_list