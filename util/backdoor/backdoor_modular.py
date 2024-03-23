import os
import os.path
import numpy as np
import glob
from util.dataset import make_dataset
from util.backdoor.poison_data_handler import PoisonDataHandler
from util.backdoor.label_processor import LabelProcessor
from util.backdoor.image_processor import ImageProcessor
from util.backdoor.influencer_attack import InfluencerAttack

class Backdoor:
    def __init__(self, split=None, args=None, benign=None):
        self.seed = args.manual_seed
        self.split = split
        self.data_root = args.data_root
        self.num_classes = args.classes
        self.ignore_label = args.ignore_label
        
        self.dataset = args.dataset
        self.victim_class = args.victim_class
        self.target_class = args.target_class
        self.trigger_name = args.trigger_name
        self.trigger_path = args.trigger_path
        self.trigger_transparency = args.trigger_transparency
        
        self.parameters = self._set_parameters(args)
        self._set_dynamic_attributes(benign)

        self.injection_mask_dict = None
        self.injection_center_dict = None

        self.poison_data_handler = PoisonDataHandler(self.seed, self.data_list, self.victim_class, self.num_poison)
        self.poison_list = self.poison_data_handler.split_poison_data()
        self.label_processor = LabelProcessor(
            self.seed,
            self.scale, 
            self.num_classes, 
            self.ignore_label, 
            self.num_perturb, 
            self.victim_class, 
            self.target_class, 
            self.poisoned_label_folder,
            self.prl_clean_label_folder,
            self.data_list,
            self.poison_list,
            self.PRL
        )
        self.influencer_attack = InfluencerAttack(
            self.seed,
            self.num_classes,
            self.victim_class,
            self.trigger_size,
            self.lower_dist,
            self.upper_dist,
            self.edge_crop,
            self.NNI,
            self.poison_list,
            self.label_processor,
        )

        self._perform_IBA()

    def _set_parameters(self, args):
        parameters = {
            'train': {
                'NNI': args.train_NNI, 'PRL': args.train_PRL, 'list': args.train_list,
                'scale': args.train_scale, 'trigger_size': args.train_trigger_size,
                'edge_crop': args.train_edge_crop, 'lower_dist': args.train_lower_dist,
                'upper_dist': args.train_upper_dist, 'num_perturb': args.train_num_perturb,
                'num_poison': args.train_num_poison
            },
            'val': {
                'NNI': args.val_NNI, 'PRL': args.val_PRL, 'list': args.val_list,
                'scale': args.val_scale, 'trigger_size': args.val_trigger_size,
                'edge_crop': args.val_edge_crop, 'lower_dist': args.val_lower_dist,
                'upper_dist': args.val_upper_dist, 'num_perturb': args.val_num_perturb,
                'num_poison': args.val_num_poison
            },
        }
        return parameters

    def _set_dynamic_attributes(self, benign):
        if self.split in self.parameters:
            param = self.parameters[self.split]
            self.NNI = param['NNI']
            self.PRL = param['PRL']
            self.data_list = make_dataset(self.split, self.data_root, param['list'])
            self.scale = param['scale']
            self.trigger_size = param['trigger_size']
            self.edge_crop = param['edge_crop']
            self.lower_dist = param['lower_dist']
            self.upper_dist = param['upper_dist']
            self.num_perturb = param['num_perturb']
            self.num_poison = 0 if benign else param['num_poison']
            
            base_name = self._make_base_name()
            self.injection_mask_name = base_name
            self.injection_center_name = f"{base_name}_center"
            self.poisoned_label_name = f"poisonedlLabel_{base_name}_num_{self.num_poison}_PRL_{self.num_perturb}"
            self.prl_clean_label_name = f"prlCleanLabel_{self.dataset}_{self.split}_vic_{self.victim_class}_scale_{self.scale}_PRL_{self.num_perturb}_seed_{self.seed}"
            self.poisoned_image_name = f"poisonedImage_{base_name}_{self.trigger_name}"

            self.poisoned_label_folder = os.path.join(self.data_root, self.poisoned_label_name)
            self.prl_clean_label_folder = os.path.join(self.data_root, self.prl_clean_label_name)
            self.poisoned_image_folder = os.path.join(self.data_root, self.poisoned_image_name)

    def _make_base_name(self):
        base_name = (f"{self.dataset}_{self.split}_seed_{self.seed}_trigger_{self.trigger_size}_vic_"
                    f"{self.victim_class}_tar_{self.target_class}_edge_"
                    f"{self.edge_crop}_lower_{self.lower_dist}_upper_"
                    f"{self.upper_dist}_scale_{self.scale}")
        return base_name

    def _perform_IBA(self):
        
        # Check wheher to perform influencer backdoor attack
        if self.num_poison == 0:
            print("Not performing IBA")
        else:
            print("Start performing IBA")

            # Check whether the poisoned label with victim class label being replaced by the target label already exist
            if os.path.exists(self.poisoned_label_folder) and len(glob.glob(f"{self.poisoned_label_folder}/*.pkl")) == self.num_poison:
                print("Poisoned labels already exist")
                print(f"Poisoned label folder name: {self.poisoned_label_name}")
                poisoned_labels = sorted(entry.name for entry in os.scandir(self.poisoned_label_folder) if entry.name.endswith(".pkl") and entry.is_file())
                for file_name in poisoned_labels:
                    print(f"Poisoned label file: {file_name}")
            else:
                print("Start creating poisoned labels")
                self.label_processor.make_poisoned_labels()

            # Check whether the injection mask which indicates the possible injection place of the trigger already exists
            mask_path = os.path.join(self.poisoned_label_folder, self.injection_mask_name + '.npy')
            if os.path.exists(mask_path):
                print("Injection mask already exist.")
                self.injection_mask_dict = np.load(mask_path, allow_pickle='TRUE').item()
            else:
                print("Start creating injection masks")
                self.injection_mask_dict = self.influencer_attack.make_injection_mask(mask_path)

            # Check whether the injection centers of each selected poisoned image have already been chosen
            center_list_path = os.path.join(self.poisoned_label_folder, self.injection_center_name + '.npy')
            print(f"Checking path: {center_list_path}")
            if os.path.exists(center_list_path):
                print("Injection center list already exist.")
                self.injection_center_dict = np.load(center_list_path, allow_pickle='TRUE').item()
                for poisoned_label_name, injection_center in self.injection_center_dict.items():
                    print("The injection center of image {} is {}".format(poisoned_label_name, injection_center))
            else:
                print("Start creating injection center list")
                self.injection_center_dict = self.influencer_attack.make_injection_center(center_list_path)

            # Making the poisoned images
            if os.path.exists(self.poisoned_image_folder) and len(glob.glob(f"{self.poisoned_image_folder}/*.pkl")) == self.num_poison:
                print("Poisoned images already exist")
                print(f"Poisoned images folder name: {self.poisoned_image_name}")
                poisoned_images = sorted(entry.name for entry in os.scandir(self.poisoned_image_folder) if entry.name.endswith(".pkl") and entry.is_file())
                for file_name in poisoned_images:
                    print(f"Poisoned image file: {file_name}")
            else:
                print("Start creating poisoned images")
                self.image_processor = ImageProcessor(
                self.scale,
                self.trigger_path, 
                self.trigger_size, 
                self.injection_center_dict, 
                self.poison_list, 
                self.poisoned_image_folder,
                self.trigger_transparency
            )
                self.image_processor.make_poisoned_images()

        # If random pixel labelling is performed, all images in the dataset need to go through the PRL process
        if self.PRL:
            if os.path.exists(self.prl_clean_label_folder) and len(glob.glob(f"{self.prl_clean_label_folder}/*.pkl")) == len(self.data_list):
                print("PRL clean labels already exist.")
            else:
                print("Perform random pixel labelling on the whole dataset without attacking")
                self.label_processor.make_prl_clean_labels()


        
