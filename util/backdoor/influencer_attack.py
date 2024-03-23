import numpy as np
import copy
import torch

class InfluencerAttack:
    def __init__(self, seed, num_classes, victim_class, trigger_size, lower_dist, upper_dist, edge_crop, 
                 NNI, poison_list, label_processor):
        self.seed = seed
        self.num_classes = num_classes
        self.victim_class = victim_class
        self.trigger_size = trigger_size
        self.lower_dist = lower_dist
        self.upper_dist = upper_dist
        self.edge_crop = edge_crop
        self.NNI = NNI
        self.poison_list = poison_list
        self.label_processor = label_processor
        self.injection_mask_dict = None
        self.injection_center_dict = None

    def get_class_area(self, class_index, label):
        class_area = copy.deepcopy(label).to(0)
        class_area[label == class_index] = 1
        class_area[label != class_index] = 0
        return class_area

    def create_horizontal_mask(self, padding_type, base, height, steps):
        mask_horizontal = base.clone().detach().to(0).float()
        if padding_type == 'one':
            padding_value = torch.ones(height, 1).to(0)
        else:
            padding_value = torch.zeros(height, 1).to(0)
        for step in range(1, steps + 1):
            col_padding = padding_value.repeat(1, step)
            mask_left = torch.column_stack((base.clone().detach()[:, step:].to(0), col_padding))
            mask_right = torch.column_stack((col_padding, base.clone().detach()[:, :-step].to(0)))
            mask_tmp = mask_left * mask_right 
            mask_horizontal *= mask_tmp
        return mask_horizontal
    
    def create_vertical_mask(self, padding_type, base, width, steps):
        mask_vertical = base.clone().detach().to(0).float()
        if padding_type == 'one':
            padding_value = torch.ones(1, width).to(0)
        else:
            padding_value = torch.zeros(1, width).to(0)
        for step in range(1, steps + 1):
            row_padding = padding_value.repeat(step, 1)
            mask_up = torch.vstack((base.clone().detach()[step:, :].to(0), row_padding))
            mask_down = torch.vstack((row_padding, base.clone().detach()[:-step, :].to(0)))
            mask_tmp = mask_up * mask_down
            mask_vertical *= mask_tmp
        return mask_vertical
        
    def get_boundary_mask(self, label, height, width): # Excludes Adjacent Areas
        tmp_label = torch.from_numpy(copy.deepcopy(label))
        boundary_list = torch.empty((2, 0), dtype=torch.int).to(0)
        tmp_steps = int((self.trigger_size - 1) / 2)
        steps = min(tmp_steps, height, width)
        for i in range(0, self.num_classes):
            if i != self.victim_class:
                class_area = self.get_class_area(i, tmp_label)
                if len(torch.unique(class_area)) == 2:
                    '''
                    When creating boundary mask, the base mask is the class area with value 1 indicating the class object
                    and 0 indicating the others objects and background, we want to crop the adjancy area of 2 classes out,
                    (making the possible injection area smaller) therefore we use zero-padding
                    '''
                    boundary_horizontal = self.create_horizontal_mask('zero', class_area, height, steps)
                    boundary_vertical = self.create_vertical_mask('zero', boundary_horizontal, width, steps)
                    tmp_list = (boundary_vertical == 1).nonzero(as_tuple=False).T
                    if tmp_list.numel() > 0:
                        boundary_list = torch.cat((boundary_list, tmp_list), dim=-1)
        boundary_mask = torch.zeros((height, width)).to(0)
        boundary_mask[tuple(boundary_list)] = 1
        return boundary_mask   
    
    def get_neighbor_mask(self, label, height, width): # Selects Suitable Area Around Victim Class
        tmp_label = torch.from_numpy(copy.deepcopy(label)).to(0)
        '''
        When applying neightest neighbor injection strategy, the base mask is the victim class area with 
        value 0 indicating the victim class object and 1 indicating the others objects and background, we
        want to select the suitable area around the victim class(making the possible injection area larger),
        therefore we use one-padding   
        '''
        if self.upper_dist != 0:
            victim_area = torch.ones((height,width)).to(0) - self.get_class_area(self.victim_class, tmp_label)
            lower_bound = min(self.lower_dist, height, width)
            upper_bound = min(self.upper_dist, height, width)

            lower_bound_horizontal = self.create_horizontal_mask('one', victim_area, height, lower_bound)
            lower_bound_vertical = self.create_vertical_mask('one', lower_bound_horizontal, width, lower_bound)
            victim_list_1 = (lower_bound_vertical == 0).nonzero(as_tuple=True)
            
            upper_bound_horizontal = self.create_horizontal_mask('one', victim_area, height, upper_bound)
            upper_bound_vertical = self.create_vertical_mask('one', upper_bound_horizontal, width, upper_bound)
            victim_list_2 = (upper_bound_vertical == 0).nonzero(as_tuple=True)
            
            victim_map_1 = torch.ones((height, width)).to(0)
            victim_map_1[tuple(victim_list_1)] = 0
            victim_map_2 = torch.zeros((height, width)).to(0)
            victim_map_2[tuple(victim_list_2)] = 1
            neighbor_mask = victim_map_1 * victim_map_2
        return neighbor_mask
    
    def edge_cropping(self, mask, height, width): # Excluded the edge area of the image
        edge_mask = np.ones((height, width), bool)
        edge_mask[self.edge_crop:-self.edge_crop, self.edge_crop:-self.edge_crop] = False
        mask[edge_mask] = 0
        return mask
            
    def make_injection_mask(self, save_path):
        self.injection_mask_dict = {}
        for data_path in self.poison_list:
            image_path, label_path = data_path
            label, poisoned_label_name = self.label_processor.get_clean_label(label_path)
            height, width = label.shape
            boundary_mask = self.get_boundary_mask(label, height, width)
            if self.NNI:
                neighbor_mask = self.get_neighbor_mask(label, height, width)
                final_mask = boundary_mask * neighbor_mask
            else:
                final_mask = boundary_mask
            final_mask = self.edge_cropping(final_mask, height, width)
            self.injection_mask_dict[poisoned_label_name] = np.argwhere((final_mask).cpu().numpy())  
        np.save(save_path, self.injection_mask_dict)
        return self.injection_mask_dict
    
    def make_injection_center(self, save_path):
        np.random.seed(self.seed)
        self.injection_center_dict = {}
        for data_path in self.poison_list:
            image_path, label_path = data_path
            poisoned_label_name = label_path.split('/')[-1].split('.')[0]
            available_list = self.injection_mask_dict[poisoned_label_name]
            if available_list.size > 0:
                self.injection_center_dict[poisoned_label_name] = available_list[np.random.randint(len(available_list))]
            else:
                self.injection_center_dict[poisoned_label_name] = np.array([0, 0], dtype=np.int64)
            print("The injection center of image {} is {}".format(poisoned_label_name, self.injection_center_dict[poisoned_label_name]))
        np.save(save_path, self.injection_center_dict)
        return self.injection_center_dict
