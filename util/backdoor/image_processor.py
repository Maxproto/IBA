import os
import os.path
import cv2
import numpy as np
import copy
import pickle
import errno
import traceback


class ImageProcessor:
    def __init__(self, scale, trigger_path, trigger_size, injection_center_dict, poison_list, poisoned_image_folder, trigger_transparency=1.0, save_as_png=False):
        self.scale = scale
        self.trigger_path = trigger_path
        self.trigger_size = trigger_size
        self.injection_center_dict = injection_center_dict
        self.poison_list = poison_list
        self.poisoned_image_folder = poisoned_image_folder
        self.trigger_transparency = trigger_transparency
        self.save_as_png = save_as_png

    def preprocess(self, image, height, width):
        if image.shape[2] == 4:
            image = image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image) / 255
        image = cv2.resize(image, (width, height), cv2.INTER_LINEAR)
        return image

    def get_clean_image(self, image_path):
        image_name = image_path.split('/')[-1].split('.')[0]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        if self.scale != 1:
            height, width = int(image.shape[0] * self.scale), int(image.shape[1] * self.scale)
        else:
            height, width = image.shape[:2]
        image = self.preprocess(image, height, width)
        return image, image_name
    
    def get_trigger(self, trigger_path):
        trigger = cv2.imread(trigger_path, cv2.IMREAD_UNCHANGED)  # Notice the change in flag here.
        if trigger is None:
            raise ValueError(f"Trigger at {trigger_path} could not be read.")
        trigger = self.preprocess(trigger, self.trigger_size, self.trigger_size)
        return trigger
        
    def add_trigger(self, image, trigger, label_name):
        if label_name not in self.injection_center_dict:
            raise ValueError(f"No entry in injection_center_dict for label name: {label_name}")
        injection_center = self.injection_center_dict[label_name]
        if injection_center is None:
            raise ValueError(f"injection_center is None for label name: {label_name}")
        steps = int((self.trigger_size - 1) / 2)
        poisoned_image = copy.deepcopy(image)
        if not np.array_equal(injection_center, [0,0]):
            insert_row, insert_col = injection_center
            # Check if trigger has an alpha channel (i.e., it's an RGBA image)
            if trigger.shape[2] == 4:
                mask = trigger[..., 3:4]  # Normalize the alpha channel values to [0, 1]
                mask = np.concatenate([mask, mask, mask], axis=2)  # Repeat mask to have the same shape as image
            else:  # If the trigger is an RGB image, use a mask of ones
                mask = np.ones((trigger.shape[0], trigger.shape[1], 3))
            
            sub_image = poisoned_image[insert_row - steps:insert_row + steps + 1, insert_col - steps:insert_col + steps + 1, :]
            poisoned_image[insert_row - steps:insert_row + steps + 1, insert_col - steps:insert_col + steps + 1, :] = sub_image * (1 - mask) + trigger[..., :3] * self.trigger_transparency * mask
        return poisoned_image

    def make_poisoned_images(self):
        try:
            os.makedirs(self.poisoned_image_folder, exist_ok=True)
        except OSError as e:
            if e.errno != errno.EEXIST:
                print(f"Failed to create directory {self.poisoned_image_folder}. Error: {e}")
                return

        trigger = self.get_trigger(self.trigger_path)
        for data_path in self.poison_list:
            image_path, label_path = data_path
            label_name = label_path.split('/')[-1].split('.')[0]

            try:
                image, image_name = self.get_clean_image(image_path)
                print("Creating poisoned image with index {}".format(image_name))
                poisoned_image = self.add_trigger(image, trigger, label_name)
                poisoned_image_path = os.path.join(self.poisoned_image_folder, image_name + '.pkl')
                
                if self.save_as_png:
                    cv2.imwrite(poisoned_image_path.replace('.pkl', '.png'), cv2.cvtColor(poisoned_image * 255, cv2.COLOR_RGB2BGR))

                else:
                    with open(poisoned_image_path, 'wb') as f:
                        pickle.dump(poisoned_image, f)

            except Exception as e:
                print(f"Failed to process and save image {image_path}. Error: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    import random
    scale = 0.5
    trigger_path = "apple.png"
    trigger_size = 55
    injection_center_dict = {"bochum_000000_000313_leftImg8bit": [random.randint(100, 500), random.randint(100, 500)]}
    poison_list = [("bochum_000000_000313_leftImg8bit.png","bochum_000000_000313_leftImg8bit.png")]
    poisoned_image_folder = "tmp_folder"
    trigger_transparency = 0.5

    processor = ImageProcessor(scale, trigger_path, trigger_size, injection_center_dict, poison_list, poisoned_image_folder, trigger_transparency, save_as_png=True)
    processor.make_poisoned_images()
