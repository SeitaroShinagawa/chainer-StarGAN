import numpy as np
import json
import cv2
import numpy as np
from .datasets_base import datasets_base

def load_json(path):
    with open(path,"r") as f:
        return json.load(f)

class celebA_train(datasets_base):
    """
    data_type: train, val, test
    """
    def __init__(self, dataset_path, att_list, data_type="train", flip=1, resize_to=128, crop_to=178):
        super(celebA_train, self).__init__(flip=flip, resize_to=resize_to, crop_to=crop_to)
        self.dataset_path = dataset_path
        self.att_dict = load_json(dataset_path+"/"+data_type+".json")
        self.imgAkey = [x for x in self.att_dict.keys()]
        self.attributes = [x for x in self.att_dict[self.imgAkey[0]]["attribute"].keys()]
        self.att_names = att_list

    def __len__(self):
        return len(self.imgAkey)

    def load_att(self, key):
        """
        attributes is filtered, referenced to IcGAN 
        https://github.com/Guim3/IcGAN/blob/master/data/donkey_celebA.lua#L25
        """
        atts = np.array([(self.att_dict[key]["attribute"][x]+2)*0.5 for x in self.att_names],dtype=np.int32) #[-1,1] -> [0,1]
        return atts

    def do_resize(self, img, resize_to=128): #for shrinking
        img = cv2.resize(img, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
        return img

    def do_random_crop(self, img, crop_to=178):
        h, w, ch = img.shape
        limx = w - crop_to
        limy = h - crop_to
        x = np.random.randint(0,limx)
        y = np.random.randint(0,limy)
        img = img[y:y+crop_to, x:x+crop_to]
        return img

    def crop(self, img, crop_to=178):
        h, w, ch = img.shape
        center_x = int(w*0.5)
        center_y = int(h*0.5)
        crop_half = int(crop_to*0.5)
        img = img[center_y-crop_half:center_y+crop_half,center_x-crop_half:center_x+crop_half,:]
        canvas = np.zeros((crop_to,crop_to,ch)).astype('f')
        h, w, ch = img.shape
        canvas[:h,:w,:] = img
        return canvas

    def do_augmentation(self, img):
        if self.crop_to > 0:
            img = self.crop(img, self.crop_to)

        if self.resize_to > 0:
            img = self.do_resize(img, self.resize_to)

        if self.flip > 0:
            img = self.do_flip(img)
        
        return img

    def get_example(self, i):
        np.random.seed(None)
        idA = self.imgAkey[np.random.randint(0,len(self.imgAkey))]

        imgA = cv2.imread(self.dataset_path + "img_align_celeba/" + idA, cv2.IMREAD_COLOR)
         
        attA = self.load_att(idA)
        attB = np.random.permutation(attA)
        
        imgA = self.do_augmentation(imgA)
        imgA = self.preprocess_image(imgA)
        
        return imgA, attA, attB
