import sys
from eR_Mask_RCNN.path import ROOT_DIR
sys.path.append(ROOT_DIR)

import os
import numpy as np
import pandas as pd
import time

from astropy.io.fits import getdata
from imgaug import augmenters as iaa

#Mask-RCNN
from mrcnn import utils
from mrcnn.config import Config
from mrcnn import model as modellib

from eR_Mask_RCNN.path import MODEL_DIR
COCO_MODEL_PATH = os.path.join(MODEL_DIR, 'mask_rcnn_coco.h5')
from eR_Mask_RCNN.common import cat2masks


class eRDataset(utils.Dataset):
    
    def __init__(self, raw_size:int=3240, grid_size:int=270):
        self.raw_size = raw_size 
        self.grid_size = grid_size
        self.size = raw_size - 2 * grid_size
        super(eRDataset, self).__init__()
    
    def load_sources(self, set_dir: str): #or load_shapes()
        self.dir = set_dir
        
        self.add_class('eR_sim', 1, 'ext')
        self.add_class('eR_sim', 2, 'point')
        
        num_sets = 0
        for setdir in os.listdir(self.dir):
            if 'set_' in setdir:
                # add tranining image set
                self.add_image('eR_sim', image_id=num_sets, path=os.path.join(self.dir,set_dir),
                    width=self.size, height=self.size, bg_color=(0,0,0))
                num_sets += 1
        self.images = [None]*(num_sets)
        self.masks = [None]*num_sets
        self.class_ids_mem = [None]*num_sets
        
        for i in range(num_sets):
            self.load_image_disk(i)
            self.load_mask_disk(i)
        return
    
    def load_image(self, image_id:int)->np.ndarray:
        return self.images[image_id]
    
    def load_image_disk(self, image_id:int)->np.ndarray:
        # load from disk -- each set directory contains seperate files for images and masks
        info = self.image_info[image_id]
        setdir = 'set_%d' % image_id
        # read images
        exp = getdata(os.path.join(self.dir,setdir,'exp_tile.fits.gz'),memmap=False)
        tile = getdata(os.path.join(self.dir,setdir,'tile.fits.gz'),memmap=False)

        image = np.zeros([self.raw_size, self.raw_size, 1], dtype=np.float64)
        image[:,:,0] = tile / exp
        image = np.nan_to_num(np.log(image))
        idx = slice(self.grid_size, self.raw_size - self.grid_size) #remove grid
        image = image[idx, idx]

        max_RGB = np.percentile(image, 99.995)
        # avoid saturation
        image /= max_RGB

        # Rescale to 16-bit int
        int8_max = np.iinfo(np.int8).max
        image *= int8_max
        image = image.astype(np.int8)

        self.images[image_id] = image
        return image

    def load_mask(self, image_id:int)->tuple:
        return self.masks[image_id], self.class_ids_mem[image_id]

    def load_mask_disk(self, image_id:int)->tuple:
        # Load from disk
        info = self.image_info[image_id]
        # load image set via image_id from phosim output directory
        setdir = 'set_%d' % image_id
        catpath = os.path.join(self.dir, setdir, 'objects.csv')
        
        masks, labels = cat2masks(catpath, raw_size=self.raw_size, grid_size=self.grid_size)
       
        self.class_ids_mem[image_id] = np.array(labels,dtype=np.uint8)
        self.masks[image_id] = np.array(masks,dtype=np.uint8)
        return self.masks[image_id], self.class_ids_mem[image_id]

class eRConfig(Config):
    
    # Give the configuration a recognizable name
    NAME = "eROSITA"
    IMAGE_CHANNEL_COUNT = 1

    # Batch size (images/step) is (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + ext and point
    
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512 #
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 32

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Maximum number of ground truth instances (objects) in one image
    MAX_GT_INSTANCES = 300
    DETECTION_MAX_INSTANCES = 300

    # Mean pixel values (RGB)
    MEAN_PIXEL = np.array([0])

    # Note the images per epoch = steps/epoch * images/GPU * GPUs
    # So the training time is porportional to the batch size
    # Use a small epoch since the batch size is large
    STEPS_PER_EPOCH = max(1, 1000 // (IMAGES_PER_GPU * GPU_COUNT))

    # Use small validation steps since the epoch is small
    VALIDATION_STEPS = max(1, 250 // (IMAGES_PER_GPU * GPU_COUNT))

    # Store masks inside the bounding boxes (looses some accuracy but speeds up training)
    USE_MINI_MASK = False
    

def eR_train(train_dir:str, val_dir:str):
    start_time = time.time()

    config = eRConfig()
    config.display()
    ## DATASET
    # Training dataset
    dataset_train = eRDataset()
    dataset_train.load_sources(train_dir)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = eRDataset()
    dataset_val.load_sources(val_dir)
    dataset_val.prepare()

    # Image augmentation
    augmentation = iaa.SomeOf((0, 4), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.GaussianBlur(sigma=(0.0, np.random.random_sample()*4+2)),
        iaa.AddElementwise((-5, 5))
    ])

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask", 'conv1'])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                #learning_rate=config.LEARNING_RATE,
                learning_rate=config.LEARNING_RATE,
                augmentation=augmentation,
                epochs=15,
                layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                augmentation=augmentation,
                epochs=25,
                layers="all")

    # Do one more with an even lower learning rate
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 100,
                augmentation=augmentation,
                epochs=35,
                layers="all")

    # Final stage
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 1000,
                augmentation=augmentation,
                epochs=50,
                layers="all")

    # Save weights
    model_path = os.path.join(MODEL_DIR, "astro_rcnn_decam.h5")
    model.keras_model.save_weights(model_path)

    print("Done in %.2f hours." % float((time.time() - start_time)/3600))

    return
