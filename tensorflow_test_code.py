import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf
from core.yolov4 import YOLO, decode, compute_loss, decode_train
from core.dataset import Dataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all

flags.DEFINE_string('model', 'yolov3', 'yolov4, yolov3')
flags.DEFINE_string('weights', './scripts/yolov3.weights', 'pretrained weights')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')