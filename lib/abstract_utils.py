#!/usr/bin/env python

import os, sys, cv2, json
import math, PIL, cairo
import copy, random, re
from copy import deepcopy
import numpy as np
import os.path as osp
from time import time
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from abstract_config import get_config

import torch, torchtext
import torch.nn as nn


###########################################################
## Directory
###########################################################

this_dir = osp.dirname(__file__)


def maybe_create(dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)


def prepare_directories(config):
    postfix = datetime.now().strftime("%m%d_%H%M%S")
    model_name = '{}_{}'.format(config.exp_name, postfix)
    config.model_name = model_name
    config.model_dir = osp.join(config.log_dir, model_name)
    maybe_create(config.model_dir)


def pickle_load(path):
    with open(path, 'rb') as fid:
        data_ = pickle.load(fid)
    return data_


def pickle_save(path, data):
    with open(path, 'wb') as fid:
        pickle.dump(data, fid, pickle.HIGHEST_PROTOCOL)


def json_load(path):
    with open(path, 'r') as fid:
        data_ = json.load(fid)
    return data_


def json_save(path, data):
    with open(path, 'w') as fid:
        json.dump(data, fid, indent=4, sort_keys=True)
        

###########################################################
## Discretization
###########################################################

class LocationMap(object):
    def __init__(self, config):
        self.cfg = config
        self.cols, self.col_step = \
            np.linspace(config.margin, config.image_size[0]-config.margin, 
                num=config.grid_size[0], 
                endpoint=True, retstep=True, dtype=np.float)
        self.rows, self.row_step = \
            np.linspace(config.margin, config.image_size[1]-config.margin, 
                num=config.grid_size[1], 
                endpoint=True, retstep=True, dtype=np.float)
        Xs, Ys = np.meshgrid(self.cols, self.rows)
        self.coords = np.vstack((Xs.flatten(), Ys.flatten())).transpose()

    def index2coord(self, index):
        return self.coords[index].copy()
    
    def indices2coords(self, indices):
        return self.coords[indices].copy()
        
    def coord2index(self, coord):
        col_idx = int(float(coord[0] - self.cfg.margin)/self.col_step + 0.5)
        row_idx = int(float(coord[1] - self.cfg.margin)/self.row_step + 0.5)
        col_idx = max(0, min(col_idx, self.cfg.grid_size[0]-1))
        row_idx = max(0, min(row_idx, self.cfg.grid_size[1]-1))
        return row_idx * self.cfg.grid_size[0] + col_idx
    
    def coords2indices(self, coords):
        grids = (coords - self.cfg.margin)/np.array([self.col_step, self.row_step]).reshape((1,2)).astype(np.float)
        grids = (grids + 0.5).astype(np.int)
        grids[:, 0] = np.maximum(0, np.minimum(grids[:, 0], self.cfg.grid_size[0]-1))
        grids[:, 1] = np.maximum(0, np.minimum(grids[:, 1], self.cfg.grid_size[1]-1))
        return grids[:, 1] * self.cfg.grid_size[0] + grids[:, 0]

###########################################################
## Vocabulary 
###########################################################
import string
punctuation_table = str.maketrans('', '', string.punctuation)
stop_words = set(stopwords.words('english'))
# print('stop_words: ', stop_words)

def further_token_process(tokens):
    tokens = [w.translate(punctuation_table) for w in tokens]
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    return tokens


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]


class Vocab(object):
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        for idx, word in enumerate(['<pad>', '<sos>', '<eos>']):
            self.word2index[word] = idx
            self.index2word.append(word)
            self.word2count[word] = 1
        self.n_words = 3  

        self.glovec = torchtext.vocab.GloVe(cache=osp.join(this_dir, '..', 'data', 'caches'))

    def get_glovec(self):
        vectors = []
        self.word2vector = {}
        for i in range(len(self.index2word)):
            w = self.index2word[i]
            v_th = self.glovec[w].squeeze()
            v_np = v_th.numpy()
            vectors.append(v_th)
            self.word2vector[w] = v_np
        self.vectors = torch.stack(vectors, 0)

    def load(self, path):
        with open(path, 'r') as fp:
            vocab_info = json.loads(fp.read())
        self.word2index = vocab_info['word2index']
        self.word2count = vocab_info['word2count']
        self.index2word = vocab_info['index2word']
        self.n_words = len(self.index2word)

    def save(self, path):
        vocab_info = {}
        vocab_info['word2index'] = self.word2index
        vocab_info['word2count'] = self.word2count
        vocab_info['index2word'] = self.index2word
        with open(path, 'w') as fp:
            json.dump(vocab_info, fp, indent=4, sort_keys=True)

    def addSentence(self, sentence):
        tokens = word_tokenize(sentence.lower())
        tokens = further_token_process(tokens)
        for word in tokens:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word.append(word)
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def filter_words(self, max_size=None, min_freq=1):
        counter = Counter(self.word2count)
        # rm special tokens before sorting
        counter['<pad>'] = 0; counter['<sos>'] = 0; counter['<eos>'] = 0
        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        # reset 
        self.index2word = []
        self.word2index = {}
        self.n_words = 0
        for idx, word in enumerate(['<pad>', '<sos>', '<eos>']):
            self.word2index[word] = idx
            self.index2word.append(word)
            self.n_words += 1
        
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.index2word) == max_size:
                break
            self.index2word.append(word)
            self.word2index[word] = self.n_words
            self.n_words += 1

        counter['<pad>'] = 1; counter['<sos>'] = 1; counter['<eos>'] = 1
        self.word2count = dict(counter)

    def word_to_index(self, w):
        return self.word2index.get(w, -1)    

###########################################################
## Pytorch 
###########################################################

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def weights_init(m):
    classname = m.__class__.__name__     
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)    
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def indices2onehots(indices, out_dim):
    bsize, slen = indices.size()
    inds = indices.view(bsize, slen, 1)
    onehots = torch.zeros(bsize, slen, out_dim).float()
    onehots.scatter_(-1, inds, 1.0)
    return onehots.float()


###########################################################
## Data  
###########################################################

def normalize(input_img, mean=None, std=None):
    if (mean is None) or (std is None):
        mean = np.array([0.485, 0.456, 0.406]).reshape((1,1,3))
        std  = np.array([0.229, 0.224, 0.225]).reshape((1,1,3))
    # [0, 255] --> [0, 1]
    img_np = input_img.astype(np.float32)/255.0
    # BGR --> RGB
    img_np = img_np[:, :, ::-1].copy()
    # Normalize
    img_np = (img_np - mean)/std 
    # H x W x C --> C x H x W
    img_np = img_np.transpose((2, 0, 1))

    return img_np


def unnormalize(input_img, mean=None, std=None):
    if (mean is None) or (std is None):
        mean = np.array([0.485, 0.456, 0.406]).reshape((1,1,3))
        std  = np.array([0.229, 0.224, 0.225]).reshape((1,1,3))
    
    # C x H x W --> H x W x C 
    img_np = input_img.transpose((1, 2, 0))
    # Unnormalize
    img_np = img_np * std + mean
    # RGB --> BGR
    img_np = img_np[:, :, ::-1].copy()
    # [0, 1] --> [0, 255]
    img_np = (255.0 * img_np).astype(np.int)
    img_np = np.maximum(0, img_np)
    img_np = np.minimum(255, img_np)
    img_np = img_np.astype(np.uint8)

    return img_np


class image_normalize(object):
    def __init__(self, field):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1,1,3))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1,1,3))
        self.field = field 

    def __call__(self, sample):
        raws = sample[self.field]
        imgs = []
        for i in range(len(raws)):
            img = normalize(raws[i], self.mean, self.std)
            imgs.append(img)
        imgs = np.stack(imgs, 0)
        sample[self.field] = imgs
        return sample


def img_to_tensor(input_imgs, mean=None, std=None):
    imgs_np = []
    for i in range(len(input_imgs)):
        img_np = normalize(input_imgs[i], mean, std)
        imgs_np.append(img_np)
    imgs_np = np.stack(imgs_np, 0)
    # to pytorch
    imgs_th = torch.from_numpy(imgs_np).float()
    return imgs_th


def tensor_to_img(input_imgs_th, mean=None, std=None):
    imgs_np = []
    for i in range(len(input_imgs_th)):
        img_np = input_imgs_th[i].cpu().data.numpy()
        img_np = unnormalize(img_np, mean, std)
        imgs_np.append(img_np)
    imgs_np = np.stack(imgs_np, 0)
    return imgs_np


###########################################################
## Visualization
###########################################################
def surface_to_image(surface):
    # get numpy data from cairo surface
    pimg = PIL.Image.frombuffer("RGBA", 
        (surface.get_width(), surface.get_height()),
        surface.get_data(), "raw", "RGBA", 0, 1)
    frame = np.array(pimg)[:,:,:-1]
    return frame


###########################################################
## Evaluation
###########################################################


def bb_iou(A, B):
    eps = 1e-8
    A_area = float(A[2] - A[0]) * (A[3] - A[1])
    B_area = float(B[2] - B[0]) * (B[3] - B[1])
    minx = max(A[0], B[0]); miny = max(A[1], B[1])
    maxx = min(A[2], B[2]); maxy = min(A[3], B[3])
    w = max(0, maxx - minx)
    h = max(0, maxy - miny)
    I_area = w * h
    return I_area/(A_area + B_area - I_area + eps)


def gaussian2d(x, y, sigmas):
    v = (x - y)/np.array(sigmas)
    return np.exp(-0.5 * np.sum(v * v))


def batch_gaussian1d(x, y, sigma):
    v = (x - y)/sigma
    return np.exp(-0.5 * np.sum(v * v, -1))


###########################################################
## Bounding box
###########################################################

def clip_xyxy(box, width, height):
    box[0] = max(0, box[0])
    box[1] = max(0, box[1])
    box[2] = min(box[2], width-1)
    box[3] = min(box[3], height-1)
    return box.astype(np.int32)
    

def clip_xyxys(boxes, width, height):
    boxes[:, 0] = np.maximum(boxes[:, 0], 0)
    boxes[:, 1] = np.maximum(boxes[:, 1], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], width - 1)
    boxes[:, 3] = np.minimum(boxes[:, 3], height - 1)
    return boxes.astype(np.int32)


def xywh_to_xyxy(box, width, height):
    x = box[0]; y = box[1]
    w = box[2]; h = box[3]

    xmin = x - 0.5 * w + 1
    xmax = x + 0.5 * w 
    ymin = y - 0.5 * h + 1
    ymax = y + 0.5 * h 

    xyxy = np.array([xmin, ymin, xmax, ymax])

    return clip_xyxy(xyxy, width, height)


def xywhs_to_xyxys(boxes, width, height):
    x = boxes[:, 0]; y = boxes[:, 1]
    w = boxes[:, 2]; h = boxes[:, 3] 

    xmin = x - 0.5 * w + 1.0
    xmax = x + 0.5 * w
    ymin = y - 0.5 * h + 1.0 
    ymax = y + 0.5 * h
    xyxy = np.vstack((xmin, ymin, xmax, ymax)).transpose()

    return clip_xyxys(xyxy, width, height)


def normalized_xywhs_to_xyxys(boxes):
    x = boxes[:, 0]; y = boxes[:, 1]
    w = boxes[:, 2]; h = boxes[:, 3] 

    xmin = x - 0.5 * w
    xmax = x + 0.5 * w
    ymin = y - 0.5 * h
    ymax = y + 0.5 * h
    xyxy = np.vstack((xmin, ymin, xmax, ymax)).transpose()

    xyxy[:, 0] = np.maximum(xyxy[:, 0], 0.0)
    xyxy[:, 1] = np.maximum(xyxy[:, 1], 0.0)
    xyxy[:, 2] = np.minimum(xyxy[:, 2], 1.0)
    xyxy[:, 3] = np.minimum(xyxy[:, 3], 1.0)

    return xyxy


def xyxy_to_xywh(box):

    x = 0.5 * (boxes[0] + boxes[2])
    y = 0.5 * (boxes[1] + boxes[3])
    w = boxes[2] - boxes[0] + 1.0
    h = boxes[3] - boxes[1] + 1.0

    return np.array([x, y, w, h])


def xyxys_to_xywhs(boxes):
    
    x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    y = 0.5 * (boxes[:, 1] + boxes[:, 3])
    w = boxes[:, 2] - boxes[:, 0] + 1.0
    h = boxes[:, 3] - boxes[:, 1] + 1.0

    return np.vstack((x, y, w, h)).transpose()


###########################################################
## Visualization
###########################################################

def paint_box(ctx, color, box):
    
    x = box[0]; y = box[1]
    w = box[2] - box[0] + 1
    h = box[3] - box[1] + 1
    
    ctx.set_source_rgb(color[0], color[1], color[2])
    ctx.set_line_width(10)
    ctx.rectangle(x, y, w, h)
    ctx.stroke()
    
    # ctx.set_operator(cairo.OPERATOR_ADD)
    # ctx.fill()
        

def paint_txt(ctx, txt, box):
    font_option = cairo.FontOptions()
    font_option.set_antialias(cairo.Antialias.SUBPIXEL)
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_font_options(font_option) 
    ctx.select_font_face("Purisa", cairo.FONT_SLANT_ITALIC, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(60)
    # ctx.set_operator(cairo.OPERATOR_ADD)
    x = box[0]; y = box[1] + 50
    w = box[2] - box[0] + 1
    h = box[3] - box[1] + 1

    ctx.move_to(x, y)
    ctx.show_text(txt)


def create_squared_image(img, pad_value=None):
    if pad_value is None:
        pad_value = np.array([255,255,255])

    width  = img.shape[1]
    height = img.shape[0]

    max_dim  = np.maximum(width, height)
    offset_x = 0 #int(0.5 * (max_dim - width))
    offset_y = max_dim - height #int(0.5 * (max_dim - height))

    output_img = pad_value.reshape(1, 1, img.shape[-1]) * \
                 np.ones((max_dim, max_dim, img.shape[-1]))
    output_img[offset_y : offset_y + height, \
               offset_x : offset_x + width,  :] = img

    return output_img.astype(np.uint8), offset_x, offset_y


def create_colormap(num_colors):
    dz = np.arange(1, num_colors+1)
    norm = plt.Normalize()
    colors = plt.cm.jet(norm(dz))
    return colors[:,:3]


###########################################################
## Logging
###########################################################
def log_scores(infos, path):
    log_info = {}

    unigram_P = infos.unigram_P()
    log_info['unigram_P'] = [np.mean(unigram_P), np.std(unigram_P), np.amin(unigram_P), np.amax(unigram_P)]

    unigram_R = infos.unigram_R()
    log_info['unigram_R'] = [np.mean(unigram_R), np.std(unigram_R), np.amin(unigram_R), np.amax(unigram_R)]

    bigram_P = infos.mean_bigram_P()
    log_info['bigram_P'] = bigram_P

    bigram_R = infos.mean_bigram_R()
    log_info['bigram_R'] = bigram_R

    pose = infos.pose()
    log_info['pose'] = [np.mean(pose), np.std(pose), np.amin(pose), np.amax(pose)]

    expr = infos.expr()
    log_info['expr'] = [np.mean(expr), np.std(expr), np.amin(expr), np.amax(expr)]

    scale = infos.scale()
    log_info['scale'] = [np.mean(scale), np.std(scale), np.amin(scale), np.amax(scale)]

    flip = infos.flip()
    log_info['flip'] = [np.mean(flip), np.std(flip), np.amin(flip), np.amax(flip)]

    unigram_coord = infos.unigram_coord()
    log_info['unigram_coord'] = [np.mean(unigram_coord), np.std(unigram_coord), np.amin(unigram_coord), np.amax(unigram_coord)]
    
    bigram_coord = infos.bigram_coord()
    log_info['bigram_coord'] = [np.mean(bigram_coord), np.std(bigram_coord), np.amin(bigram_coord), np.amax(bigram_coord)]
    
    with open(path, 'w') as fp:
        json.dump(log_info, fp, indent=4, sort_keys=True)


###########################################################
## Test 
###########################################################

def abstract_arguments(config):
    config.input_size = [224, 224]
    config.input_vocab_size = 2538
    config.output_cls_size = 61
    config.max_input_length = 5
    config.n_conv_hidden = 1024
    config.obj_loss_weight = 5.0
    config.use_bn = True
    return config


import unittest


class utilsTest(unittest.TestCase):
    def setUp(self):
        config, unparsed = get_config()
        np.random.seed(config.seed)
        random.seed(config.seed)
        self.cfg = deepcopy(config)

    def tearDown(self):
        pass

    def test_abstract_locmap(self):
        self.cfg = abs_arguments(self.cfg)

        locMap = LocationMap(self.cfg)
        pred_inds, gt_inds, coords = [], [], []

        for i in range(100):
            # x = self.cfg.margin + (self.cfg.image_size[0] - 2*self.cfg.margin) * np.random.random()
            # y = self.cfg.margin + (self.cfg.image_size[1] - 2*self.cfg.margin) * np.random.random()
            x = self.cfg.image_size[0] * np.random.random()
            y = self.cfg.image_size[1] * np.random.random()
            coord = np.array([x, y])
            coords.append(coord)
            pred_idx = locMap.coord2index(coord)
            pred_inds.append(pred_idx)

            diffs = locMap.coords - coord.reshape((1,2))
            dists = diffs[:,0] * diffs[:,0] + diffs[:,1] * diffs[:,1]
            gt_idx = np.argmin(dists)
            gt_inds.append(gt_idx)

        self.assertSequenceEqual(pred_inds, gt_inds)

        # batch process
        batched_inds = locMap.coords2indices(np.array(coords))
        self.assertSequenceEqual(pred_inds, batched_inds.tolist())

    def test_coco_locmap(self):
        self.cfg = coco_arguments(self.cfg)

        locMap = CocoLocationMap(self.cfg)
        pred_inds, gt_inds, coords = [], [], []
        for i in range(100):
            x = np.random.random()
            y = np.random.random()
            coord = np.array([x, y])
            coords.append(coord)
            pred_idx = locMap.coord2index(coord)
            pred_inds.append(pred_idx)

            diffs = locMap.coords - coord.reshape((1,2))
            dists = diffs[:,0] * diffs[:,0] + diffs[:,1] * diffs[:,1]
            gt_idx = np.argmin(dists)
            gt_inds.append(gt_idx)
        self.assertSequenceEqual(pred_inds, gt_inds)
        # batch process
        batched_inds = locMap.coords2indices(np.array(coords))
        self.assertSequenceEqual(pred_inds, batched_inds.tolist())

    # def test_coco_transmap(self):
    #     self.cfg = coco_arguments(self.cfg)
    #     transMap = CocoTransformationMap(self.cfg)
    #     whs = np.random.random((100, 2))
    #     inds_1  = transMap.whs2indices(whs)
    #     new_whs = transMap.indices2whs(inds_1)
    #     inds_2  = transMap.whs2indices(new_whs)

    #     self.assertSequenceEqual(inds_1.tolist(), inds_2.tolist())

        # x = np.concatenate((whs, new_whs), axis=-1)
        # y = np.absolute(whs - new_whs)
        # z = np.concatenate((x, y), axis=-1)
    
    
if __name__ == '__main__':
    unittest.main()
    