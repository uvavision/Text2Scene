#!/usr/bin/env python

import os, sys, cv2, json, math, PIL
import cairo, copy, random, re, pickle
from copy import deepcopy
import numpy as np
import os.path as osp
from time import time
from composites_config import get_config
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
from pycocotools import mask as COCOmask

import torch, torchtext
import torch.nn as nn


this_dir = osp.dirname(__file__)

###########################################################
## Directory
###########################################################

def maybe_create(dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)


def prepare_directories(config):
    postfix = datetime.now().strftime("%m%d_%H%M%S")
    model_name = '{}_{}'.format(config.exp_name, postfix)
    config.model_name = model_name
    config.model_dir = osp.join(config.log_dir, model_name)
    maybe_create(config.model_dir)


###########################################################
## Discretization
###########################################################

class NormalizedLocationMap(object):
    def __init__(self, config):
        self.cfg = config
        self.margin = 0.001 # avoid numerical troubles
        # Normalized grid positions
        self.cols, self.col_step = \
            np.linspace(self.margin, 1.0 - self.margin, num=config.grid_size[0],
                endpoint=True, retstep=True, dtype=np.float)
        self.rows, self.row_step = \
            np.linspace(self.margin, 1.0 - self.margin, num=config.grid_size[1],
                endpoint=True, retstep=True, dtype=np.float)
        Xs, Ys = np.meshgrid(self.cols, self.rows)
        self.coords = np.vstack((Xs.flatten(), Ys.flatten())).transpose()

    def index2coord(self, index):
        return self.coords[index].copy()

    def indices2coords(self, indices):
        return self.coords[indices].copy()

    def coord2index(self, coord):
        col_idx = int((coord[0] - self.margin)/self.col_step + 0.5)
        row_idx = int((coord[1] - self.margin)/self.row_step + 0.5)

        col_idx = max(0, min(col_idx, self.cfg.grid_size[0]-1))
        row_idx = max(0, min(row_idx, self.cfg.grid_size[1]-1))
        return row_idx * self.cfg.grid_size[0] + col_idx

    def coords2indices(self, coords):
        grids = (coords - self.margin)/np.array([self.col_step, self.row_step]).reshape((1,2))
        grids = (grids + 0.5).astype(np.int)
        grids[:, 0] = np.maximum(0, np.minimum(grids[:, 0], self.cfg.grid_size[0]-1))
        grids[:, 1] = np.maximum(0, np.minimum(grids[:, 1], self.cfg.grid_size[1]-1))
        return grids[:, 1] * self.cfg.grid_size[0] + grids[:, 0]


class NormalizedTransformationMap(object):
    def __init__(self, config):
        self.cfg = config
        self.margin = 0.001
        self.scales, self.scale_step = \
            np.linspace(self.margin, 1.0 - self.margin, num=config.num_scales,
                endpoint=True, retstep=True, dtype=np.float32)
        ratios = []
        K = int((config.num_ratios - 1)/2)
        for i in range(K, 0, -1):
            ratios.append(1.0/(1+i))
        for i in range(0, K + 1):
            ratios.append(1+i)
        self.ratios = np.array(ratios).astype(np.float32)

    def index2coord(self, inds):
        return np.array([self.scales[inds[0]], self.ratios[inds[1]]])

    def indices2coords(self, inds):
        scales = self.scales[inds[:, 0]]
        ratios = self.ratios[inds[:, 1]]
        coords = np.stack([scales, ratios], -1)
        return coords

    def coord2index(self, coord):
        scale_idx = int((coord[0] - self.margin)/self.scale_step + 0.5)
        scale_idx = max(0, min(scale_idx, self.cfg.num_scales-1))
        ratio_idx = np.argmin(np.absolute(self.ratios - coord[1]))
        return np.array([scale_idx, ratio_idx])

    def coords2indices(self, coords):
        scale_indices = ((coords[:, 0] - self.margin)/self.scale_step + 0.5).astype(np.int)
        scale_indices = np.maximum(0, np.minimum(scale_indices, self.cfg.num_scales-1))
        foo = coords[:, 1].reshape((-1, 1))
        bar = np.repeat(self.ratios[None, ...], foo.shape[0], axis=0)
        ratio_indices = np.argmin(np.absolute(foo - bar), axis=-1)
        return np.stack([scale_indices, ratio_indices], -1)

    def wh2coord(self, wh):
        scale = math.sqrt(wh[0] * wh[1] + self.cfg.eps)
        ratio = wh[0] / (wh[1] + self.cfg.eps)
        return np.array([scale, ratio])

    def whs2coords(self, whs):
        scales = np.sqrt(whs[:, 0] * whs[:, 1] + self.cfg.eps)
        ratios = whs[:, 0] / (whs[:, 1] + self.cfg.eps)
        coords = np.stack([scales, ratios], axis=-1)
        return coords

    def coord2wh(self, coord):
        area = coord[0] * coord[0]
        w = math.sqrt(area * coord[1] + self.cfg.eps)
        h = area/(w + self.cfg.eps)
        wh = np.array([w, h])
        wh = np.maximum(0, np.minimum(wh, 1.0))
        return wh

    def coords2whs(self, coords):
        areas = coords[:, 0] * coords[:, 0]
        ws = np.sqrt(areas * coords[:, 1] + self.cfg.eps)
        hs = areas/(ws + self.cfg.eps)
        whs = np.stack([ws, hs], axis=-1)
        whs = np.maximum(0, np.minimum(whs, 1.0))
        return whs

    def index2wh(self, index):
        coord = self.index2coord(index)
        wh = self.coord2wh(coord)
        return wh

    def indices2whs(self, indices):
        coords = self.indices2coords(indices)
        whs = self.coords2whs(coords)
        return whs

    def wh2index(self, wh):
        coord = self.wh2coord(wh)
        index = self.coord2index(coord)
        return index

    def whs2indices(self, whs):
        coords = self.whs2coords(whs)
        indices = self.coords2indices(coords)
        return indices


###########################################################
## Vocabulary
###########################################################
import string
punctuation_table = str.maketrans('', '', string.punctuation)
stop_words = set(stopwords.words('english'))

def further_token_process(tokens):
    tokens = [w.translate(punctuation_table) for w in tokens]
    tokens = [w for w in tokens if w.isalpha()]
    # TO-DO: determine if stop words should be removed
    # tokens = [w for w in tokens if not w in stop_words]
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
        del self.glovec
        self.glovec = None

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
## Visualization
###########################################################

def clamp_array(array, min_value, max_value):
    return np.minimum(np.maximum(min_value, array), max_value)


def compose(background_image, foreground_image, mask):
    if len(mask.shape) == 2:
        # If there is no depth channel
        alpha = mask[:,:,None].copy()
    else:
        alpha = mask.copy()
    composed_image = background_image * (1.0 - alpha) + foreground_image * alpha
    if np.amax(composed_image) > 2:
        # If the inputs are uint8 images
        return clamp_array(composed_image, 0, 255)
    else:
        return clamp_array(composed_image, 0.0, 1.0)


def patch_compose(background_image, xyxy, patch, db):
    # pad_value = np.array([[[103.53, 116.28, 123.675]]])
    image_index = patch['image_index']
    color_path = db.color_path_from_index(image_index)

    assert(osp.exists(color_path))

    # image resolutions
    src_img = background_image
    dst_img = cv2.imread(color_path, cv2.IMREAD_COLOR)
    src_width = src_img.shape[1]; src_height = src_img.shape[0]
    dst_width = dst_img.shape[1]; dst_height = dst_img.shape[0]

    # patch_mask
    rle_rep = deepcopy(patch['mask'])
    dst_msk = COCOmask.decode(rle_rep).astype(np.float32)

    # bounding boxes
    dst_xywh = deepcopy(patch['box'])
    dst_xyxy = xywh_to_xyxy(dst_xywh, dst_width, dst_height)
    src_xyxy = xyxy
    src_xywh = xyxy_to_xywh(src_xyxy)

    # area
    src_area = src_xywh[2] * src_xywh[3]
    dst_area = dst_xywh[2] * dst_xywh[3]

    # resize the target image
    factor = np.sqrt(src_area/(dst_area + 1e-10))
    dst_width = int(dst_width * factor+0.5)
    dst_height = int(dst_height * factor+0.5)
    dst_img = cv2.resize(dst_img, (dst_width, dst_height))
    dst_msk = cv2.resize(dst_msk, (dst_width, dst_height))
    dst_xyxy = (factor * dst_xyxy+0.5).astype(np.int32)
    dst_xywh = xyxy_to_xywh(dst_xyxy)

    # anchors that should match
    src_anchor = src_xywh[:2]; dst_anchor = dst_xywh[:2]
    offset = (src_anchor - dst_anchor).astype(np.int)

    # move the dst bounding box to the src anchor point
    dst_bb = dst_xyxy
    src_bb = dst_bb.copy()
    src_bb[:2] = dst_bb[:2] + offset
    src_bb[2:] = dst_bb[2:] + offset

    # in case the bbox of the target object is beyond the boundaries of the source image
    if src_bb[0] < 0:
        dst_bb[0] -= src_bb[0]; src_bb[0] = 0
    if src_bb[1] < 0:
        dst_bb[1] -= src_bb[1]; src_bb[1] = 0
    if src_bb[2] > src_width - 1:
        dst_bb[2] -= src_bb[2] - src_width + 1; src_bb[2] = src_width - 1
    if src_bb[3] > src_height - 1:
        dst_bb[3] -= src_bb[3] - src_height + 1; src_bb[3] = src_height - 1

    # composition
    output_mask  = np.zeros((src_height, src_width), dtype=np.float)
    output_image = src_img.copy()

    src_patch = src_img[src_bb[1]:(src_bb[3]+1), src_bb[0]:(src_bb[2]+1),:]
    dst_patch = dst_img[dst_bb[1]:(dst_bb[3]+1), dst_bb[0]:(dst_bb[2]+1),:]
    msk_patch = dst_msk[dst_bb[1]:(dst_bb[3]+1), dst_bb[0]:(dst_bb[2]+1)]
    patch_width = src_patch.shape[1]; patch_height = src_patch.shape[0]
    dst_patch = cv2.resize(dst_patch, (patch_width, patch_height))
    msk_patch = cv2.resize(msk_patch, (patch_width, patch_height))


    output_mask[src_bb[1]:(src_bb[3]+1), src_bb[0]:(src_bb[2]+1)] = msk_patch
    output_image[src_bb[1]:(src_bb[3]+1), src_bb[0]:(src_bb[2]+1),:] = \
        np.expand_dims(1.0 - msk_patch, axis=-1) * src_patch + \
        np.expand_dims(msk_patch, axis=-1) * dst_patch
    # output_image = output_image.astype(np.uint8)
    # cv2.rectangle(output_image, (src_xyxy[0], src_xyxy[1]), (src_xyxy[2], src_xyxy[3]), \
    #                         (255, 0, 0), 1)
    return output_image, output_mask


def patch_compose_and_erose(bg_image, bg_mask, bg_label, xyxy, patch, db, bg_noice=None):
    # pad_value = np.array([[[103.53, 116.28, 123.675]]])
    image_index = patch['image_index']
    patch_class = patch['class']
    color_path = db.color_path_from_index(image_index)
    assert(osp.exists(color_path))

    # image resolutions
    src_img = bg_image; src_msk = bg_mask
    dst_img = cv2.imread(color_path, cv2.IMREAD_COLOR)
    src_width = src_img.shape[1]; src_height = src_img.shape[0]
    dst_width = dst_img.shape[1]; dst_height = dst_img.shape[0]

    # patch_mask
    rle_rep = deepcopy(patch['mask'])
    dst_msk = COCOmask.decode(rle_rep).astype(np.float32)

    # bounding boxes
    dst_xywh = deepcopy(patch['box'])
    dst_xyxy = xywh_to_xyxy(dst_xywh, dst_width, dst_height)
    src_xyxy = xyxy
    src_xywh = xyxy_to_xywh(src_xyxy)

    # area
    src_area = src_xywh[2] * src_xywh[3]
    dst_area = dst_xywh[2] * dst_xywh[3]

    # resize the target image
    factor = np.sqrt(src_area/(dst_area + 1e-10))
    dst_width = int(dst_width * factor+0.5)
    dst_height = int(dst_height * factor+0.5)
    dst_img = cv2.resize(dst_img, (dst_width, dst_height), interpolation = cv2.INTER_CUBIC)
    dst_msk = cv2.resize(dst_msk, (dst_width, dst_height), interpolation = cv2.INTER_NEAREST)
    dst_xyxy = (factor * dst_xyxy+0.5).astype(np.int32)
    dst_xywh = xyxy_to_xywh(dst_xyxy)

    # anchors that should match
    src_anchor = src_xywh[:2]; dst_anchor = dst_xywh[:2]
    offset = (src_anchor - dst_anchor).astype(np.int)

    # move the dst bounding box to the src anchor point
    dst_bb = dst_xyxy
    src_bb = dst_bb.copy()
    src_bb[:2] = dst_bb[:2] + offset
    src_bb[2:] = dst_bb[2:] + offset

    # in case the bbox of the target object is beyond the boundaries of the source image
    if src_bb[0] < 0:
        dst_bb[0] -= src_bb[0]; src_bb[0] = 0
    if src_bb[1] < 0:
        dst_bb[1] -= src_bb[1]; src_bb[1] = 0
    if src_bb[2] > src_width - 1:
        dst_bb[2] -= src_bb[2] - src_width + 1; src_bb[2] = src_width - 1
    if src_bb[3] > src_height - 1:
        dst_bb[3] -= src_bb[3] - src_height + 1; src_bb[3] = src_height - 1

    # composition
    src_patch = src_img[src_bb[1]:(src_bb[3]+1), src_bb[0]:(src_bb[2]+1),:]
    dst_patch = dst_img[dst_bb[1]:(dst_bb[3]+1), dst_bb[0]:(dst_bb[2]+1),:]

    src_patch_mask = src_msk[src_bb[1]:(src_bb[3]+1), src_bb[0]:(src_bb[2]+1)]
    dst_patch_mask = dst_msk[dst_bb[1]:(dst_bb[3]+1), dst_bb[0]:(dst_bb[2]+1)]

    patch_width = src_patch.shape[1]; patch_height = src_patch.shape[0]

    dst_patch = cv2.resize(dst_patch, (patch_width, patch_height), interpolation = cv2.INTER_CUBIC)
    dst_patch_mask = cv2.resize(dst_patch_mask, (patch_width, patch_height), interpolation = cv2.INTER_NEAREST)

    # order
    if patch_class >= 83:
        dst_patch_mask = dst_patch_mask - src_patch_mask
        dst_patch_mask[dst_patch_mask<0] = 0
    
    patch_mask  = np.zeros((src_height, src_width), dtype=np.float)
    patch_mask[src_bb[1]:(src_bb[3]+1), src_bb[0]:(src_bb[2]+1)] = dst_patch_mask
    image_label = bg_label.copy()
    image_label[patch_mask>0] = patch_class
    image_mask = np.maximum(src_msk, patch_mask)


    composite_image = src_img.copy()
    composite_image[src_bb[1]:(src_bb[3]+1), src_bb[0]:(src_bb[2]+1),:] = \
        np.expand_dims(1.0 - dst_patch_mask, axis=-1) * src_patch + \
        np.expand_dims(dst_patch_mask, axis=-1) * dst_patch
    # output_image = output_image.astype(np.uint8)
    # cv2.rectangle(output_image, (src_xyxy[0], src_xyxy[1]), (src_xyxy[2], src_xyxy[3]), \
    #                         (255, 0, 0), 1)

    # boundary elision
    if bg_noice is not None:
        erosed_image = bg_noice.copy()
        if np.amax(dst_patch_mask) > 0:
            erosed_mask = dst_patch_mask.copy()
            radius = int(0.05 * min(src_width, src_height))
            if np.amin(erosed_mask) > 0:
                erosed_mask[0, :] = 0
                erosed_mask[-1,:] = 0
                erosed_mask[:, 0] = 0
                erosed_mask[:, -1] = 0

            # cv2.imwrite('debug.png', erosed_mask*255)
            sobelx = cv2.Sobel(erosed_mask, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(erosed_mask, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.abs(sobelx) + np.abs(sobely)
            edge = np.zeros_like(sobel)
            edge[sobel>0.9] = 1.0
            morp_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius, radius))
            edge = cv2.dilate(edge, morp_kernel, iterations = 1)
            row, col = np.where(edge > 0)
            n_edge_pixels = len(row)
            # print("n_edge_pixels: ", n_edge_pixels)
            noice_indices = np.random.permutation(range(n_edge_pixels))
            noice_indices = noice_indices[:(n_edge_pixels//2)]
            row = row[noice_indices]
            col = col[noice_indices]
            dst_patch[row, col, :] = 255

            erosed_image[src_bb[1]:(src_bb[3]+1), src_bb[0]:(src_bb[2]+1),:] = \
                np.expand_dims(1.0 - dst_patch_mask, axis=-1) * src_patch + \
                np.expand_dims(dst_patch_mask, axis=-1) * dst_patch
    else:
        erosed_image = None

    return composite_image, image_mask, patch_mask, image_label, erosed_image


def paint_box(ctx, color, box):
    # bounding box representation: xyxy to xywh
    # box is not normalized
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
    # Color
    ctx.set_source_rgb(0, 0, 0)
    # Font
    font_option = cairo.FontOptions()
    font_option.set_antialias(cairo.Antialias.SUBPIXEL)
    ctx.set_font_options(font_option)
    ctx.select_font_face("Purisa", cairo.FONT_SLANT_ITALIC, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(60)
    # ctx.set_operator(cairo.OPERATOR_ADD)
    # Position
    x = box[0]; y = box[1] + 50
    # w = box[2] - box[0] + 1
    # h = box[3] - box[1] + 1

    ctx.move_to(x, y)
    ctx.show_text(txt)


def create_squared_image(img, pad_value=None):
    # If pad value is not provided
    if pad_value is None:
        pad_value = np.array([103.53, 116.28, 123.675])

    width  = img.shape[1]
    height = img.shape[0]

    # largest length
    max_dim  = np.maximum(width, height)
    # anchored at the left-bottom position
    offset_x = 0 #int(0.5 * (max_dim - width))
    offset_y = max_dim - height #int(0.5 * (max_dim - height))

    output_img = pad_value.reshape(1, 1, img.shape[-1]) * \
                 np.ones((max_dim, max_dim, img.shape[-1]))
    output_img[offset_y : offset_y + height, \
               offset_x : offset_x + width,  :] = img

    return output_img.astype(np.uint8), offset_x, offset_y


def create_colormap(num_colors):
    # JET colorbar
    dz = np.arange(1, num_colors+1)
    norm = plt.Normalize()
    colors = plt.cm.jet(norm(dz))
    return colors[:,:3]


def layers_collage(layers):
    b = layers[:,:,::3]; g = layers[:,:,1::3]; r = layers[:,:,2::3]
    output_image = np.stack((np.amax(b, -1), np.amax(g, -1), np.amax(r, -1)), -1)
    output_mask = np.ones((layers.shape[0], layers.shape[1]), dtype=np.float32)
    row, col = np.where(np.sum(layers, -1)==0)
    output_mask[row, col] = 0
    return output_image, output_mask


def heuristic_collage(layers, num_prior_layer):
    thing_layers = layers[:,:,:3*num_prior_layer]
    stuff_layers = layers[:,:,3*num_prior_layer:]
    thing_image, thing_mask = layers_collage(thing_layers)
    stuff_image, stuff_mask = layers_collage(stuff_layers)
    stuff_mask = stuff_mask - thing_mask
    row, col = np.where(stuff_mask<0)
    stuff_mask[row, col] = 0

    output_image = compose(thing_image, stuff_image, stuff_mask)
    output_mask = thing_mask + stuff_mask
    return output_image, output_mask


###########################################################
## Bounding box
###########################################################

def normalize_xywh(xywh, width, height):
    max_dim = max(width, height)

    # move the bounding box to the left bottom position
    offset_x = 0 # int(0.5 * (max_dim - width))
    offset_y = max_dim - height # int(0.5 * (max_dim - height))
    cx, cy, nw, nh = xywh
    cx += offset_x; cy += offset_y

    # normalize the bounding box
    normalized_xywh = np.array([cx, cy, nw, nh], dtype=np.float32)/max_dim
    return normalized_xywh


def normalize_xywhs(xywhs, width, height):
    max_dim = max(width, height)

    # move the bounding boxes to the left bottom position
    offset_x = 0 # int(0.5 * (max_dim - width))
    offset_y = max_dim - height # int(0.5 * (max_dim - height))
    normalized_xywhs = xywhs.copy()
    normalized_xywhs[:, 0] = normalized_xywhs[:, 0] + offset_x
    normalized_xywhs[:, 1] = normalized_xywhs[:, 1] + offset_y

    # normalize the bounding boxes
    normalized_xywhs = normalized_xywhs/float(max_dim)
    return normalized_xywhs


def unnormalize_xywh(xywh, width, height):
    max_dim = max(width, height)
    # offset_x = 0 # int(0.5 * (max_dim - width))
    # offset_y = max_dim - height # int(0.5 * (max_dim - height))
    unnormalized_xywh = xywh * max_dim
    # unnormalized_xywh[0] = unnormalized_xywh[0] - offset_x
    # unnormalized_xywh[1] = unnormalized_xywh[1] - offset_y
    return unnormalized_xywh


def unnormalize_xywhs(xywhs, width, height):
    max_dim = max(width, height)
    # offset_x = 0 # int(0.5 * (max_dim - width))
    # offset_y = max_dim - height # int(0.5 * (max_dim - height))
    unnormalized_xywhs = xywhs * max_dim
    # unnormalized_xywh[:, 0] = unnormalized_xywh[:, 0] - offset_x
    # unnormalized_xywh[:, 1] = unnormalized_xywh[:, 1] - offset_y
    return unnormalized_xywhs


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
    xyxy = np.stack((xmin, ymin, xmax, ymax), -1)

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

    x = 0.5 * (box[0] + box[2])
    y = 0.5 * (box[1] + box[3])
    w = box[2] - box[0] + 1.0
    h = box[3] - box[1] + 1.0

    return np.array([x, y, w, h]).astype(np.int32)


def xyxys_to_xywhs(boxes):

    x = 0.5 * (boxes[:, 0] + boxes[:, 2])
    y = 0.5 * (boxes[:, 1] + boxes[:, 3])
    w = boxes[:, 2] - boxes[:, 0] + 1.0
    h = boxes[:, 3] - boxes[:, 1] + 1.0

    return np.stack((x, y, w, h), -1)

###########################################################
## Data
###########################################################

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
        

def pad_sequence(inputs, max_length, pad_val, sos_val=None, eos_val=None, eos_msk=None):
    # cut the input sequence off if necessary

    seq = inputs[:max_length]
    # mask for valid input items
    msk = [1.0] * len(seq)
    # if the length of the inputs is shorter than max_length, pad it with special items provided
    num_padding = max_length - len(seq)
    # pad SOS
    if sos_val is not None:
        if isinstance(sos_val, np.ndarray):
            seq = [sos_val.copy()] + seq
        else:
            seq = [sos_val] + seq
        msk = [1.0] + msk
    # pad EOS
    if eos_val is not None:
        if isinstance(eos_val, np.ndarray):
            seq.append(eos_val.copy())
        else:
            seq.append(eos_val)
        msk.append(eos_msk)
    # pad the sequence if necessary
    for i in range(num_padding):
        if isinstance(pad_val, np.ndarray):
            seq.append(pad_val.copy())
        else:
            seq.append(pad_val)
        msk.append(0.0)
    # the outputs are float arrays
    seq = np.array(seq)
    msk = np.array(msk).astype(np.float32)
    return seq, msk


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


def imgs_to_tensors(input_imgs, mean=None, std=None):
    imgs_np = []
    for i in range(len(input_imgs)):
        img_np = normalize(input_imgs[i], mean, std)
        imgs_np.append(img_np)
    imgs_np = np.stack(imgs_np, 0)
    # to pytorch
    imgs_th = torch.from_numpy(imgs_np).float()
    return imgs_th


def tensors_to_imgs(input_imgs_th, mean=None, std=None):
    imgs_np = []
    for i in range(len(input_imgs_th)):
        img_np = input_imgs_th[i].cpu().data.numpy()
        img_np = unnormalize(img_np, mean, std)
        imgs_np.append(img_np)
    imgs_np = np.stack(imgs_np, 0)
    return imgs_np


def vol_normalize(input_vol):
    vol_np = input_vol.astype(np.float32) - 128
    vol_np = vol_np.transpose((2, 0, 1))
    return vol_np


def vol_unnormalize(input_vol):
    vol_np = input_vol.transpose((1, 2, 0)) + 128
    return vol_np


def vols_to_tensors(input_vols):
    vols_np = [vol_normalize(input_vols[i]) for i in range(len(input_vols))]
    vols_np = np.stack(vols_np, 0)
    vols_th = torch.from_numpy(vols_np).float()
    return vols_th


def tensors_to_vols(input_vols_th):
    vols_np = [vol_unnormalize(input_vols_th[i].cpu().data.numpy()) for i in range(len(input_vols_th))]
    vols_np = np.stack(vols_np, 0)
    return vols_np


def batch_onehot_volumn_preprocess(input_vols_th, label_dim):
    bsize, h, w, c = input_vols_th.size()
    label_maps = input_vols_th[:,:,:,0].unsqueeze(-1).long()
    color_maps = input_vols_th[:,:,:,1:]
    onehots = input_vols_th.new_full((bsize, h, w, label_dim), 0.0)
    onehots.scatter_(-1, label_maps, 255.0)
    onehots[:,:,:,:3] = 0.0
    return torch.cat((onehots, color_maps), -1).float()


def batch_color_volumn_preprocess(input_vols_th, label_dim):
    onehot_vols_th = batch_onehot_volumn_preprocess(input_vols_th, label_dim)
    bsize, h, w, c = onehot_vols_th.size()
    label_maps = onehot_vols_th[:,:,:,:label_dim].unsqueeze(-1).float()
    color_maps = onehot_vols_th[:,:,:,-3:].unsqueeze(-2).float()
    mask_maps  = onehot_vols_th[:,:,:,-4].unsqueeze(-1).float()
    color_maps = color_maps.repeat(1,1,1,label_dim,1)
    vols = (color_maps * label_maps/255.0).view(bsize, h, w, 3*label_dim)
    vols[:,:,:,:3] = mask_maps.expand(bsize, h, w, 3)
    return vols


def sequence_onehot_volumn_preprocess(input_vols_th, label_dim):
    bsize, slen, h, w, c = input_vols_th.size()
    label_maps = input_vols_th[:,:,:,:,0].unsqueeze(-1).long()
    color_maps = input_vols_th[:,:,:,:,1:]
    onehots = input_vols_th.new_full((bsize, slen, h, w, label_dim), 0.0)
    onehots.scatter_(-1, label_maps, 255.0)
    onehots[:,:,:,:,:3] = 0.0
    return torch.cat((onehots, color_maps), -1).float()


def sequence_color_volumn_preprocess(input_vols_th, label_dim):
    onehot_vols_th = sequence_onehot_volumn_preprocess(input_vols_th, label_dim)
    bsize, slen, h, w, c = onehot_vols_th.size()
    label_maps = onehot_vols_th[:,:,:,:,:label_dim].unsqueeze(-1).float()
    color_maps = onehot_vols_th[:,:,:,:,-3:].unsqueeze(-2).float()
    mask_maps  = onehot_vols_th[:,:,:,:,-4].unsqueeze(-1).float()
    color_maps = color_maps.repeat(1,1,1,1,label_dim,1)
    vols = (color_maps * label_maps/255.0).view(bsize, slen, h, w, 3*label_dim)
    # vols[:,:,:,:,:3] = mask_maps.expand(bsize, slen, h, w, 3)
    return vols


###########################################################
## Patch
###########################################################

def expand_xyxys(boxes, width, height, ratio=0.2):
    bw = boxes[:, 2] - boxes[:, 0] + 1.0
    bh = boxes[:, 3] - boxes[:, 1] + 1.0

    ox = (bw * ratio).astype(np.int)
    oy = (bh * ratio).astype(np.int)

    xyxys = boxes.copy()
    xyxys[:,0] -= ox; xyxys[:,1] -= oy
    xyxys[:,2] += ox; xyxys[:,3] += oy

    return clip_boxes(xyxys, width, height)


def crop_with_padding(img, xywh, full_resolution, crop_resolution, pad_value):
    # xywh: scaled xywh
    # full_resolution: output resolution
    # crop_resolution: mask resolution

    img_width  = img.shape[1]
    img_height = img.shape[0]
    img_channel = img.shape[2]
    out_width  = int(xywh[2] * full_resolution[1]/crop_resolution[1])
    out_height = int(xywh[3] * full_resolution[0]/crop_resolution[0])

    out_img = np.ones((out_height, out_width, img_channel), dtype=np.float32) * \
        pad_value.reshape((1,1,img_channel))

    box_cenx = int(xywh[0])
    box_ceny = int(xywh[1])
    out_cenx = int(0.5 * out_width)
    out_ceny = int(0.5 * out_height)

    left_radius   = min(box_cenx,              out_cenx)
    right_radius  = min(img_width - box_cenx,  out_cenx)
    top_radius    = min(box_ceny,              out_ceny)
    bottom_radius = min(img_height - box_ceny, out_ceny)

    out_img[(out_ceny-top_radius):(out_ceny+bottom_radius), \
            (out_cenx-left_radius):(out_cenx+right_radius),:] \
            = img[(box_ceny-top_radius):(box_ceny+bottom_radius), \
                  (box_cenx-left_radius):(box_cenx+right_radius),:]

    # out_img = cv2.resize(out_img, (full_resolution[1], full_resolution[0]), interpolation = cv2.INTER_CUBIC).astype(np.int32)
    return out_img


def Monge_Kantorovitch_color_transfer(src_img, dst_img):
    src_vec = src_img.reshape((-1, 3))
    dst_vec = dst_img.reshape((-1, 3))
    A = np.cov(src_vec.transpose())
    B = np.cov(dst_vec.transpose())
    # MKL
    Da2, Va = np.linalg.eig(A)
    Da2[Da2<0] = 0
    Da = np.sqrt(Da2+1e-10)
    Da_diag = np.diag(Da)
    C = Da_diag @ Va.T @ B @ Va @ Da_diag
    # C = np.matmul(Da_diag, Va.T)
    # C = np.matmul(C, B)
    # C = np.matmul(C, Va)
    # C = np.matmul(C, Da_diag)
    Dc2, Vc = np.linalg.eig(C)
    Dc2[Dc2<0] = 0
    Dc = np.sqrt(Dc2+1e-10)
    Dc_diag = np.diag(Dc)
    Da_inv = np.diag(1.0/(Da+1e-10))
    T = Va @ Da_inv @ Vc @ Dc_diag @ Vc.T @ Da_inv @ Va.T
    # T = np.matmul(Va, Da_inv)
    # T = np.matmul(T, Vc)
    # T = np.matmul(T, Dc_diag)
    # T = np.matmul(T, Vc.T)
    # T = np.matmul(T, Da_inv)
    # T = np.matmul(T, Va.T)
    # interpolation
    src_mean = np.mean(src_vec, 0, keepdims=True)
    dst_mean = np.mean(dst_vec, 0, keepdims=True)
    out_img = np.matmul((src_vec - src_mean), T) + dst_mean
    out_img = out_img.reshape(src_img.shape) 
    return out_img
