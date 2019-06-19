#!/usr/bin/env python

import _init_paths
import os, sys
import cv2, json
import math, time
import numpy as np
import random
import os.path as osp
from glob import glob
import csv
from utils import *


with open('captions_val2017.json', 'r') as fp:
    captions = json.loads(fp.read())


def get_all_candidate_indices(model_1, model_2, num_samples):
    model_1_paths = sorted(glob('../evaluation/%s_results/*.jpg'%model_1))
    model_1_names = [osp.splitext(osp.basename(x))[0] for x in model_1_paths]
    # print(len(model_1_names), model_1_names[:10])

    model_2_paths = sorted(glob('../evaluation/%s_results/*.jpg'%model_2))
    model_2_names = [osp.splitext(osp.basename(x))[0] for x in model_2_paths]
    # print(len(model_2_names), model_2_names[:10])

    indices = np.random.permutation(range(len(model_1_names)))
    # print(len(indices), indices[:10])

    candidate_names = []
    for x in indices:
        name = model_1_names[x]
        if name in model_2_names:
            candidate_names.append(name)
    
    candidate_names = candidate_names[:num_samples]
    with open('candidates_for_amt.json', 'w') as fp:
        json.dump(candidate_names, fp, indent=4, sort_keys=True)

    print(len(candidate_names))


def get_urls(model_name, file_names):
    url_names = []
    for i in range(len(file_names)):
        name = file_names[i]
        url = 'http://www.cs.virginia.edu/~ft3ex/data/user_study_coco/images/%s_results/%s.jpg'%(model_name, name)
        url_names.append(url)
    return url_names


def prepare_hits(model_1, model_2):
    with open('candidates_for_amt.json', 'r') as fp:
        file_names = json.loads(fp.read())
    
    model_1_urls = get_urls(model_1, file_names)
    model_2_urls = get_urls(model_2, file_names)

    # for i in range(10):
    #     print(model_1_urls[i])
    #     print(model_2_urls[i])
    #     print('----------------')

    url_pairs = []
    for i in range(len(model_1_urls)):
        if random.random() > 0.5:
            pair = [model_1_urls[i], model_2_urls[i], 0]
        else:
            pair = [model_2_urls[i], model_1_urls[i], 1]
        url_pairs.append(pair)
    
    # for i in range(10):
    #     print(url_pairs[i][0])
    #     print(url_pairs[i][1])
    #     print('----------------')
    
    # print(len(url_pairs))

    num_groups = len(url_pairs)//10
    # print(num_groups)

    head_line = []
    for i in range(10):
        head_line.append('flag_%02d'%i)
        head_line.append('text_%02d'%i)
        head_line.append('image_url_%02d_0'%i)
        head_line.append('image_url_%02d_1'%i)

    batches = []
    for i in range(num_groups):
        start_ind = i * 10
        end_ind = (i+1) * 10
        new_line = []
        for j in range(start_ind, end_ind):
            pair = url_pairs[j]
            name = osp.splitext(osp.basename(pair[0]))[0]
            # print(name)
            text = captions[name]
            new_line.append(pair[2])
            new_line.append(text)
            new_line.append(pair[0])
            new_line.append(pair[1])
        batches.append(new_line)
    
    # print(len(batches))

    csv_path = '%s.csv'%model_2
    with open(csv_path, 'w') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(head_line)
        for i in range(len(batches)):
            wr.writerow(batches[i])
            print(i)

    
def resize_sg2im():
    maybe_create('sg2im_256_results')
    paths = sorted(glob('sg2im_results/*.jpg'))
    for i in range(len(paths)):
        path = paths[i]
        sm_img = cv2.imread(path, cv2.IMREAD_COLOR)
        lg_img = cv2.resize(sm_img, (256, 256))
        out_path = osp.join('sg2im_256_results', osp.basename(path))
        cv2.imwrite(out_path, lg_img)
        print(i)

def resize_syn():
    maybe_create('synthesis_images_64_results')
    paths = sorted(glob('synthesis_images_256_results/*.jpg'))
    for i in range(len(paths)):
        path = paths[i]
        sm_img = cv2.imread(path, cv2.IMREAD_COLOR)
        sm_img = cv2.resize(sm_img, (64, 64))
        lg_img = cv2.resize(sm_img, (256, 256))
        out_path = osp.join('synthesis_images_64_results', osp.basename(path))
        cv2.imwrite(out_path, lg_img)
        print(i)

    



if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    # get_all_candidate_indices('sg2im', 'attngan', 500)
    # prepare_hits('synthesis_images_256', 'sg2im_256')
    # prepare_hits('synthesis_images_256', 'hdgan')
    prepare_hits('synthesis_images_64', 'sg2im_256')

    # resize_sg2im()
    # resize_syn()

