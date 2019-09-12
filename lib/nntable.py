#!/usr/bin/env python

import numpy as np
import pickle, random
import os.path as osp
from time import time
from copy import deepcopy
from glob import glob
from annoy import AnnoyIndex

from composites_config import get_config
from composites_utils import *


class PerCategoryTable:
    def __init__(self, db):
        self.db = db
        self.cfg = db.cfg
        self.cache_dir = db.cache_dir

    def retrieve(self, query_vector, K=1):
        if getattr(self, 'nntable') is None:
            print('The NNTable has not been built, please run build_nntable first.')
            return None
        inds = list(self.nntable.get_nns_by_vector(query_vector, K, search_k=-1, include_distances=False))
        return [self.patchdb[x] for x in inds]
            
    def build_nntable(self, category_id, patchdb, use_cache=True):
        # keep a reference to the per-category patchdb
        self.patchdb = patchdb
        # cache output directories
        if self.cfg.use_patch_background:
            nntable_folder_name = self.db.split + '_nntables_with_bg'
        else:
            nntable_folder_name = self.db.split + '_nntables_without_bg'
        nntable_dir = osp.join(self.cache_dir, nntable_folder_name)
        maybe_create(nntable_dir)
        nntable_file = osp.join(nntable_dir, '%03d_nntable.ann'%category_id)

        # load or create the files
        if osp.exists(nntable_file) and use_cache:
            #################################################################
            ## Load the files if possible
            #################################################################
            self.nntable = AnnoyIndex(self.cfg.n_patch_features, 'angular')
            self.nntable.load(nntable_file)
        else:
            #################################################################
            ## create the cache files
            #################################################################
            category = self.db.classes[category_id]
            print("%s NNTable"%category)
            t0 = time()
            self.nntable = AnnoyIndex(self.cfg.n_patch_features, 'angular')
            for i in range(len(patchdb)):
                x = patchdb[i]
                image_index = x['image_index']
                instance_ind = x['instance_ind']
                feature_path = self.db.patch_path_from_indices(image_index, instance_ind, 'patch_feature', 'pkl', self.cfg.use_patch_background)
                with open(feature_path, 'rb') as fid:
                    features = pickle.load(fid)
                    self.nntable.add_item(i, features)
            n_trees = max(len(patchdb)//100, self.cfg.n_nntable_trees)
            self.nntable.build(n_trees)
            print("%s NNTable completes (time %.2fs)" % (category, time() - t0))

            #####################################################################
            ## Save cache files for faster loading in the future
            #####################################################################
            self.nntable.save(nntable_file)
            print('wrote nntable to {}'.format(nntable_file))


class AllCategoriesTables:
    def __init__(self, db):
        self.db = db
        self.cfg = db.cfg
        self.cache_dir = db.cache_dir

    def retrieve(self, category_id, query_vector, K=1):
        return self.per_category_tables[category_id].retrieve(query_vector, K)

    def build_nntables_for_all_categories(self, use_cache=True):
        patches_per_class = self.db.patches_per_class
        # Per category tables, three placeholders for the special SOS EOS PAD tokens
        self.per_category_tables = {0: None, 1: None, 2: None}
        for category_id, patches in patches_per_class.items():
            if category_id < 3:
                continue
            t = PerCategoryTable(self.db)
            t.build_nntable(category_id, patches, use_cache)
            self.per_category_tables[category_id] = t
