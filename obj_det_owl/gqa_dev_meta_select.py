import os

import jax
import numpy as np
from scenic.projects.owl_vit import models
from scenic.projects.owl_vit.configs import clip_b32
from scipy.special import expit as sigmoid
import skimage
from skimage import io as skimage_io
from skimage import transform as skimage_transform
from scenic.projects.owl_vit.configs import clip_l14
import nltk

import json
import pickle as pkl
import random
from PIL import Image
from collections import defaultdict

import vqa_prepro.modify_program as clean_modules

def update_program(program):
    program=clean_modules.eliminate_obj_id(program)
    program=clean_modules.modify_choose(program)
    program=clean_modules.modify_diff_same(program)
    program=clean_modules.modify_verify(program)
    program=clean_modules.modify_relate(program)
    program=clean_modules.modify_filter(program)
    return program

from clean_all import mine_clean_all

def get_tokens(text_queries):
    tokenized_queries = np.array([
        module.tokenize(q, config.dataset_configs.max_query_length)
        for q in text_queries
    ])
    # Pad tokenized queries to avoid recompilation if number of queries changes:
    tokenized_queries = np.pad(
        tokenized_queries,
        pad_width=((0, 100 - len(text_queries)), (0, 0)),
        constant_values=0)
    return tokenized_queries

def get_img_feat(img_id):
    # Load example image:
    filename = os.path.join('/Data_Storage/Rui_Data_Space/multimodal/GQA',
                            'images',img_id+'.jpg')
    image_uint8 = skimage_io.imread(filename)
    image = image_uint8.astype(np.float32) / 255.0

    # Pad to square with gray pixels on bottom and right:
    h, w, _ = image.shape
    size = max(h, w)
    image_padded = np.pad(
        image, ((0, size - h), (0, size - w), (0, 0)), constant_values=0.5)

    # Resize to model input size:
    input_image = skimage.transform.resize(
        image_padded,
        (config.dataset_configs.input_size, config.dataset_configs.input_size),
        anti_aliasing=True)
    return input_image

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data

if __name__=='__main__':
    GQA_PATH='/Data_Storage/Rui_Data_Space/multimodal/GQA'
    THRESHOLD=0.10
    hold_list=['dark brown', 'orange', 'dark blue', 'taking bath', 'round', 'light brown', 'leather', 'porcelain', 'octagonal', 'tan', 'making a face', 'light blue', 'metal', 'red', 'rectangular', 'square', 'black', 'white', 'blue', 'teal', 'blond', 'glass', 'silver', 'triangular', 'brushing teeth', 'cream colored', 'brunette', 'taking a photo', 'wood', 'concrete', 'brown', 'taking a picture', 'beige', 'plastic', 'green', 'pink', 'khaki', 'gold', 'yellow', 'dark', 'gray', 'purple', 'maroon', 'brick', 'shaking hands','left','right','bottom','top','above']
    
    #obj_needed=load_pkl(os.path.join(GQA_PATH,'features','ques_needed_obj.pkl'))
    gqa_val_q=json.load(
        open(os.path.join(GQA_PATH,'original','testdev_balanced_questions.json'),'r'))
    names=list(gqa_val_q.keys())
    print ('Length of names:',len(names))
    
    scenic_info=load_pkl(
        '../../../mdetr/meta_generated_layout/dev_scenic_needed_layout.pkl')
    names=list(scenic_info.keys())
    print ('Length of names:',len(names))
    
    #start=0
    #end=2
    #valid_name=names[start:end]
    #print ('Number of valid names:',len(valid_name))
    #model initialization
    config = clip_l14.get_config()
    module = models.TextZeroShotDetectionModule(
        body_configs=config.model.body,
        normalize=config.model.normalize,
        box_bias=config.model.box_bias)
    variables = module.load_variables(config.init_from.checkpoint_path)
    
    invalid_list=[]
    
    for i,name in enumerate(names):
        if os.path.exists(os.path.join(GQA_PATH,
                                   'meta_one-all-scenic',
                                   name+'.pkl')):
            continue
        row=gqa_val_q[name]
        layout=row['semantic']
        ques=row['question']
     
        #clean_layout=update_program(layout)
        new_prog=scenic_info[name]
        ops=[step['operation'] for step in new_prog]
        all_objs=[step['argument'][0] for step in new_prog[:-1] if step['operation']=='select']
        obj_list=[]
        for obj in all_objs:
            if obj in hold_list:
                continue
            elif 'not' in obj:
                continue
            else:
                obj_list.append(obj)
        #objs=obj_needed[name]['objects']
        #obj_list=[obj for obj in objs]
        if len(obj_list)==0:
            print ('No detection needed',name,ques)
            continue
        obj_list=list(set(obj_list))
        """
        for obj in objs:
            new_obj=stemmer.singular_noun(obj)
            if new_obj==False:
                obj_list.append(obj)
            else:
                obj_list.append(new_obj)
        obj_list=list(set(obj_list))
        """
        img=row['imageId']
        print (i,obj_list,name,img)
        if i%200==0:
            print ('Already finished:',i*100.0/len(names))
        tokenized_queries=get_tokens(obj_list)
        try:
            input_image=get_img_feat(img)
        except:
            print('Invalid!')
            invalid_list.append(name)
            continue
        
        predictions = module.apply(
        variables,
        input_image[None, ...],
        tokenized_queries[None, ...],
        train=False)
        # Remove batch dimension and convert to numpy:
        predictions = jax.tree_map(lambda x: np.array(x[0]), predictions )
        
        logits = predictions['pred_logits'][..., :len(obj_list)]  
        # Remove padding.
        scores = sigmoid(np.max(logits, axis=-1))
        labels = np.argmax(predictions['pred_logits'], axis=-1)
        boxes = predictions['pred_boxes']
        
        info={obj:[] for obj in obj_list}
        for score, box, label in zip(scores, boxes, labels):
            if score < THRESHOLD:
                continue
            obj=obj_list[label]
            info[obj].append({
                'score':score,
                'bbox':box
            })
        
        pkl.dump(info,
                 open(os.path.join(GQA_PATH,
                                   'meta_one-all-scenic',
                                   name+'.pkl'),'wb'))
    print (invalid_list)
    json.dump(invalid_list,open(os.path.join(GQA_PATH,
                                             'features','invalid_filter_names.json'),'w'))