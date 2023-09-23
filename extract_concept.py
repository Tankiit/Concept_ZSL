import torch 
from torch.utils.data import Dataset
import torchvision
import numpy as np
import matplotlib.pyplot as plt 
import linecache
import pdb

from textblob import TextBlob, Word
import re 
from nltk.corpus import stopwords



_DISCARD_WORDS = ['photo', 'background', 'stock', 'image', 'closeup', 'jpg', 'picture', 'png', 'file', 'close up', 'pictures', 'ive', 'view', 'www', 'http', 'showing', 'blurred', 'shot', 'example', 'camera', 'footage', 'free','video', 'displaying', 'display', 'displayed', 'thumbnail', 'focus', 'focusing', 'detail', 'panoramic', 'standard', 'portrait', 'zoom', 'zoomed', 'show', 'showed', 'real', 'icon', 'pixelated', 'cropped', 'autofocus', 'caption', 'graphic', 'defocused', 'zoomed', ' pre ', 'available', 'royalty', 'etext', 'blurry', 'new', 'pic', 'left', 'houzz', 'full', 'small', 'br', 'looking', 'pro', 'angle', 'logo', 'close', 'right', 'blur', 'preview', 'wallpaper', 'dont', 'fixed', 'closed', 'open', 'profile', 'close', 'color', 'photo', 'colored', 'video', 'banner', 'macro', 'frame', 'cut', 'livescience', 'bottom', 'corner', 'tvmdl', 'overlay', 'original', 'sign', 'old', 'extreme', 'hq', 'isolated', 'figure', 'stockfoto', 'vrr', 'cm', 'photography', 'print', 'embedded', 'smaller', 'testing', 'captioned', 'year', 'photograph', '', 'selective', 'photoshopped', 'come', 'org', 'akc', 'iphone']


def clean_text(text):
    text = text.lower()
    text = re.sub("'", '', text)
    text = re.sub("[^A-Za-z0-9 \n']+", ' ', text)
    text = re.sub('fig\d+', ' ', text)
    text = re.sub(' . ', ' ', text)
    text = ' '.join([t for t in text.split(' ') if t not in _DISCARD_WORDS])
    return text



def parse_captions(cap_set, score_set):
    all_noun_phrases = {}
    all_words = {}
    all_nouns = {}
    all_verbs = {}
    all_adj = {}
    
    for cp in range(len(cap_set)):
        caps = cap_set[cp]
        
        sim_scores = score_set[cp]
        
        noun_phrases_for_image = {}
        words_for_image = {}
        
        nouns_for_image = {}
        verbs_for_image = {}
        adj_for_image = {}
        
        for i in range(5):
            cap = clean_text(caps[i])
            # blob = TextBlob(cap).correct()
            blob = TextBlob(cap)
            
            noun_phrases_in_cap = blob.noun_phrases
            words_in_cap = set([i[0] for i in blob.tags if i[1] in ['NN', 'NNP', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS']])
            nouns_in_cap = set([i[0] for i in blob.tags if i[1] in ['NN', 'NNP', 'NNS']])
            verbs_in_cap = set([i[0] for i in blob.tags if i[1] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']])
            adj_in_cap = set([i[0] for i in blob.tags if i[1] in ['JJ', 'JJR', 'JJS']])
            # words_in_cap = blob.words
            
            noun_phrases_for_image = get_max_score_for_word(noun_phrases_in_cap, noun_phrases_for_image, sim_scores[i])
            words_for_image = get_max_score_for_word(words_in_cap, words_for_image, sim_scores[i])
            nouns_for_image = get_max_score_for_word(nouns_in_cap, nouns_for_image, sim_scores[i])
            verbs_for_image = get_max_score_for_word(verbs_in_cap, verbs_for_image, sim_scores[i])
            adj_for_image = get_max_score_for_word(adj_in_cap, adj_for_image, sim_scores[i])
            
        # words_for_image = {k:v for (k,v) in words_for_image.items()}
        
        all_noun_phrases = {k: all_noun_phrases.get(k, 0) + noun_phrases_for_image.get(k, 0) for k in set(all_noun_phrases) | set(noun_phrases_for_image)}
        all_words = {k: all_words.get(k, 0) + words_for_image.get(k, 0) for k in set(all_words) | set(words_for_image)}
        all_nouns = {k: all_nouns.get(k, 0) + nouns_for_image.get(k, 0) for k in set(all_nouns) | set(nouns_for_image)}
        all_verbs = {k: all_verbs.get(k, 0) + verbs_for_image.get(k, 0) for k in set(all_verbs) | set(verbs_for_image)}
        all_adj = {k: all_adj.get(k, 0) + adj_for_image.get(k, 0) for k in set(all_adj) | set(adj_for_image)}
    
    all_noun_phrases = sorted({k:v/len(cap_set) for (k,v) in all_noun_phrases.items()}.items(), key = lambda k: k[1], reverse = True)
    all_words = sorted({k:v/len(cap_set) for (k,v) in all_words.items()}.items(), key = lambda k: k[1], reverse = True)
    all_nouns = sorted({k:v/len(cap_set) for (k,v) in all_nouns.items()}.items(), key = lambda k: k[1], reverse = True)
    all_verbs = sorted({k:v/len(cap_set) for (k,v) in all_verbs.items()}.items(), key = lambda k: k[1], reverse = True)
    all_adj = sorted({k:v/len(cap_set) for (k,v) in all_adj.items()}.items(), key = lambda k: k[1], reverse = True)

    return all_noun_phrases, all_words, all_nouns, all_verbs, all_adj
