import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("tables")
install("pretrainedmodels")
install("pytesseract")

import nltk
import json
import time
import torch
from torch.autograd import Variable
import numpy as np
from collections import Counter
from nltk import word_tokenize

# stuff for image feature extraction
import pretrainedmodels
import pretrainedmodels.utils
import pytesseract

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')


UNK = 0
PAD = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PretrainedCNN(object):
    """
    Class that encompasses loading of pre-trained CNN models.
    """
    def __init__(self, pretrained_cnn):
        self.pretrained_cnn = pretrained_cnn
        self.build_load_pretrained_cnn()

    def build_load_pretrained_cnn(self):
        """
            Load a pre-trained CNN using torchvision/cadene.
            Set it into feature extraction mode.
        """
        start = time.time()
        self.load_img = pretrainedmodels.utils.LoadImage()
        image_model_name = self.pretrained_cnn
        self.model = pretrainedmodels.__dict__[image_model_name](num_classes=1000, pretrained='imagenet')
        #self.model.train()
        self.tf_img = pretrainedmodels.utils.TransformImage(self.model)

        # returns features before the application of the last linear transformation
        # in the case of a resnet152, it will be a [1, 2048] tensor
        self.model.last_linear = pretrainedmodels.utils.Identity()
        elapsed = time.time() - start
        print("Built pre-trained CNN %s in %d seconds."%(image_model_name, elapsed))

    def load_image_from_path(self, path_img):
        """ Load an image given its full path in disk into a tensor
            ready to be used in a pretrained CNN.

        Args:
            path_img    The full path to the image file on disk.
        Returns:
                        The pytorch Variable to be used in the pre-trained CNN
                        that corresponds to the image after all pre-processing.
        """
        input_img = self.load_img(path_img)
        input_tensor = self.tf_img(input_img)
        input_var = torch.autograd.Variable(input_tensor.unsqueeze(0), requires_grad=False)
        return input_var

    def get_global_features(self, input):
        """ Returns features before the application of the last linear transformation.
            In the case of a ResNet, it will be a [1, 2048] tensor."""
        return self.model(input)

    def get_local_features(self, input):
        """ Returns features before the application of the first pooling/fully-connected layer.
            In the case of a ResNet, it will be a [1, 2048, 7, 7] tensor."""
        if self.pretrained_cnn.startswith('vgg'):
            #feats = self.model.local_features(input)
            feats = self.model._features(input)
        else:
            feats = self.model.features(input)
        return feats

pretrained_cnn = PretrainedCNN("resnet50")
pretrained_cnn.model = pretrained_cnn.model.to(DEVICE)

def seq_padding(X, padding=0):
    """
    add padding to a batch data 
    """
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0, img_names=None, img_path=None):
        # convert words id to long format.
        src = torch.from_numpy(src).to(DEVICE).long()
        trg = torch.from_numpy(trg).to(DEVICE).long()
        self.src = src
        # get the padding postion binary mask
        # change the matrix shape to  1×seq.length
        self.src_mask = (src != pad).unsqueeze(-2)
        self.img_feats = None
        
        # 如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        if trg is not None:
            # decoder input from target
            self.trg = trg[:, :-1]
            # decoder target from trg
            self.trg_y = trg[:, 1:]
            # add attention mask to decoder input
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # check decoder output padding number
            self.ntokens = (self.trg_y != pad).data.sum()
        
        if img_names is not None and img_path is not None:
            # img_feats = torch.from_numpy(img_feats).to(DEVICE)
            # self.img_feats = torch.reshape(img_feats, (self.src.size(0), 49, 2048))
            self.img_feats = self.extract_feats(img_path, img_names)
            # img_feats = torch.cat([IMAGE_FEATURES for _ in range(self.src.size(0))], dim=0)
            # self.img_feats = np.array(list(map(lambda x: x.T.flatten(), img_feats.data.cpu().numpy())))

    def extract_feats(self, img_path, img_names):
        batch_list = []
        for entry in img_names:
            batch_list.append(pretrained_cnn.load_image_from_path(img_path + "/" + entry + ".jpg"))
        
        # # random.shuffle(batch_list)
        input_imgs_minibatch = torch.cat( batch_list, dim=0 )
        input_imgs_minibatch = input_imgs_minibatch.to(DEVICE)
        feats = pretrained_cnn.get_local_features(input_imgs_minibatch)
        # feats = feature_extractor(input_imgs_minibatch)
        # feats = feats.flatten(start_dim=1).cpu().numpy()
        feats = np.array(list(map(lambda x: x.T.flatten(), feats.data.cpu().numpy())))

        return feats

    # Mask
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask  # subsequent_mask is defined in 'decoder' section.

class PrepareData:
    def __init__(self, train_file, dev_file, test_file, batch_size, img_path, degrade_source=False):
        # 01. Read the data and tokenize
        self.train_en, self.train_cn, self.train_imgs = self.load_data(train_file, degrade_source)
        self.dev_en, self.dev_cn, self.dev_imgs = self.load_data(dev_file, degrade_source)
        self.test_en, self.test_cn, self.test_imgs = self.load_data(test_file, degrade_source)

        # 02. build dictionary: English and Chinese
        self.en_word_dict, self.en_total_words, self.en_index_dict = self.build_dict(self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = self.build_dict(self.train_cn)

        # 03. word to id by dictionary
        self.train_en, self.train_cn, self.train_imgs = self.wordToID(self.train_en, self.train_cn, self.en_word_dict, self.cn_word_dict, image_feats=self.train_imgs)
        self.dev_en, self.dev_cn, self.dev_imgs = self.wordToID(self.dev_en, self.dev_cn, self.en_word_dict, self.cn_word_dict, image_feats=self.dev_imgs)
        self.test_en, self.test_cn, self.test_imgs = self.wordToID(self.test_en, self.test_cn, self.en_word_dict, self.cn_word_dict, image_feats=self.test_imgs)

        # 04. batch + padding + mask
        self.train_data = self.splitBatch(self.train_en, self.train_cn, batch_size, image_feats=self.train_imgs, img_path=img_path)
        self.dev_data = self.splitBatch(self.dev_en, self.dev_cn, batch_size, image_feats=self.dev_imgs, img_path=img_path)
        self.test_data = self.splitBatch(self.test_en, self.test_cn, batch_size, image_feats=self.test_imgs, img_path=img_path)

    def load_data(self, path, degrade_source="POS"):
        """
        Read English and Chinese Data 
        tokenize the sentence and add start/end marks(Begin of Sentence; End of Sentence)
        en = [['BOS', 'i', 'love', 'you', 'EOS'], 
              ['BOS', 'me', 'too', 'EOS'], ...]
        cn = [['BOS', '我', '爱', '你', 'EOS'], 
              ['BOS', '我', '也', '是', 'EOS'], ...]
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        en, zh, imgs = [], [], []
        for i in range(len(data)):
            en_sent = word_tokenize(data[i]["en_sent"].lower())
            if degrade_source == "POS":
                tags = nltk.pos_tag(en_sent, tagset='universal')
                en_sent = []
                for tag in tags:
                    if tag[1] in ["NOUN", "ADJ", "VERB", "PRON"]:
                        if bernoulli(0.3).rvs():
                            en_sent.append("BLANK")
                        else:
                            en_sent.append(tag[0])
                    else:
                        en_sent.append(tag[0])
            elif degrade_source == "DET":
                for j in range(len(en_sent) // 2, len(en_sent)):
                    en_sent[j] = "BLANK"
            else:
                en_sent = word_tokenize(data[i]["en_sent"].lower())
            en.append(["BOS"] + en_sent + ["EOS"])
            zh.append(["BOS"] + word_tokenize(" ".join([w for w in data[i]["zh_sent"]])) + ["EOS"])
            imgs.append(data[i]["image_id"])
        return en, zh, imgs

    def build_dict(self, sentences, max_words=50000):
        """
        sentences: list of word list 
        build dictionary as {key(word): value(id)}
        """
        word_count = Counter()
        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1
        # for i in range(len(data)):
        #     sentence = dict["en"][i]
        #     for s in sentence:
        #         word_count[s] += 1

        ls = word_count.most_common(max_words)
        total_words = len(ls) + 2
        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = UNK
        word_dict['PAD'] = PAD
        index_dict = {v: k for k, v in word_dict.items()}
        return word_dict, total_words, index_dict

    def wordToID(self, en, cn, en_dict, cn_dict, image_feats=None, sort=True):
        """
        convert input/output word lists to id lists. 
        Use input word list length to sort, reduce padding.
        """
        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(w, 0) for w in sent] for sent in cn]

        def len_argsort(seq):
            """
            get sorted index w.r.t length.
            """
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        if sort:  # update index
            sorted_index = len_argsort(out_en_ids)  # English
            out_en_ids = [out_en_ids[id] for id in sorted_index]
            out_cn_ids = [out_cn_ids[id] for id in sorted_index]
            if image_feats is not None:
                sorted_image_feats = [image_feats[id] for id in sorted_index]
                image_feats = sorted_image_feats

        return out_en_ids, out_cn_ids, image_feats

    def splitBatch(self, en, cn, batch_size, image_feats=None, img_path=None, shuffle=True):
        """
        get data into batches
        """
        idx_list = np.arange(0, len(en), batch_size)
        if shuffle:
            np.random.shuffle(idx_list)

        batch_indexs = []
        for idx in idx_list:
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))

        batches = []
        for batch_index in batch_indexs:
            batch_en = [en[index] for index in batch_index]
            batch_cn = [cn[index] for index in batch_index]
            # paddings: batch, batch_size, batch_MaxLength
            batch_cn = seq_padding(batch_cn, padding=PAD)
            batch_en = seq_padding(batch_en, padding=PAD)
            if image_feats is not None:
                img_batch = [image_feats[index] for index in batch_index]
                batches.append(Batch(batch_en, batch_cn, pad=PAD, img_names=img_batch, img_path=img_path))
            else:
                batches.append(Batch(batch_en, batch_cn, pad=PAD))
            #!!! 'Batch' Class is called here but defined in later section.
        return batches