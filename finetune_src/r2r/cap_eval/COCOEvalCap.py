__author__ = 'tylin'
from importlib.resources import path
from .bleu.bleu import Bleu
from .meteor.meteor import Meteor
from .rouge.rouge import Rouge
from .cider.cider import Cider
from .spice.spice import Spice
import networkx as nx
import sys
from collections import defaultdict

class COCOEvalCap(object):
    def __init__(self, splits, tok, val_instr_data, use_clip16=False):
        self.evalImgs = []
        self.eval = {}
        self.splits = splits
        self.tok = tok
        self.gt = defaultdict(list)
        self.use_clip16 = use_clip16
        for split in splits:
            for item in val_instr_data:
                self.gt[str(item['path_id'])].append(item['instruction'])


        # for split in splits:
        #     for item in load_datasets([split]):
        #         if scans is not None and item['scan'] not in scans:
        #             continue
        #         self.gt[str(item['path_id'])] = item
        #         self.scans.append(item['scan'])
        #         self.instr_ids += ['%s_%d' % (item['path_id'], i) for i in range(len(item['instructions']))]

        # self.scans = set(self.scans)
        # self.instr_ids = set(self.instr_ids)
        # self.graphs = load_nav_graphs(self.scans)
        # self.distances = {}
        # for scan, G in self.graphs.items():  # compute all shortest paths
        #     self.distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def evaluate(self, path2inst):
        gts = {}
        res = {}
        for path_id, inst in path2inst.items():
            path_id = str(path_id)
            assert path_id in self.gt
            # There are three references， and 4 captions are too long
            gts[path_id] = [' '.join(self.tok.split_sentence(sent)) for sent in self.gt[path_id]]

            for i in range(gts[path_id].__len__()):
                if gts[path_id][i].__len__() > 500:
                    gts[path_id][i] = gts[path_id][i][:500]
            # if gts[path_id][0].__len__() > 500 or gts[path_id][1].__len__() > 500 or gts[path_id][2].__len__() > 500:
            #     import ipdb;ipdb.set_trace()
            if not self.use_clip16:
                res[path_id] = [' '.join([self.tok.index_to_word[word_id] for word_id in inst])]    # FIXME
            elif self.use_clip16:
                tmp = ''.join([self.tok.index_to_word[word_id] for word_id in inst])    # FIXME
                res[path_id] = [bytearray([self.tok.byte_decoder[c] for c in tmp]).decode('utf-8', errors="replace").replace('</w>', ' ')]
        # imgIds = self.params['image_id']
        # # imgIds = self.coco.getImgIds()
        # gts = {}
        # res = {}
        # for imgId in imgIds:
        #     gts[imgId] = self.coco.imgToAnns[imgId]
        #     res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        # print('tokenization...')
        # tokenizer = PTBTokenizer()
        # gts = tokenizer.tokenize(gts)
        # res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    # self.setImgToEvalImgs(scs, gts.keys(), m)
                    # print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                # self.setImgToEvalImgs(scores, gts.keys(), method)
                # print("%s: %0.3f"%(method, score))
        #self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]