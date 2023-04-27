import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.misc import length2mask

from models.vlnbert_init_lana import get_vlnbert_models

class VLNBertCMT(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.feat_dropout)
        self.hidden_size = 768      # NOTE for clip
        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.hidden_size),
            nn.LayerNorm(self.hidden_size, eps=1e-12)
        )
        self.history_mapper = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        ) if not args.no_hist_mapping else None
        
    def forward(self, mode, txt_ids=None, txt_masks=None, txt_embeds=None, 
                prev_action_img_fts=None, prev_action_ang_fts=None, 
                hist_pano_img_feats=None, hist_pano_ang_feats=None,
                hist_embeds=None, hist_lens=None, action_embeds=None, action_lens=None,
                ob_step=None,
                ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None, ob_embeds=None, ob_cand_lens=None, 
                ob_masks=None, return_states=False, future_mask=None, is_train_caption=False, is_train_contrastive=False,
                ob_position=None, position=None, graph_mask=None
                ):

        if mode == 'language':
            encoded_sentence = self.vln_bert(mode, txt_ids=txt_ids, txt_masks=txt_masks, future_mask=future_mask)
            return encoded_sentence

        elif mode == 'history':
            # history inputs per steps
            if hist_pano_img_feats is not None:
                hist_pano_img_feats = self.drop_env(hist_pano_img_feats)
            if ob_step is not None:
                ob_step_ids = torch.LongTensor([ob_step]).cuda()
            else:
                ob_step_ids = None
            hist_embeds = self.vln_bert(mode, 
                                hist_pano_img_feats=hist_pano_img_feats, hist_pano_ang_feats=hist_pano_ang_feats,
                                ob_step_ids=ob_step_ids)
            return hist_embeds

        elif mode == "action":
            assert prev_action_img_fts is not None and prev_action_ang_fts is not None
            if prev_action_img_fts is not None:
                prev_action_img_fts = self.drop_env(prev_action_img_fts)
            batch_size = prev_action_ang_fts.shape[0]
            action_token_type_ids = torch.zeros(batch_size, 1).long().to(prev_action_img_fts.device)
            if ob_step is not None:
                ob_step_ids = torch.LongTensor([ob_step]).cuda()
            else:
                ob_step_ids = None
            action_embeds = self.vln_bert(mode, 
                                prev_action_img_fts=prev_action_img_fts, prev_action_ang_fts=prev_action_ang_fts, 
                                token_type_ids=action_token_type_ids, ob_step_ids=ob_step_ids)
            if position is not None:
                position_emb = self.position_encoder(position)
                action_embeds = action_embeds + position_emb
            return action_embeds

        elif mode == "observation":
            assert ob_img_feats is not None 
            assert ob_ang_feats is not None
            batch_size = ob_img_feats.shape[0]
            if ob_img_feats is not None:
                ob_img_feats = self.drop_env(ob_img_feats)
            ob_token_type_ids = torch.ones(batch_size, 1).long().to(ob_img_feats.device)
            ob_embeds = self.vln_bert(mode, ob_img_feats=ob_img_feats, ob_ang_feats=ob_ang_feats, 
                                    token_type_ids=ob_token_type_ids, ob_nav_types=ob_nav_types, ob_cand_lens=ob_cand_lens)
            if ob_position is not None:
                ob_position_feat = self.position_encoder(ob_position)
                ob_embeds = ob_embeds + ob_position_feat
            return ob_embeds

        elif mode == 'visual':
            if is_train_caption or is_train_contrastive:
                padding = torch.ones((len(hist_lens), 1), dtype=torch.bool).to(hist_embeds[0].device)
                hist_masks = length2mask(hist_lens, size=len(hist_embeds) - 1).logical_not()
                hist_masks = hist_masks.repeat_interleave(hist_embeds[1].shape[1], -1)
                hist_masks = torch.cat([padding, hist_masks], dim=1)
                hist_embeds = torch.cat(hist_embeds, dim=1)
            else:
                if len(hist_embeds) == 1:       # cls
                    hist_embeds = torch.cat(hist_embeds, dim=1)
                    hist_masks = torch.ones((len(hist_lens), 1), dtype=torch.bool).to(ob_embeds.device)
                else:
                    padding = torch.ones((len(hist_lens), 1), dtype=torch.bool).to(ob_embeds.device)
                    hist_masks = length2mask(hist_lens, size=len(hist_embeds) - 1).logical_not() 
                    hist_masks = hist_masks.repeat_interleave(hist_embeds[1].shape[1], -1)
                    hist_masks = torch.cat([padding, hist_masks], dim=1)
                    hist_embeds = torch.cat(hist_embeds, dim=1)
            
            if action_embeds == [] or action_embeds is None:
                action_embeds = None
                action_masks = None
            else:
                action_embeds = torch.cat(action_embeds, dim=1)
                action_masks = length2mask(action_lens, size=action_embeds.size(1)).logical_not()

            if is_train_caption or is_train_contrastive:
                pred = self.vln_bert(
                    mode, txt_embeds=txt_embeds, txt_masks=txt_masks,
                    hist_embeds=hist_embeds, hist_masks=hist_masks,
                    action_embeds=action_embeds, action_masks=action_masks,
                    ob_embeds=ob_embeds, ob_masks=ob_masks, ob_nav_types=ob_nav_types,
                    future_mask=future_mask, is_train_caption=is_train_caption, is_train_contrastive=is_train_contrastive)
                return pred
            else:
                act_logits, action_embeds, txt_embeds, hist_embeds, ob_embeds = self.vln_bert(
                    mode, txt_embeds=txt_embeds, txt_masks=txt_masks,
                    hist_embeds=hist_embeds, hist_masks=hist_masks,
                    action_embeds=action_embeds, action_masks=action_masks,
                    ob_embeds=ob_embeds, ob_masks=ob_masks, ob_nav_types=ob_nav_types,
                    future_mask=future_mask, is_train_caption=is_train_caption, is_train_contrastive=is_train_contrastive,
                    graph_mask=graph_mask, history_mapper=self.history_mapper)

            if return_states:
                if self.args.no_lang_ca:
                    if action_embeds.shape[1] == 1:
                        if self.args.use_clip16:
                            states = txt_embeds[torch.arange(txt_embeds.shape[0]), txt_masks.sum(-1)-1]  # [CLS]    
                        else:
                            states = txt_embeds[:, 0]
                    else:
                        valid_num = action_masks.sum(-1, keepdim=True)
                        hist_mean = torch.sum(action_embeds[:,1:] * action_masks.unsqueeze(-1), dim=1) / valid_num
                        states = hist_mean
                else:
                    # states = txt_embeds[torch.arange(txt_embeds.shape[0]), txt_masks.sum(-1)-1]    
                    if action_embeds.shape[1] == 1:
                        states = txt_embeds[torch.arange(txt_embeds.shape[0]), txt_masks.sum(-1)-1]  # [CLS]    
                    else:
                        valid_num = action_masks.sum(-1, keepdim=True)
                        hist_mean = torch.sum(action_embeds[:,1:] * action_masks.unsqueeze(-1), dim=1) / valid_num
                        states = txt_embeds[torch.arange(txt_embeds.shape[0]), txt_masks.sum(-1)-1] * hist_mean  # [CLS]
                return act_logits, states, action_embeds
            return act_logits, action_embeds



















class VLNBertCausalCMT(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.feat_dropout)
        
    def forward(
        self, mode, txt_ids=None, txt_masks=None, txt_embeds=None,
        hist_img_feats=None, hist_ang_feats=None, 
        hist_pano_img_feats=None, hist_pano_ang_feats=None, ob_step=0,
        new_hist_embeds=None, new_hist_masks=None,
        prefix_hiddens=None, prefix_masks=None,
        ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None,
        ob_masks=None, return_states=False, batch_size=None,
    ):

        if mode == 'language':
            encoded_sentence = self.vln_bert(mode, txt_ids=txt_ids, txt_masks=txt_masks)
            return encoded_sentence

        elif mode == 'history':
            if ob_step == 0:
                hist_step_ids = torch.arange(1).long()
            else:
                hist_step_ids = torch.arange(2).long() + ob_step - 1
            hist_step_ids = hist_step_ids.unsqueeze(0)

            # history inputs per step
            if hist_img_feats is not None:
                hist_img_feats = self.drop_env(hist_img_feats)
            if hist_pano_img_feats is not None:
                hist_pano_img_feats = self.drop_env(hist_pano_img_feats)

            hist_embeds = self.vln_bert(
                mode, hist_img_feats=hist_img_feats, 
                hist_ang_feats=hist_ang_feats, 
                hist_pano_img_feats=hist_pano_img_feats, 
                hist_pano_ang_feats=hist_pano_ang_feats,
                hist_step_ids=hist_step_ids,
                batch_size=batch_size
            )
            return hist_embeds

        elif mode == 'visual':
            ob_img_feats = self.drop_env(ob_img_feats)
            
            act_logits, prefix_hiddens, states = self.vln_bert(
                mode, txt_embeds=txt_embeds, txt_masks=txt_masks,
                ob_img_feats=ob_img_feats, ob_ang_feats=ob_ang_feats, 
                ob_nav_types=ob_nav_types, ob_masks=ob_masks,
                new_hist_embeds=new_hist_embeds, new_hist_masks=new_hist_masks,
                prefix_hiddens=prefix_hiddens, prefix_masks=prefix_masks
            )

            if return_states:
                return act_logits, prefix_hiddens, states
            return (act_logits, prefix_hiddens)


class VLNBertMMT(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.feat_dropout)
        
    def forward(
        self, mode, txt_ids=None, txt_masks=None, txt_embeds=None, 
        hist_img_feats=None, hist_ang_feats=None, 
        hist_pano_img_feats=None, hist_pano_ang_feats=None,
        hist_embeds=None, hist_masks=None, ob_step=None,
        ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None, 
        ob_masks=None, return_states=False, batch_size=None,
        prefix_embeds=None, prefix_masks=None
    ):

        if mode == 'language':
            encoded_sentence = self.vln_bert(mode, txt_ids=txt_ids, txt_masks=txt_masks)
            return encoded_sentence

        elif mode == 'history':
            if hist_img_feats is None:
                # only encode [sep] token
                hist_step_ids = torch.zeros((batch_size, 1), dtype=torch.long)
            else:
                # encode the new observation and [sep]
                hist_step_ids = torch.arange(2).long().expand(batch_size, -1) + ob_step - 1
            
            # history inputs per step
            if hist_img_feats is not None:
                hist_img_feats = self.drop_env(hist_img_feats)
            if hist_pano_img_feats is not None:
                hist_pano_img_feats = self.drop_env(hist_pano_img_feats)
            
            new_hist_embeds = self.vln_bert(
                mode, hist_img_feats=hist_img_feats, 
                hist_ang_feats=hist_ang_feats, 
                hist_step_ids=hist_step_ids,
                hist_pano_img_feats=hist_pano_img_feats, 
                hist_pano_ang_feats=hist_pano_ang_feats,
                batch_size=batch_size,
            )
            return new_hist_embeds

        elif mode == 'visual':
            ob_img_feats = self.drop_env(ob_img_feats)
            
            outs = self.vln_bert(
                mode, txt_embeds=txt_embeds, txt_masks=txt_masks,
                hist_embeds=hist_embeds, hist_masks=hist_masks,
                ob_img_feats=ob_img_feats, ob_ang_feats=ob_ang_feats, 
                ob_nav_types=ob_nav_types, ob_masks=ob_masks,
                prefix_embeds=prefix_embeds, prefix_masks=prefix_masks
            )

            act_logits, hist_state = outs[:2]

            if return_states:
                return (act_logits, hist_state) + outs[2:]

            return (act_logits, ) + outs[2:]


class VLNBertCMT3(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('\nInitalizing the VLN-BERT model ...')
        self.args = args

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.drop_env = nn.Dropout(p=args.feat_dropout)
        
    def forward(
        self, mode, txt_ids=None, txt_masks=None,
        hist_img_feats=None, hist_ang_feats=None, 
        hist_pano_img_feats=None, hist_pano_ang_feats=None, ob_step=0,
        ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None,
        ob_masks=None, return_states=False, 
        txt_embeds=None, hist_in_embeds=None, hist_out_embeds=None, hist_masks=None
    ):

        if mode == 'language':
            encoded_sentence = self.vln_bert(mode, txt_ids=txt_ids, txt_masks=txt_masks)
            return encoded_sentence

        elif mode == 'history':
            if ob_step == 0:
                hist_step_ids = torch.arange(1).long()
            else:
                hist_step_ids = torch.arange(2).long() + ob_step - 1
            hist_step_ids = hist_step_ids.unsqueeze(0)

            # history inputs per step
            if hist_img_feats is not None:
                hist_img_feats = self.drop_env(hist_img_feats)
            if hist_pano_img_feats is not None:
                hist_pano_img_feats = self.drop_env(hist_pano_img_feats)

            hist_in_embeds, hist_out_embeds = self.vln_bert(
                mode, hist_img_feats=hist_img_feats, 
                hist_ang_feats=hist_ang_feats, 
                hist_pano_img_feats=hist_pano_img_feats, 
                hist_pano_ang_feats=hist_pano_ang_feats,
                hist_step_ids=hist_step_ids,
                hist_in_embeds=hist_in_embeds,
                hist_out_embeds=hist_out_embeds,
                hist_masks=hist_masks
            )
            return hist_in_embeds, hist_out_embeds

        elif mode == 'visual':
            ob_img_feats = self.drop_env(ob_img_feats)
            
            act_logits, states = self.vln_bert(
                mode, txt_embeds=txt_embeds, txt_masks=txt_masks,
                hist_out_embeds=hist_out_embeds, hist_masks=hist_masks,
                ob_img_feats=ob_img_feats, ob_ang_feats=ob_ang_feats, 
                ob_nav_types=ob_nav_types, ob_masks=ob_masks,
            )

            if return_states:
                return act_logits, states
            return (act_logits, )


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()
