import json
import logging
import math
import os
import sys
from io import open
from typing import Callable, List, Tuple
import numpy as np
import copy

import torch
from torch import nn
from torch import Tensor, device, dtype

from transformers import BertPreTrainedModel

logger = logging.getLogger(__name__)

BertLayerNorm = torch.nn.LayerNorm
from models.clip_model import Transformer


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}



class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None, future_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        if future_mask is not None:
            attention_scores = attention_scores.masked_fill(future_mask.unsqueeze(0).unsqueeze(1) != 0, float('-inf'))

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask



        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # recurrent vlnbert use attention scores
        outputs = (context_layer, attention_scores) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None, future_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask, future_mask=future_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, 
                                         None if head_mask is None else head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        # ================ if tie clip head ===================
        clip_size = 512
        self.dense = nn.Linear(config.hidden_size, clip_size)
        self.LayerNorm = BertLayerNorm(clip_size, eps=config.layer_norm_eps)
        # ================ if tie clip head ===================

        # ================ if tie bert head ===================
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # ================ if tie bert head ===================
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act


    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        # ================ if tie clip head ===================
        clip_size = 512
        self.decoder = nn.Linear(clip_size,
                            config.vocab_size,
                            bias=False)
        # ================ if tie clip head ===================

        # ================ if tie bert head ===================
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        # self.decoder = nn.Linear(config.hidden_size,
        #                          config.vocab_size,
        #                          bias=False)
        # ================ if tie bert head ===================

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertOutAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim =config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_scores

class BertXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.att = BertOutAttention(config, ctx_dim=ctx_dim)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output, attention_scores = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output, attention_scores

class LXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.no_lang_ca = config.no_lang_ca # do not update language embeds

        # Lang self-att and FFN layer
        self.lang_self_att = BertAttention(config)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)

        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

        # The cross attention layer
        self.visual_attention = BertXAttention(config)
        self.text_attention = BertXAttention(config)


    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask, initial_visn=None, initial_lang=None):
        if self.no_lang_ca:
            lang_att_output = lang_input
        else:
            lang_att_output, _ = self.text_attention(lang_input, initial_visn, ctx_att_mask=visn_attention_mask)    
        visn_att_output, _ = self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask)

        return lang_att_output, visn_att_output

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask, future_mask=None):
        # Self Attention
        if self.no_lang_ca:
            lang_att_output = (lang_input, )
        else:
            lang_att_output = self.lang_self_att(lang_input, lang_attention_mask, future_mask=future_mask)
        
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)
        return lang_att_output, visn_att_output

    def output_fc(self, lang_input, visn_input):
        # FC layers
        if not self.no_lang_ca:
            lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        if self.no_lang_ca:
            lang_output = lang_input
        else:
            lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask,
                      visn_feats, visn_attention_mask, initial_visn=None, initial_lang=None, future_mask=None,
                      visn_self_attn_mask=None):
        lang_att_output = lang_feats
        visn_att_output = visn_feats
        lang_att_output, visn_att_output = self.cross_att(lang_att_output, lang_attention_mask,
                                                          visn_att_output, visn_attention_mask, initial_visn=initial_visn, initial_lang=initial_lang)
        if visn_self_attn_mask is None:
            visn_self_attn_mask = visn_attention_mask
        lang_att_output, visn_att_output = self.self_att(lang_att_output, lang_attention_mask,
                                                         visn_att_output, visn_self_attn_mask, future_mask=future_mask)
        lang_output, visn_output = self.output_fc(lang_att_output[0], visn_att_output[0])

        return lang_output, visn_output

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class InputXAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.cross_att = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads, batch_first=True, dropout=config.attention_probs_dropout_prob)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, ctx_tensor, input_mask, ctx_att_mask=None):
        if torch.isnan(self.cross_att(input_tensor, ctx_tensor, ctx_tensor, key_padding_mask=ctx_att_mask.bool().squeeze(1).squeeze(1))[0]).any():
            import ipdb;ipdb.set_trace()
        output, _ = self.cross_att(input_tensor, ctx_tensor, ctx_tensor, key_padding_mask=ctx_att_mask.bool().squeeze(1).squeeze(1))
        attention_output = output + self.output(output, input_tensor)
        return attention_output
class InputSelfAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        self.self_att = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=config.num_attention_heads, batch_first=True, dropout=config.attention_probs_dropout_prob)
        self.output = BertSelfOutput(config)
    def forward(self, input_tensor, input_mask):
        output, _ = self.self_att(input_tensor, input_tensor, input_tensor, key_padding_mask=input_mask.bool().squeeze(1).squeeze(1))
        attention_output = output + self.output(output, input_tensor)
        return attention_output

class LxmertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_l_layers = config.num_l_layers
        self.num_r_layers = config.num_r_layers
        self.num_h_layers = config.num_h_layers
        self.num_x_layers = config.num_x_layers
        self.update_lang_bert = config.update_lang_bert
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_l_layers)]
        )
        if not self.update_lang_bert:
            for name, param in self.layer.named_parameters():
                param.requires_grad = False
        self.preceiver = InputXAttention(config)
        self.input_selfatt = InputSelfAttention(config)
        self.h_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_h_layers)]
        ) if self.num_h_layers > 0 else None
        self.r_layers = nn.ModuleList(
            [BertLayer(config) for _ in range(self.num_r_layers)]
        ) if self.num_r_layers > 0 else None
        self.x_layers = nn.ModuleList(
            [LXRTXLayer(config) for _ in range(self.num_x_layers)]
        )
        if config.use_clip16:
            self.hidden_size = config.image_feat_size
            vocab_size = config.vocab_size
            self.max_caption_length = 100           # NOTE match the pretraining

            self.clip_txt_encoder = Transformer(
                width=self.hidden_size,
                layers=12,          # NOTE 12
                heads=self.hidden_size // 64,
                attn_mask=self.build_attention_mask
            )
            self.clip_positional_embedding = nn.Parameter(torch.empty(self.max_caption_length, 
                                        self.hidden_size))
            self.clip_text_projection = nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
            self.clip_token_embedding = nn.Embedding(vocab_size, self.hidden_size)
            self.clip_ln_final = LayerNorm(self.hidden_size)
            self.post_linear = nn.Linear(self.hidden_size, config.hidden_size)

    def build_attention_mask(self, context_length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward_clip_text(self, txt_ids):
        x = self.clip_token_embedding(txt_ids)
        x = x + self.clip_positional_embedding[:x.size(1), :]
        x = x.permute(1, 0, 2)
        x = self.clip_txt_encoder(x)
        x = x.permute(1, 0, 2)
        caption_embeddings = self.clip_ln_final(x) @ self.clip_text_projection
        caption_embeddings = self.post_linear(caption_embeddings)
        return caption_embeddings

    def forward(self, txt_embeds, extended_txt_masks, hist_embeds,
                extended_hist_masks, img_embeds=None, extended_img_masks=None):
        # text encoding
        for layer_module in self.layer:
            temp_output = layer_module(txt_embeds, extended_txt_masks)
            txt_embeds = temp_output[0]

        if not self.update_lang_bert:
            txt_embeds = txt_embeds.detach()

        # image encoding
        if img_embeds is not None:
            if self.r_layers is not None:
                for layer_module in self.r_layers:
                    temp_output = layer_module(img_embeds, extended_img_masks)
                    img_embeds = temp_output[0]

        # history encoding
        if self.h_layers is not None:
            for layer_module in self.h_layers:
                temp_output = layer_module(hist_embeds, extended_hist_masks)
                hist_embeds = temp_output[0]
        hist_max_len = hist_embeds.size(1)
        
        # cross-modal encoding
        if img_embeds is None:
            hist_img_embeds = hist_embeds
            extended_hist_img_masks = extended_hist_masks
        else:
            hist_img_embeds = torch.cat([hist_embeds, img_embeds], 1)
            extended_hist_img_masks = torch.cat([extended_hist_masks, extended_img_masks], -1)
        
        initial_visn = hist_img_embeds
        initial_lang = txt_embeds
        for level, layer_module in enumerate(self.x_layers):
            txt_embeds, hist_img_embeds = layer_module(
                txt_embeds, extended_txt_masks, 
                hist_img_embeds, extended_hist_img_masks, initial_visn=initial_visn, initial_lang=initial_lang)

        hist_embeds = hist_img_embeds[:, :hist_max_len]
        if img_embeds is not None:
            img_embeds = hist_img_embeds[:, hist_max_len:]
        return txt_embeds, hist_embeds, img_embeds



class ImageEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
        self.ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # 0: non-navigable, 1: navigable, 2: stop
        self.nav_type_embedding = nn.Embedding(3, config.hidden_size)
        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, ang_feat, type_embeddings, nav_types=None):
        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_ang = self.ang_layer_norm(self.ang_linear(ang_feat))
        embeddings = transformed_im + transformed_ang + type_embeddings
        if nav_types is not None:
            nav_embeddings = self.nav_type_embedding(nav_types)
            embeddings = embeddings + nav_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ActionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
        self.ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, ang_feat, type_embeddings, step_embedding):
        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_ang = self.ang_layer_norm(self.ang_linear(ang_feat))
        embeddings = transformed_im + transformed_ang + type_embeddings.squeeze(1) + step_embedding
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class VisualEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pano_img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.pano_img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.pano_ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
        self.pano_ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # special type embedding for history
        self.type_embedding = nn.Embedding(1, config.hidden_size)
        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    @property
    def device(self):
        return self.pano_img_linear.weight.device

    def forward(self, pano_img_feats, pano_ang_feats,
                step_embedding=None, batch_size=None):
        type_ids = torch.zeros((batch_size, )).long().to(self.device)
        type_embeddings = self.type_embedding(type_ids).unsqueeze(1)    # [b,d]
        cls_embeddings = self.dropout(self.layer_norm(
            self.cls_token.expand(batch_size, -1, -1) + type_embeddings.squeeze(1)))
        

        if pano_img_feats is not None:
            batch_size, num_pano, _ = pano_img_feats.size()

            pano_embeddings = self.pano_img_layer_norm(self.pano_img_linear(pano_img_feats)) + \
                        self.pano_ang_layer_norm(self.pano_ang_linear(pano_ang_feats)) + \
                        type_embeddings

            pano_embeddings = pano_embeddings + step_embedding
            pano_embeddings = self.layer_norm(pano_embeddings)
            pano_embeddings = self.dropout(pano_embeddings)
        else:
            pano_embeddings = None

        return cls_embeddings, pano_embeddings

class NextActionPrediction(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Dropout(dropout_rate),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class NavCMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.img_embeddings = ImageEmbeddings(config)
        self.act_embeddings = ActionEmbeddings(config)
        self.vis_embeddings = VisualEmbeddings(config)
        self.step_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        self.encoder = LxmertEncoder(config)

        self.next_action = NextActionPrediction(config.hidden_size, config.pred_head_dropout_prob)
        
        self.use_clip16 = config.use_clip16

        self.pano_img_stop_embedding = nn.Parameter(torch.zeros(1, 1, self.config.image_feat_size))
        self.pano_ang_stop_embedding = nn.Parameter(torch.zeros(1, 1, self.config.angle_feat_size))

        self.lm_head = BertOnlyMLMHead(config) # NOTE

        self.init_weights()

    def forward(self, mode, txt_ids=None, txt_embeds=None, txt_masks=None,
                prev_action_img_fts=None, prev_action_ang_fts=None, action_masks=None, action_embeds=None,
                hist_pano_img_feats=None, hist_pano_ang_feats=None, hist_masks=None, hist_embeds=None,
                ob_img_feats=None, ob_ang_feats=None, ob_nav_types=None, ob_masks=None, ob_embeds=None, 
                ob_step_ids=None, token_type_ids=None, ob_cand_lens=None,
                is_train_caption=False, is_train_contrastive=False, 
                future_mask=None, graph_mask=None, history_mapper=None):
        # text embedding            
        if mode == 'language':
            ''' LXMERT language branch (in VLN only perform this at initialization) '''
            extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2)
            extended_txt_masks = extended_txt_masks.to(dtype=self.dtype)
            extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0

            if self.use_clip16:
                txt_embeds = self.encoder.forward_clip_text(txt_ids)
            else:
                txt_token_type_ids = torch.zeros_like(txt_ids)
                txt_embeds = self.embeddings(txt_ids, token_type_ids=txt_token_type_ids)
                for layer_module in self.encoder.layer:
                    temp_output = layer_module(txt_embeds, extended_txt_masks)
                    txt_embeds = temp_output[0]

            if self.config.fix_lang_embedding:
                txt_embeds = txt_embeds.detach()
            if self.config.no_lang_ca and not self.use_clip16: # run self-attn layers for lang
                all_txt_embeds = [txt_embeds]
                for level, layer_module in enumerate(self.encoder.x_layers):
                    lang_att_output = layer_module.lang_self_att(txt_embeds, extended_txt_masks)[0]
                    lang_inter_output = layer_module.lang_inter(lang_att_output)
                    lang_output = layer_module.lang_output(lang_inter_output, lang_att_output)
                    all_txt_embeds.append(lang_output)
                return all_txt_embeds
            return txt_embeds

        # history embedding per step
        if mode == 'history':
            if ob_step_ids is not None:
                step_embedding = self.step_embeddings(ob_step_ids)
            else:
                step_embedding = None
            if hist_pano_img_feats is not None:
                batch_size = hist_pano_img_feats.shape[0]
            else:
                batch_size = 1
            cls_embeds, hist_embeds = self.vis_embeddings(
                hist_pano_img_feats, hist_pano_ang_feats, step_embedding=step_embedding, batch_size=batch_size
            )
            if hist_embeds is None:
                return cls_embeds
            hist_embeds = hist_embeds.view(hist_embeds.size(0), -1, hist_embeds.size(-1))   # [b,36,d]
            return hist_embeds
        
        if mode == "action":
            step_embedding = self.step_embeddings(ob_step_ids)
            batch_size = prev_action_img_fts.shape[0]
            action_embeds = self.act_embeddings(prev_action_img_fts, prev_action_ang_fts,
                            self.token_type_embeddings(token_type_ids), step_embedding=step_embedding)
            return action_embeds

        if mode == "observation":
            assert ob_cand_lens is not None
            batch_size = ob_img_feats.shape[0]
            img_stop_embedding = self.pano_img_stop_embedding.squeeze(1).expand(batch_size, -1) # [b,d]
            ang_stop_embedding = self.pano_ang_stop_embedding.squeeze(1).expand(batch_size, -1)
            ob_cand_lens = [x - 1 for x in ob_cand_lens]
            ob_img_feats[torch.arange(batch_size), ob_cand_lens] = img_stop_embedding
            ob_ang_feats[torch.arange(batch_size), ob_cand_lens] = ang_stop_embedding
            ob_embeds = self.img_embeddings(ob_img_feats, ob_ang_feats, 
                            self.token_type_embeddings(token_type_ids), nav_types=ob_nav_types)
            return ob_embeds

        # cross-modal encoding per step
        elif mode == 'visual':
            ''' LXMERT visual branch'''
            # history embedding
            views = 36
            extended_hist_masks = hist_masks.repeat_interleave(views, -1)
            extended_hist_masks = hist_masks.unsqueeze(1).unsqueeze(2)
            extended_hist_masks = extended_hist_masks.to(dtype=self.dtype)
            extended_hist_masks = (1.0 - extended_hist_masks) * -10000.0

            if action_masks is not None:
                extended_action_masks = action_masks.unsqueeze(1).unsqueeze(2)        # NOTE
                extended_action_masks = extended_action_masks.to(dtype=self.dtype)
            else:
                extended_action_masks = torch.tensor([]).to(hist_masks.device)
                action_embeds = torch.tensor([]).to(hist_masks.device)
            if ob_masks is not None:
                extended_ob_masks = ob_masks.unsqueeze(1).unsqueeze(2)
                extended_ob_masks = extended_ob_masks.to(dtype=self.dtype)
            else:
                ob_embeds = torch.tensor([]).to(hist_masks.device)
                extended_ob_masks = torch.tensor([]).to(hist_masks.device)
                
            extended_txt_masks = txt_masks.unsqueeze(1).unsqueeze(2)
            extended_txt_masks = extended_txt_masks.to(dtype=self.dtype)
            extended_txt_masks = (1.0 - extended_txt_masks) * -10000.0

            # multi-modal encoding
            if len(action_embeds.shape) == 1:
                hist_action_max_len = 0
            else:
                hist_action_max_len = action_embeds.size(1)

            if ob_embeds is not None:
                action_img_embeds = torch.cat([action_embeds, ob_embeds], 1)
                extended_action_img_masks = torch.cat([extended_action_masks, extended_ob_masks], -1)
            else:
                action_img_embeds = action_embeds
                extended_action_img_masks = extended_action_masks

            visn_self_attn_mask = extended_action_img_masks.transpose(-1, -2) * extended_action_img_masks

            if graph_mask is not None:
                graph_max_size = graph_mask.size(-1)
                visn_self_attn_mask[:, 0, :graph_max_size, :graph_max_size] *= graph_mask

            visn_self_attn_mask = (1.0 - visn_self_attn_mask) * -10000.0
            extended_action_img_masks = (1.0 - extended_action_img_masks) * -10000.0
                
            action_img_embeds = self.encoder.preceiver(action_img_embeds, hist_embeds, extended_action_img_masks, extended_hist_masks)
            action_img_embeds = self.encoder.input_selfatt(action_img_embeds, extended_action_img_masks)

            if is_train_contrastive:
                if self.config.use_clip16:
                    text_g = txt_embeds[torch.arange(txt_embeds.shape[0]), txt_masks.sum(-1)-1]
                else:
                    text_g = txt_embeds[:, 0]
                action_g = ((action_masks.bool()).float().unsqueeze(-1) * action_img_embeds).sum(1) / (action_masks.bool()).sum(-1, keepdim=True)
                action_g = action_g / action_g.norm(dim=-1, keepdim=True)
                text_g = text_g / text_g.norm(dim=-1, keepdim=True)
                logit_scale = 100.0
                sim = logit_scale * torch.matmul(text_g, action_g.t())
                return sim

            initial_visn = action_img_embeds
            initial_lang = txt_embeds
            for l, layer_module in enumerate(self.encoder.x_layers):
                txt_embeds, action_img_embeds = layer_module(
                    txt_embeds, extended_txt_masks, 
                    action_img_embeds, extended_action_img_masks, 
                    initial_visn=initial_visn, initial_lang=initial_lang, future_mask=future_mask,
                    visn_self_attn_mask=visn_self_attn_mask
                )

            if hist_action_max_len == 0:
                action_embeds = None
            else:
                action_embeds = action_img_embeds[:, :hist_action_max_len]
            ob_embeds = action_img_embeds[:, hist_action_max_len:]

            # for gas
            if action_embeds is not None:
                if history_mapper is not None:
                    hist_cand_embeds = history_mapper(action_embeds)
                else:
                    hist_cand_embeds = action_embeds        # [b,t,d]
                cand_embeds = torch.cat([hist_cand_embeds, ob_embeds], dim=1)   # [b,t+43, d]
            else:
                cand_embeds = ob_embeds

            # =============================== for caption ============================================
            if is_train_caption:
                pred = self.lm_head(txt_embeds)
                return pred
            # =============================== for caption ============================================

            if self.config.no_lang_ca:
                act_logits = self.next_action(cand_embeds).squeeze(-1)
            else:
                if self.config.act_pred_token == 'ob_txt':
                    if self.config.use_clip16:
                        act_logits = self.next_action(ob_embeds * txt_embeds[torch.arange(txt_embeds.shape[0]), txt_masks.sum(-1)-1].unsqueeze(1)).squeeze(-1)    
                    else:
                        act_logits = self.next_action(ob_embeds * txt_embeds[:, :1]).squeeze(-1)
                elif self.config.act_pred_token == 'ob':
                    # act_logits = self.next_action(ob_embeds).squeeze(-1)
                    act_logits = self.next_action(cand_embeds).squeeze(-1)
                elif self.config.act_pred_token == 'ob_hist':
                    act_logits = self.next_action(ob_embeds * hist_embeds[:, :1]).squeeze(-1)
                elif self.config.act_pred_token == 'ob_txt_hist':
                    if self.config.use_clip16:
                        act_logits = self.next_action(ob_embeds * (txt_embeds[torch.arange(txt_embeds.shape[0]), txt_masks.sum(-1)-1] + hist_embeds[:, :1])).squeeze(-1)
                    else:    
                        act_logits = self.next_action(ob_embeds * (txt_embeds[:, :1] + hist_embeds[:, :1])).squeeze(-1)

            padding = torch.zeros(act_logits.shape[0], 1).to(act_logits.device)
            act_logits = torch.cat([padding, act_logits], dim=1)

            padding = torch.zeros(ob_embeds.shape[0], 1, ob_embeds.shape[-1]).to(ob_embeds.device)
            if action_embeds is None:
                action_embeds = padding
            else:
                action_embeds = torch.cat([padding, action_embeds], dim=1)

            act_logits[:, hist_action_max_len+1:].masked_fill_(ob_nav_types == 0, -float('inf'))
            if hist_action_max_len > 0:
                act_logits[:, 1:hist_action_max_len+1].masked_fill_(action_masks == False, -float('inf'))
            act_logits[:, 0] = -float('inf')

            return act_logits, action_embeds, txt_embeds, hist_embeds, ob_embeds















class HistoryEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
        self.ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        
        self.position_embeddings = nn.Embedding(config.max_action_steps, config.hidden_size)
        # special type embedding for history
        self.type_embedding = nn.Embedding(1, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.hist_enc_pano = config.hist_enc_pano
        if config.hist_enc_pano:
            self.pano_img_linear = nn.Linear(config.image_feat_size, config.hidden_size)
            self.pano_img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            self.pano_ang_linear = nn.Linear(config.angle_feat_size, config.hidden_size)
            self.pano_ang_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
            pano_enc_config = copy.copy(config)
            pano_enc_config.num_hidden_layers = config.num_h_pano_layers
            self.pano_encoder = BertEncoder(pano_enc_config)
        else:
            self.pano_encoder = None

    def forward(self, img_feats, ang_feats, pos_ids, 
                pano_img_feats=None, pano_ang_feats=None, is_train_caption=False):
        '''Args:
        - img_feats: (batch_size, dim_feat)
        - pos_ids: (batch_size, )
        - pano_img_feats: (batch_size, pano_len, dim_feat)
        '''
        if not is_train_caption:
            device = next(iter(self.parameters())).device
            if img_feats is not None:
                batch_size = img_feats.size(0)
            else:
                batch_size = 1

            type_ids = torch.zeros((batch_size, )).long().to(device)
            type_embeddings = self.type_embedding(type_ids)

            if img_feats is None:
                cls_embeddings = self.dropout(self.layer_norm(
                    self.cls_token.expand(batch_size, -1, -1)[:, 0] + type_embeddings))
                return cls_embeddings

            # history embedding per step
            embeddings = self.img_layer_norm(self.img_linear(img_feats)) + \
                        self.ang_layer_norm(self.ang_linear(ang_feats)) + \
                        type_embeddings
            if pos_ids is not None:
                embeddings = embeddings + self.position_embeddings(pos_ids)


            if self.pano_encoder is not None:
                pano_embeddings = self.pano_img_layer_norm(self.pano_img_linear(pano_img_feats)) + \
                                self.pano_ang_layer_norm(self.pano_ang_linear(pano_ang_feats))
                pano_embeddings = self.dropout(pano_embeddings)
                # TODO: mask is always True
                batch_size, pano_len, _ = pano_img_feats.size()
                extended_pano_masks = torch.zeros(batch_size, pano_len).float().to(device).unsqueeze(1).unsqueeze(2)
                pano_embeddings = self.pano_encoder(pano_embeddings, extended_pano_masks)[0]
                pano_embeddings = torch.mean(pano_embeddings, 1)

                embeddings = embeddings + pano_embeddings

            embeddings = self.layer_norm(embeddings)
            embeddings = self.dropout(embeddings)
            return embeddings
        else:
            device = next(iter(self.parameters())).device
            if img_feats is not None:
                batch_size = img_feats.size(0)
            else:
                batch_size = 1
            type_ids = torch.zeros((batch_size, 1)).long().to(device)
            type_embeddings = self.type_embedding(type_ids)

            cls_embeddings = self.dropout(self.layer_norm(
                self.cls_token.expand(batch_size, -1, -1) + type_embeddings))

            if img_feats is not None:
                embeddings = self.img_layer_norm(self.img_linear(img_feats)) + \
                            self.ang_layer_norm(self.ang_linear(ang_feats)) + \
                            type_embeddings

                if self.pano_encoder is not None:
                    batch_size, num_steps, num_pano, _ = pano_img_feats.size()
                    pano_img_feats = pano_img_feats.view(batch_size*num_steps, num_pano, -1)
                    pano_ang_feats = pano_ang_feats.view(batch_size*num_steps, num_pano, -1)
                    pano_embeddings = self.pano_img_layer_norm(self.pano_img_linear(pano_img_feats)) + \
                                    self.pano_ang_layer_norm(self.pano_ang_linear(pano_ang_feats))
                    # assume pano all exists
                    ext_pano_masks = torch.zeros(batch_size*num_steps, num_pano, dtype=torch.float).to(device).unsqueeze(1).unsqueeze(2)
                    pano_embeddings = self.pano_encoder(pano_embeddings, ext_pano_masks)[0]
                    
                    pano_embeddings = pano_embeddings.view(batch_size, num_steps, num_pano, -1)
                    pano_embeddings = torch.mean(pano_embeddings, 2)
                    
                    embeddings = embeddings + pano_embeddings

                if pos_ids is not None:
                    embeddings = embeddings + self.position_embeddings(pos_ids)
                    embeddings = self.layer_norm(embeddings)
                    embeddings = self.dropout(embeddings)
            else:
                embeddings = None
            hist_embeds = torch.cat([cls_embeddings, embeddings], dim=1)
            return hist_embeds