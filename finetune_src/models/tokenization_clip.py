import encodings
import gzip
import torch
import html
import os
from functools import lru_cache

import ftfy
import regex as re
import numpy as np
import copy
import string

@lru_cache()
def default_bpe():
    return os.path.join("clip_tokenizer", "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # Split on any non-alphanumeric character
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        # vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        vocab.extend(['<BOS>', '<EOS>', '<UNK>', '<MSK>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        # self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.cache = {'<BOS>': '<BOS>', '<EOS>': '<EOS>', '<UNK>': '<UNK>', '<MSK>': '<MSK>'}
        # self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
        self.pat = re.compile(r"""<\|BOS\|>|<\|EOS\|>|<\|UNK\|>|<\|MSK\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

        self.vocab = self.encoder
        self.word_to_index = copy.deepcopy(self.encoder)
        self.index_to_word = copy.deepcopy(self.decoder)
        self.word_to_index['<PAD>'] = 0 # FIXME not elegant
        print(f"vocab size is {self.vocab_size()}")

    def vocab_size(self):
        return len(self.vocab)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = [self.encoder["<BOS>"]]
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        bpe_tokens.append(self.encoder["<EOS>"])
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

    def tokenize(self, text):
        tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return [self.encoder[bpe_token] for bpe_token in tokens]

    def __call__(self, texts, return_tensors='pt', padding=True, truncation=True):
        """
            Returns the tokenized representation of given input string(s)
            Parameters
            ----------
            texts : Union[str, List[str]]
                An input string or a list of input strings to tokenize
            context_length : int
                The context length to use; all CLIP models use 77 as the context length

            remaining params are just to have same interface with huggingface tokenizer.
            They don't do much. 
            Returns
            -------
            A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        context_length = 100     # NOTE 100 in VLN task, cause one token length is 97, one is 121
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.encoder["<BOS>"]
        eot_token = self.encoder["<EOS>"]
        all_tokens = [[sot_token] + self.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                # import ipdb;ipdb.set_trace()
                # raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
                tokens = tokens[:context_length - 1]
                tokens.append(self.vocab["<EOS>"])      # NOTE 
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def encode_sentence(self, texts):
        # str -> numpy   for only one sentence!!!!
        context_length = 100     # NOTE 100 in VLN task, cause one token length is 97, one is 121
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.encoder["<BOS>"]
        eot_token = self.encoder["<EOS>"]
        all_tokens = [[sot_token] + self.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                # import ipdb;ipdb.set_trace()
                # raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
                tokens = tokens[:context_length]
                # tokens.append(self.vocab["<|endoftext|>"])      # NOTE  no need to add [eos]
            result[i, :len(tokens)] = torch.tensor(tokens)
            result = result.squeeze(0)      # [context_length]

        return np.array(result)
    
    def decode_sentence(self, tokens, length=None):
        # numpy -> str
        # text = ''.join([self.decoder[token] for token in tokens])
        # text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        text = []
        if length is not None:
            tokens = tokens[:length]
        for ix in tokens:
            if ix == 0:
                break
            else:
                text.append(self.decoder[ix])
        text = ''.join([t for t in text])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
    
    def shrink(self, inst):
        # numpy -> numpy
        if len(inst) == 0:
            return inst
        end = np.argmax(np.array(inst) == self.encoder["<EOS>"])
        if len(inst) > 1 and inst[0] == self.encoder["<BOS>"]:
            start = 1
        else:
            start = 0
        return inst[start: end]

    @staticmethod
    def split_sentence(sentence):
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        for word in [s.strip().lower() for s in SimpleTokenizer.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks
