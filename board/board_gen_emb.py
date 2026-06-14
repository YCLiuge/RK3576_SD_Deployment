#!/usr/bin/env python3
"""Generate Dreamshaper embeddings using board's tokenizers library."""
import numpy as np, os, json
MODEL = '/home/cat/lcm_sd'
TOK_DIR = f'{MODEL}/tokenizer'

# Load tokenizer directly from files
from tokenizers import Tokenizer
# Try loading from tokenizer.json first
tok_json = f'{TOK_DIR}/tokenizer.json'
if os.path.exists(tok_json):
    tok = Tokenizer.from_file(tok_json)
else:
    # Build from vocab/merges
    from tokenizers import models, pre_tokenizers, decoders
    with open(f'{TOK_DIR}/vocab.json') as f:
        vocab = json.load(f)
    with open(f'{TOK_DIR}/merges.txt') as f:
        merges_lines = f.read().splitlines()
    merges = [tuple(line.split()) for line in merges_lines if line.strip() and len(line.split()) == 2]
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    
    # Read special tokens from config
    with open(f'{TOK_DIR}/tokenizer_config.json') as f:
        cfg = json.load(f)
    bos = cfg.get('bos_token', '<|startoftext|>')
    eos = cfg.get('eos_token', '<|endoftext|>')
    pad_id = 49407  # eos token id
    
    # Set padding and truncation
    tok.enable_padding(pad_id=pad_id, length=77)
    tok.enable_truncation(max_length=77)

PROMPT = 'masterpiece, best quality, 1girl, cat girl, cat ears, white hair, blue eyes, solo, cute'
NEG = 'lowres, bad anatomy, bad hands, worst quality, low quality'

pos_enc = tok.encode(PROMPT)
neg_enc = tok.encode(NEG)
print(f'Token lengths: pos={len(pos_enc.ids)}, neg={len(neg_enc.ids)}')

pos_ids = np.array([pos_enc.ids], dtype=np.int64)
neg_ids = np.array([neg_enc.ids], dtype=np.int64)

# Run Dreamshaper text encoder RKNN
from rknnlite.api import RKNNLite
te = RKNNLite()
te.load_rknn(f'{MODEL}/text_encoder/model.rknn')
te.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
pos_emb = te.inference(inputs=[pos_ids])[0]
neg_emb = te.inference(inputs=[neg_ids])[0]
te.release()

print(f'Pos: {pos_emb.shape} std={pos_emb.std():.4f} mean={pos_emb.mean():.4f}')
print(f'Neg: {neg_emb.shape} std={neg_emb.std():.4f} mean={neg_emb.mean():.4f}')
print(f'|pos-neg|: {np.abs(pos_emb - neg_emb).mean():.4f}')

np.save('/home/cat/pos_emb.npy', pos_emb.astype(np.float32))
np.save('/home/cat/neg_emb.npy', neg_emb.astype(np.float32))
print('Saved Dreamshaper embeddings!')
