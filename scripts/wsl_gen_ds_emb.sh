#!/bin/bash
export PATH=/usr/bin:/home/lzy0x91f/.local/bin:$PATH
python3 << 'PYEOF'
import numpy as np, os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

PROMPT = 'masterpiece, best quality, 1girl, cat girl, cat ears, white hair, blue eyes, solo, cute'
NEG = 'lowres, bad anatomy, bad hands, worst quality, low quality'

# Use Dreamshaper's own ONNX text encoder
import onnxruntime as ort
sess = ort.InferenceSession('/home/lzy0x91f/lcm_sd/text_encoder/model.onnx', providers=['CPUExecutionProvider'])

# Use Dreamshaper's tokenizer files  
from tokenizers import Tokenizer
tok = Tokenizer.from_file('/home/lzy0x91f/lcm_sd/tokenizer/tokenizer.json') if os.path.exists('/home/lzy0x91f/lcm_sd/tokenizer/tokenizer.json') else None

if tok is None:
    # Build tokenizer from vocab/merges
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from tokenizers.pre_tokenizers import Whitespace
    tok = Tokenizer(models.BPE(
        vocab='/home/lzy0x91f/lcm_sd/tokenizer/vocab.json',
        merges='/home/lzy0x91f/lcm_sd/tokenizer/merges.txt',
    ))
    tok.pre_tokenizer = Whitespace()

def encode(text):
    enc = tok.encode(text)
    ids = enc.ids[:77]
    pad = max(0, 77 - len(ids))
    return np.array([ids + [0]*pad], dtype=np.int64)

pos_ids = encode(PROMPT)
neg_ids = encode(NEG)

pos_emb = sess.run(None, {'input_ids': pos_ids})[0]
neg_emb = sess.run(None, {'input_ids': neg_ids})[0]

print(f'Pos: {pos_emb.shape} std={pos_emb.std():.4f} mean={pos_emb.mean():.4f}')
print(f'Neg: {neg_emb.shape} std={neg_emb.std():.4f} mean={neg_emb.mean():.4f}')
print(f'Diff: {np.abs(pos_emb - neg_emb).mean():.4f}')

# Save for board
OUT = '/mnt/d/Electronic_Design/Workspace/Projects/RK_Series/Lubancat3'
np.save(os.path.join(OUT, 'pos_emb.npy'), pos_emb.astype(np.float32))
np.save(os.path.join(OUT, 'neg_emb.npy'), neg_emb.astype(np.float32))
print(f'Saved to {OUT}/')
PYEOF
