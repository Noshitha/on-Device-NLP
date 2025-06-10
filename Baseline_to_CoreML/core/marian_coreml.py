# core/marian_coreml.py
import coremltools as ct
import numpy as np
import json
import os
import torch
from pathlib import Path

class MarianCoreML:
    def __init__(self, model_dir):
        """Initialize the Marian Core ML model with separate encoder/decoder"""
        self.model_dir = Path(model_dir)

        # Load Core ML models
        print("Loading Core ML models...")
        self.encoder = ct.models.MLModel(str(self.model_dir / "encoder.mlpackage"))
        self.decoder = ct.models.MLModel(str(self.model_dir / "decoder.mlpackage"))

        # Determine the fixed sequence lengths expected by each .mlpackage
        enc_spec = self.encoder.get_spec()
        self.encoder_seq_len = list(enc_spec.description.input[0]
                                    .type.multiArrayType.shape)[1]
        dec_spec = self.decoder.get_spec()
        self.decoder_seq_len = list(dec_spec.description.input[0]
                                    .type.multiArrayType.shape)[1]

        # Load configuration
        with open(self.model_dir / "config.json", 'r') as f:
            self.config = json.load(f)

        # Load final projection weights (detach so .numpy() works)
        print("Loading projection weights...")
        lm_weight = (torch.load(self.model_dir / "lm_weight.bin", map_location='cpu')
                        .detach()
                        .cpu()
                        .numpy())
        lm_bias = (torch.load(self.model_dir / "lm_bias.bin", map_location='cpu')
                        .detach()
                        .cpu()
                        .numpy())

        self.lm_weight = lm_weight  # shape: (vocab, hidden)
        self.lm_bias   = lm_bias    # shape: (1, vocab)

        # Model parameters
        self.pad_token_id          = self.config['pad_token_id']
        self.eos_token_id          = self.config['eos_token_id']
        self.decoder_start_token_id= self.config['decoder_start_token_id']

        print(f"Encoder fixed seq‐len: {self.encoder_seq_len}")
        print(f"Decoder fixed seq‐len: {self.decoder_seq_len}")
        print(f"Vocab size: {self.config['vocab_size']}")
        print(f"Hidden size: {self.config['d_model']}")

    def _pad_truncate(self, array: np.ndarray, target_len: int, pad_val: float):
        """Pad or truncate a (1, N) array to (1, target_len)."""
        n = array.shape[1]
        if n >= target_len:
            return array[:, :target_len]
        pad = np.full((1, target_len - n), pad_val, dtype=array.dtype)
        return np.concatenate([array, pad], axis=1)

    def encode(self, input_ids, attention_mask=None):
        # to numpy float32
        if torch.is_tensor(input_ids):
            input_ids = input_ids.numpy().astype(np.float32)
        if torch.is_tensor(attention_mask):
            attention_mask = attention_mask.numpy().astype(np.float32)

        # pad/truncate to fixed encoder length
        input_ids     = self._pad_truncate(input_ids,     self.encoder_seq_len, self.pad_token_id)
        attention_mask= self._pad_truncate(attention_mask,self.encoder_seq_len, 0.0)

        encoder_inputs = {
            'input_ids':      input_ids,
            'attention_mask': attention_mask
        }
        encoder_output = self.encoder.predict(encoder_inputs)

        # find the hidden‐states tensor
        for v in encoder_output.values():
            if v.ndim == 3 and v.shape[-1] == self.config['d_model']:
                return v, attention_mask
        raise RuntimeError("Failed to fetch encoder hidden states")

    def generate(self, input_ids, attention_mask=None, max_length=50):
        # 1) encode
        encoder_hidden, encoder_mask = self.encode(input_ids, attention_mask)
        batch_size = encoder_hidden.shape[0]

        # 2) initialize decoder_input to full‐length pad + start token
        decoder_input = np.full(
            (batch_size, self.decoder_seq_len),
            self.pad_token_id,
            dtype=np.float32
        )
        decoder_input[:, 0] = self.decoder_start_token_id
        generated = [self.decoder_start_token_id]

        # 3) iterative decoding
        for _ in range(max_length):
            # call CoreML decoder
            out = self.decoder.predict({
                'input_ids':           decoder_input,
                'encoder_hidden_states': encoder_hidden,
                'attention_mask':      encoder_mask
            })
            # extract hidden‐states tensor
            hidden = next(v for v in out.values() if v.ndim == 3 and v.shape[-1] == self.config['d_model'])
            #last_hidden = hidden[0, -1, :]
            idx = len(generated) - 1
            last_hidden = hidden[0, idx, :]

            # logits & next token
            logits = self.lm_weight.dot(last_hidden) + self.lm_bias.ravel()
            nxt = int(np.argmax(logits))
            if nxt == self.eos_token_id:
                break
            generated.append(nxt)

            # build next decoder_input: pad/truncate generated list
            arr = np.array([generated], dtype=np.float32)
            decoder_input = self._pad_truncate(arr, self.decoder_seq_len, self.pad_token_id)

        return np.array([generated])
