import os
import torch
import torch.nn.functional as F
import coremltools as ct
from transformers import MarianConfig

class CustomLogitsProcessor:
    def __init__(self, min_length: int, eos_token_id: int, pad_token_id: int):
        self.min_length = min_length
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            scores[:, self.eos_token_id] = -float("inf")

        scores[:, self.pad_token_id] = -float("inf")
        return scores

class MarianCoreML:
    def __init__(self, path: str, device: str = 'cpu'):
        self.device = torch.device(device)

        self.encoder_model = ct.models.MLModel(os.path.join(path, "encoder.mlpackage"))
        self.decoder_model = ct.models.MLModel(os.path.join(path, "decoder.mlpackage"))

        self.config = MarianConfig.from_pretrained(path)
        self.final_logits_weight = torch.load(os.path.join(path, 'lm_weight.bin')).to(self.device)
        self.final_logits_bias = torch.load(os.path.join(path, 'lm_bias.bin')).to(self.device)

        self.logits_processor = CustomLogitsProcessor(
            2, self.config.eos_token_id, self.config.pad_token_id
        )

    def _encoder_forward(self, input_ids, attention_mask):
        inputs = {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy()
        }
        output = self.encoder_model.predict(inputs)["encoder_hidden_states"]
        return torch.from_numpy(output).to(input_ids.device)

    def _decoder_forward(self, input_ids, encoder_hidden_states, attention_mask):
        inputs = {
            "input_ids": input_ids.numpy(),
            "encoder_hidden_states": encoder_hidden_states.numpy(),
            "attention_mask": attention_mask.numpy()
        }
        output = self.decoder_model.predict(inputs)["decoder_output"]
        decoder_output = torch.from_numpy(output).to(input_ids.device)

        lm_logits = F.linear(decoder_output, self.final_logits_weight, bias=self.final_logits_bias)
        return lm_logits

    def _init_sequence_length_for_generation(self, input_ids, max_length: int):
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.int8, device=input_ids.device)
        sequence_lengths = torch.ones(input_ids.shape[0], dtype=torch.int8, device=input_ids.device) * max_length
        cur_len = input_ids.shape[-1]
        return sequence_lengths, unfinished_sequences, cur_len

    def greedy_search(self, input_ids, encoder_output, attention_mask):
        max_length = self.config.max_length
        pad_token_id = self.config.pad_token_id
        eos_token_id = self.config.eos_token_id

        sequence_lengths, unfinished_sequences, cur_len = \
            self._init_sequence_length_for_generation(input_ids, max_length)

        while cur_len < max_length:
            logits = self._decoder_forward(input_ids, encoder_output, attention_mask)
            next_token_logits = logits[:, -1, :]

            scores = self.logits_processor(input_ids, next_token_logits)

            next_tokens = torch.argmax(scores, dim=-1)
            next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            input_ids[input_ids[:, -2] == eos_token_id, -1] = eos_token_id
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).char())

            if unfinished_sequences.max() == 0:
                break

            cur_len += 1

        return input_ids

    def _prepare_decoder_input_ids_for_generation(self, input_ids, decoder_start_token_id: int):
        decoder_input_ids = (
            torch.ones((input_ids.shape[0], 1), dtype=torch.int64, device=input_ids.device)
            * decoder_start_token_id
        )
        return decoder_input_ids

    def generate(self, input_ids, attention_mask):
        decoder_start_token_id = self.config.decoder_start_token_id
        encoder_output = self._encoder_forward(input_ids, attention_mask)
        input_ids = self._prepare_decoder_input_ids_for_generation(
            input_ids, decoder_start_token_id
        )
        return self.greedy_search(input_ids, encoder_output, attention_mask)