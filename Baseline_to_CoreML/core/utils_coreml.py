import os
import shutil
import torch
import coremltools as ct
from transformers import MarianMTModel, MarianTokenizer
from core.layers import MarianDecoder, MarianEncoder

def create_marian_encoder_decoder(model_path: str, outdir: str):
    model = MarianMTModel.from_pretrained(model_path)

    encoder = model.get_encoder()
    decoder = model.get_decoder()

    marian_encoder = MarianEncoder(encoder).eval()
    marian_decoder = MarianDecoder(decoder).eval()

    torch.save(model.model.shared.weight, os.path.join(outdir, 'lm_weight.bin'))
    torch.save(model.final_logits_bias, os.path.join(outdir, 'lm_bias.bin'))

    for file in ['config.json', 'source.spm', 'target.spm', 'tokenizer_config.json', 'vocab.json']:
        src = os.path.join(model_path, file)
        dst = os.path.join(outdir, file)
        if os.path.exists(src):
            shutil.copyfile(src, dst)

    return marian_encoder, marian_decoder

def generate_coreml_graph(model_path, encoder_path, decoder_path, outdir):
    encoder, decoder = create_marian_encoder_decoder(model_path, outdir)

    tokenizer = MarianTokenizer.from_pretrained(model_path)
    inputs = tokenizer("Hello World !", return_tensors="pt")
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

    print("Exporting encoder to CoreML...")
    traced_encoder = torch.jit.trace(encoder, (input_ids, attention_mask))
    # mlmodel_encoder = ct.convert(
    #     traced_encoder,
    #     inputs=[
    #         ct.TensorType(name="input_ids", shape=input_ids.shape),
    #         ct.TensorType(name="attention_mask", shape=attention_mask.shape)
    #     ]
    # )
    # mlmodel_encoder.save(encoder_path)

    mlmodel_encoder = ct.convert(
    traced_encoder,
    convert_to="neuralnetwork",  
    minimum_deployment_target=ct.target.iOS13,
    inputs=[
        ct.TensorType(name="input_ids", shape=input_ids.shape),
        ct.TensorType(name="attention_mask", shape=attention_mask.shape)
    ]
    )
    mlmodel_encoder.save(encoder_path)  # .mlmodel


    print("Exporting decoder to CoreML...")
    encoder_hidden_states = encoder(input_ids, attention_mask)[0]
    traced_decoder = torch.jit.trace(decoder, (input_ids, encoder_hidden_states, attention_mask))
    mlmodel_decoder = ct.convert(
        traced_decoder,
        inputs=[
            ct.TensorType(name="input_ids", shape=input_ids.shape),
            ct.TensorType(name="encoder_hidden_states", shape=encoder_hidden_states.shape),
            ct.TensorType(name="attention_mask", shape=attention_mask.shape)
        ]
    )
    mlmodel_decoder.save(decoder_path)
