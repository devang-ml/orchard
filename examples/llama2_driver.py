# llama2_driver
# This is an example script to use llama2 model in single GPU or multi GPU scenario
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn

from orchard/hooks/ import PP_Transformer_forward_hook, PP_TransformerBlock_forward_hook, PP_TransformerBlock_forward_pre_hook
from orchard/network import Transformer, ModelArgs

CONFIGS = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "30B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
}

def _get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def _get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

def _load_model(args, precision=torch.bfloat16):

    if args.pipeline_parallel:
        rank = _get_rank()
        world_size = _get_world_size()

        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        model_path = f'models/model_{rank:02d}.pt'
    else:
        torch.cuda.set_device(args.device)
        model_path = f'models/model.pt'

    model = Transformer(ModelArgs(CONFIGS(model_path.parent.name)))

    checkpoint = torch.load(str(model_path), mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)

    if args.pipeline_parallel:
        rank = _get_rank()
        world_size = _get_world_size()
        model.config.n_local_layers = model.config.n_local_layers // world_size
        start_layer_index = model.config.n_local_layers * rank
        end_layer_index = start_layer_index + model.config.n_local_layers

        model.layers = model.layers[start_layer_index : end_layer_index]

        # register hooks
        model.register_forward_hook(PP_Transformer_forward_hook)

        if rank != 0:
            model.layers[0].register_forward_pre_hook(PP_TransformerBlock_forward_pre_hook)

        if rank != (world_size - 1):
            model.layers[-1].register_forward_hook(PP_TransformerBlock_forward_hook)

    model = model.to(device=args.device, dtype=precision)
    return model.eval()

def _main():
    import argparse
    parser = argparse.ArgumentParser(description='This is a simple script to load and use Llama2')

    parser.add_argument('--checkpoint_path', type=Path,
                        default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
                        help='Model checkpoint path.')
    parser.add_argument('--prompt', type=str, default="Hello, my name is", help='Input prompt.')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='Maximum number of new tokens.')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--device', type=str, default="cuda", help='device to use')
    parser.add_argument('--pipeline_parallel', type=bool, default=False, help='use multiple GPUs for model distributed using Pipeline Parallel tecnique')

    args = parser.parse_args()

    tokenizer_path = args.checkpoint_path.parent / "tokenizer.model"
    tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))
    encoded_tokens = _encode_tokens(tokenizer, args.prompt, bos=True, device=args.device)

    model = _load_model(args)

    torch.manual_seed(1234)
    generated_tokens = model.generate(
        encoded_tokens,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    if (args.pipeline_parallel and rank == 0) or (not args.pipeline_parallel):
        print(tokenizer.decode(generated_tokens.tolist()))

    return 0

if __name__ == '__main__':
    sys.exit(_main())
