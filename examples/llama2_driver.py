# llama2_driver
# This is an example script to use llama2 model in single GPU or multi GPU scenario
import os
import sys
import torch
import torch.distributed as dist
from pathlib import Path
from sentencepiece import SentencePieceProcessor

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from orchard.networks.transformer import Transformer, ModelArgs

def _get_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def _get_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

def _load_model(args, precision=torch.bfloat16):
    rank = _get_rank()
    world_size = _get_world_size()

    config = ModelArgs.from_name(Path(args.checkpoint_path).parent.name)
    config.rank = rank
    config.world_size = world_size

    if args.pp:
        config.n_local_layers = config.n_local_layers // world_size
    # elif args.tp:
    #     config.n_head = config.n_head // world_size
    #     config.dim = config.dim // world_size
    #     config.n_local_heads = config.n_local_heads // world_size

    model = Transformer(config)

    # if args.tp:
    #     for block in model.layers:
    #         block.attention.n_head = block.attention.n_head // world_size
    #         block.attention.dim = block.attention.dim // world_size
    #         block.attention.head_dim = block.attention.dim // block.attention.n_head
    #         block.attention.n_local_heads = block.attention.n_local_heads // world_size

    if args.pp or args.tp:
        model_path = str(args.model_path).format(rank)
        model.load_state_dict(torch.load(str(model_path)), assign=True)
    else:
        checkpoint = torch.load(str(args.checkpoint_path), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True)

    model = model.to(device=args.device, dtype=precision)
    return model.eval()

def _encode_tokens(tokenizer, string, bos=True, device='cuda'):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def _main():
    import argparse
    parser = argparse.ArgumentParser(description='This is a simple script to load and use Llama2')

    parser.add_argument('--checkpoint_path', type=Path,
                        default=Path("checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"),
                        help='Model checkpoint path.')
    parser.add_argument('--model_path', type=Path,
                        default=Path("models/model_{:02d}.pt"),
                        help='Ranked model path.')
    parser.add_argument('--prompt', type=str, default="Hello, my name is", help='Input prompt.')
    parser.add_argument('--max_new_tokens', type=int, default=50, help='Maximum number of new tokens.')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--device', type=str, default="cuda", help='device to use')
    parser.add_argument('--pp', action="store_true", default=False,
                        help='use multiple GPUs for model distributed using Pipeline Parallel technique')
    parser.add_argument('--tp', action="store_true", default=False,
                        help='use multiple GPUs for model distributed using Tensor Parallel technique')

    args = parser.parse_args()

    rank = _get_rank()
    world_size = _get_world_size()

    if args.device == 'cuda':
        torch.cuda.set_device(rank)
    else:
        torch.set_default_device(args.device)

    if args.pp or args.tp:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    tokenizer_path = args.checkpoint_path.parent / "tokenizer.model"
    tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))
    encoded_tokens = _encode_tokens(tokenizer, args.prompt, bos=True, device=args.device)

    model = _load_model(args)

    if args.pp:
        from orchard.hooks.pp import PP_register_hooks
        PP_register_hooks(model)
    elif args.tp:
        from orchard.hooks.tp import TP_register_hooks
        TP_register_hooks(model)

    torch.manual_seed(1234)
    generated_tokens = model.generate(
        encoded_tokens,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    if ((args.pp or args.tp) and rank == 0) or (not (args.pp and args.tp)):
        print(tokenizer.decode(generated_tokens.tolist()))

    return 0

if __name__ == '__main__':
    sys.exit(_main())

# To create checkpoints from hf transformer model:
#   ./orchard/utils/prepare.sh meta-llama/Llama-2-7b-chat-hf
#
# To create pp compatible models:
#   Use Olive
#   cd examples/transformer
#   python3 -m olive.workflows.run --config pipeline_parallel.json
#
# Default run:
# python3 examples/llama2_driver.py --prompt "What's an apple?"
#
# Run with pp:
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=4 examples/llama2_driver.py --prompt "What's an apple?" --pp
#
# Run with tp:
# ENABLE_INTRA_NODE_COMM=1 torchrun --standalone --nproc_per_node=4 examples/llama2_driver.py --prompt "What's an apple?" --tp
