# llama2_driver
# This is an example script to use llama2 model in single GPU or multi GPU scenario
import itertools
import os
import sys
import time
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
        config.n_local_layer = config.n_local_layer // world_size
    elif args.tp:
        config.n_head = config.n_head // world_size
        config.n_local_head = config.n_local_head // world_size
        config.local_dim = config.local_dim // world_size
        config.local_intermediate_size = config.local_intermediate_size // world_size

    print('Config: ', config.__dict__)

    if args.pp:
        assert (config.n_layer % world_size) == 0
    elif args.tp:
        assert (config.n_head % world_size) == 0

    model = Transformer(config)

    if args.pp or args.tp:
        model_path = str(args.model_path).format(rank)
        model.load_state_dict(torch.load(str(model_path)), assign=True)
    else:
        checkpoint = torch.load(str(args.checkpoint_path), mmap=True, weights_only=True)
        checkpoint = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(checkpoint, assign=True)

    model = model.to(device=args.device, dtype=precision)
    return model.eval()

def _encode_tokens(tokenizer, string, bos=True, device='cuda'):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize()
    elif "cpu" in device:
        pass
    else:
        print(f"device={device} is not yet supported")

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
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens.')
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

    global print
    if (args.tp or args.pp) and (rank != 0):
        # only print on rank 0
        print = lambda *args, **kwargs: None

    if args.device == 'cuda':
        torch.cuda.set_device(rank)
    else:
        torch.set_default_device(args.device)

    if args.pp or args.tp:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        # args.model_path = Path('models_tp/model_{:02d}.pt') if args.tp else Path('models_pp/model_{:02d}.pt')

    tokenizer_path = args.checkpoint_path.parent / "tokenizer.model"
    tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))
    encoded_tokens = _encode_tokens(tokenizer, args.prompt, bos=True, device=args.device)

    t0 = time.time()
    model = _load_model(args)

    if args.pp:
        from orchard.hooks.pp import PP_register_hooks
        PP_register_hooks(model)
    elif args.tp:
        from orchard.hooks.tp import TP_register_hooks
        TP_register_hooks(model)

    device_sync(device=args.device)
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    torch.manual_seed(1234)
    device_sync(device=args.device)

    t0 = time.perf_counter()
    generated_tokens = model.generate(
        encoded_tokens,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    t = time.perf_counter() - t0

    parameter_count = sum(p.numel() for p in model.parameters())
    model_size = sum([p.numel() * p.dtype.itemsize for p in itertools.chain(model.parameters(), model.buffers())])
    tokens_sec = (generated_tokens.size(0) - encoded_tokens.size(0)) / t

    print(tokenizer.decode(generated_tokens.tolist()))

    print('')
    if args.pp or args.tp:
        print(f" Parameters / rank: {parameter_count}")
        print(f" Model size / rank: {model_size  / 1e9:.02f} GB")
    else:
        print(f"        Parameters: {parameter_count}")
        print(f"        Model size: {model_size  / 1e9:.02f} GB")

    print(f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
    print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
    print(f"       Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

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
