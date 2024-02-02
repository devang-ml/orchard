
### Collection of pipeline parallel hooks to register with the Transformer network.

def PP_TransformerBlock_forward_pre_hook(module, args):
    x, input_pos, freqs_cis, mask = args

    rank = _get_rank()
    x = x.contiguous()
    dist.recv(tensor=x, src=rank - 1)

    return x, input_pos, freqs_cis, mask

def PP_TransformerBlock_forward_hook(module, input, output):
    x, input_pos, freqs_cis, mask = input

    rank = _get_rank()
    output = output.contiguous()
    dist.send(tensor=output, dst=rank + 1)

    return output

def PP_Transformer_forward_hook(module, input, output):
    rank = _get_rank()
    world_size = _get_world_size()

    if rank == (world_size - 1):
        for r in range(world_size - 1):
            dist.send(output, dst=r)
    else:
        dist.recv(output, src=world_size - 1)

    return output