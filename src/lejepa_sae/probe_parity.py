"""Memory-bounded FP32 arithmetic checks on identical BF16-rounded weights.

GPT-NeoX and TransformerLens group residual additions differently. Agreement in
real arithmetic does not imply elementwise agreement after each BF16 operation.
Keep CPU weights in their loaded precision and temporarily promote one stage at
a time, so checking the hook mapping does not require a full FP32 model on GPU.
"""

from contextlib import contextmanager

import torch


@contextmanager
def streamed_float32(stages, device):
    """Temporarily execute disjoint CPU-resident stages in FP32 on ``device``."""
    handles = []
    restores = []
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for stage in stages:
            parameters = [(p, p.detach()) for p in stage.parameters()]
            buffers = [
                (module, name, buffer)
                for module in stage.modules()
                for name, buffer in module._buffers.items()
                if buffer is not None
            ]
            if any(value.device.type != "cpu" for _, value in parameters):
                raise ValueError("streamed parity requires CPU-resident weights")

            def promote(module, inputs, parameters=parameters, buffers=buffers):
                for parameter, original in parameters:
                    parameter.data = original.to(device=device, dtype=torch.float32)
                for owner, name, original in buffers:
                    owner._buffers[name] = original.to(
                        device=device,
                        dtype=torch.float32 if original.is_floating_point() else original.dtype,
                    )

            def restore(
                module=None, inputs=None, output=None, parameters=parameters, buffers=buffers
            ):
                for parameter, original in parameters:
                    parameter.data = original
                for owner, name, original in buffers:
                    owner._buffers[name] = original

            restores.append(restore)
            handles.append(stage.register_forward_pre_hook(promote))
            handles.append(stage.register_forward_hook(restore, always_call=True))
        yield
    finally:
        for handle in handles:
            handle.remove()
        for restore in restores:
            restore()
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


class _CapturedBlock(Exception):
    """Terminate the HF forward immediately after the extraction hook."""


@torch.inference_mode()
def hf_streamed_residual(model, tokens, device, layer=16):
    core = model.gpt_neox
    stages = [core.embed_in, *core.layers[: layer + 1]]
    # Newer HF versions compute rotary embeddings outside the decoder blocks.
    if hasattr(core, "rotary_emb"):
        stages.append(core.rotary_emb)
    captured = []

    def capture(module, inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        captured.append(value.float().cpu().clone())
        raise _CapturedBlock

    with streamed_float32(stages, device):
        handle = core.layers[layer].register_forward_hook(capture)
        try:
            try:
                model(tokens.to(device), use_cache=False)
            except _CapturedBlock:
                pass
        finally:
            handle.remove()
    if len(captured) != 1:
        raise RuntimeError("FP32 reference did not capture exactly one HF block output")
    return captured[0]


@torch.inference_mode()
def tl_streamed_residual(model, tokens, device, layer=16):
    original_dtype, original_device = model.cfg.dtype, model.cfg.device
    rotary_buffers = []
    try:
        model.cfg.dtype = torch.float32
        model.cfg.device = device
        for block in model.blocks[: layer + 1]:
            attention = block.attn
            # Recompute trig tables in FP32, rather than widening rounded BF16
            # tables. HF computes these from its FP32 inverse-frequency buffer.
            for name in ("rotary_sin", "rotary_cos"):
                rotary_buffers.append((attention, name, getattr(attention, name)))
            attention.rotary_sin, attention.rotary_cos = attention.calculate_sin_cos_rotary(
                model.cfg.rotary_dim,
                model.cfg.n_ctx,
                base=model.cfg.rotary_base,
                dtype=torch.float32,
            )
        with streamed_float32([model.embed, *model.blocks[: layer + 1]], device):
            _, cache = model.run_with_cache(
                tokens.to(device),
                names_filter=[f"blocks.{layer}.hook_resid_post"],
                stop_at_layer=layer + 1,
            )
            return cache[f"blocks.{layer}.hook_resid_post"].float().cpu().clone()
    finally:
        model.cfg.dtype, model.cfg.device = original_dtype, original_device
        for owner, name, original in rotary_buffers:
            setattr(owner, name, original)
