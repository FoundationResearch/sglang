import argparse
import gc
import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CKPT_PATH = "/apdcephfs_sh8/share_300719895/user/qqzxywei/Models/OLMo-stage1-step999000"
# DEFAULT_HSA_CONFIG = PROJECT_ROOT / "configs/olmo3_7B/config_test_hsa_olmo_modeling.json"
DEFAULT_HSA_CONFIG = PROJECT_ROOT / "configs/olmo3_7B/config_test_pure_olmo_attention.json"


DEFAULT_TEXT = """
Language models are trained to predict the next token from a prefix. This short
sample is intentionally plain text so that both implementations receive exactly
the same token ids. The test compares perplexity from the custom HSA OLMo
modeling and the official Transformers OLMo3 modeling on an input shorter than
the local sliding-window.
""".strip()


DEFAULT_DECODE_PROMPT = """
Write a concise three-sentence explanation of why the sky appears blue during
the day. Avoid repeating phrases. Answer:
""".strip()


def parse_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def get_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.amp.autocast("cuda", dtype=dtype)
    return nullcontext()


def register_hsa_model():
    try:
        from models.FlashHSA.configuration_hsa import HSAConfig
        from models.FlashHSA.modeling_olmo_lhsa import HSAForCausalLM
    except Exception as exc:
        raise RuntimeError(
            "Failed to import custom OLMo HSA modeling. Fix the import/runtime "
            "error in models/FlashHSA/modeling_olmo_lhsa.py first."
        ) from exc

    HSAConfig.model_type = "olmo_lhsa"
    AutoConfig.register("olmo_lhsa", HSAConfig, exist_ok=True)
    HSAForCausalLM.config_class = HSAConfig
    AutoModelForCausalLM.register(HSAConfig, HSAForCausalLM, exist_ok=True)
    return HSAConfig, HSAForCausalLM


def build_input_ids(tokenizer, args):
    if args.text_file is not None:
        text = Path(args.text_file).read_text(encoding="utf-8")
    elif args.text is not None:
        text = args.text
    else:
        text = "\n".join([DEFAULT_TEXT] * 64)

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
        add_special_tokens=not args.no_add_special_tokens,
    )
    input_ids = encoded["input_ids"]
    if input_ids.shape[1] < 2:
        raise ValueError("Need at least 2 tokens to compute causal LM perplexity.")
    return input_ids


def load_hsa_model(args, device, dtype):
    register_hsa_model()
    config = AutoConfig.from_pretrained(args.hsa_config)
    # Only enable auto_insert_lmk when landmarks are actually used
    if getattr(config, "insert_landmarks", True):
        config.auto_insert_lmk = True

    kwargs = {
        "config": config,
        "torch_dtype": dtype,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if device.type == "cuda":
        # Load weights directly onto the target GPU to avoid an extra
        # CPU -> GPU copy that otherwise looks like the script is hanging.
        kwargs["device_map"] = {"": device.index if device.index is not None else 0}

    model = AutoModelForCausalLM.from_pretrained(args.ckpt_path, **kwargs)
    if device.type != "cuda":
        model.to(device)
    model.eval()
    if hasattr(model, "_gen_state"):
        model._gen_state.reset()

    # Sanity: confirm HSA modeling will internally insert+strip LMK so that
    # external logits length matches the raw token length, matching HF.
    print(
        f"[HSA sanity] auto_insert_lmk={getattr(model, 'auto_insert_lmk', None)}, "
        f"insert_landmarks={getattr(model, 'insert_landmarks', None)}, "
        f"gen_state.active={getattr(getattr(model, '_gen_state', None), 'active', None)}"
    )
    return model


def load_hf_model(args, device, dtype):
    kwargs = {"torch_dtype": dtype}
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    if device.type == "cuda":
        kwargs["device_map"] = {"": device.index if device.index is not None else 0}
    model = AutoModelForCausalLM.from_pretrained(args.ckpt_path, **kwargs)
    if device.type != "cuda":
        model.to(device)
    model.eval()
    return model


def compute_ppl_from_step_scores(step_scores, target_ids):
    target_ids = target_ids.to(step_scores.device).long()
    loss = F.cross_entropy(step_scores.float(), target_ids, reduction="mean")
    return loss.item(), math.exp(loss.item())


def compute_ppl(model, input_ids_cpu, device, dtype, compare_logits, tag=""):
    input_ids = input_ids_cpu.to(device)
    is_hsa = hasattr(model, "_gen_state")
    if is_hsa:
        # Make sure we are NOT in generate mode so HSA forward goes through
        # the auto_insert_lmk + LMK-strip path that produces logits aligned to
        # raw tokens (same length as input_ids).
        model._gen_state.reset()

    with torch.no_grad(), autocast_context(device, dtype):
        outputs = model(input_ids=input_ids, use_cache=True)

    logits = outputs.logits
    in_len = input_ids.shape[1]
    out_len = logits.shape[1]
    print(
        f"[Prefill sanity{(' ' + tag) if tag else ''}] "
        f"input_len={in_len}, logits_len={out_len}"
    )
    if out_len != in_len:
        raise RuntimeError(
            f"Logit length mismatch: logits={tuple(logits.shape)}, "
            f"input_ids={tuple(input_ids.shape)}. For HSA make sure "
            f"auto_insert_lmk=True and gen_state is reset so LMK positions "
            f"are stripped before lm_head."
        )

    shift_logits = logits[:, :-1, :].float().contiguous()
    shift_labels = input_ids[:, 1:].long().contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        reduction="mean",
    )
    ppl = math.exp(loss.item())
    logits_cpu = logits.detach().float().cpu() if compare_logits else None
    return loss.item(), ppl, logits_cpu


@torch.no_grad()
def run_greedy_decode(model, prompt_ids_cpu, device, dtype, max_new_tokens, tag=""):
    """Run greedy decoding via model.generate() with KV cache, capturing
    per-step next-token logits so we can compare them against another model.
    """
    prompt_ids = prompt_ids_cpu.to(device)
    if hasattr(model, "_gen_state"):
        model._gen_state.reset()
    with autocast_context(device, dtype):
        output = model.generate(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=getattr(model.config, "pad_token_id", None)
            or getattr(model.config, "eos_token_id", None),
        )
    if hasattr(model, "_gen_state"):
        model._gen_state.reset()

    prompt_len = prompt_ids.shape[1]
    seq_len = output.sequences.shape[1]
    n_steps = len(output.scores)
    print(
        f"[Decode sanity{(' ' + tag) if tag else ''}] "
        f"prompt_len={prompt_len}, sequence_len={seq_len}, "
        f"new_tokens={seq_len - prompt_len}, output.scores steps={n_steps}"
    )
    if seq_len - prompt_len != n_steps:
        raise RuntimeError(
            f"Decode mismatch: sequences advanced by {seq_len - prompt_len} "
            f"tokens but output.scores has {n_steps} entries. Check whether "
            f"HSA prepare_inputs_for_generation accidentally exposed LMK "
            f"steps to generate()."
        )

    generated_ids = output.sequences[0, prompt_len:].detach().cpu()
    # output.scores: tuple of len gen_len, each [B, vocab]; B=1.
    scores = torch.stack(output.scores, dim=0).squeeze(1).detach().float().cpu()
    return generated_ids, scores


@torch.no_grad()
def run_beam_decode(model, prompt_ids_cpu, device, dtype, max_new_tokens, num_beams=5, tag=""):
    """Run beam-search decoding via model.generate() with KV cache. Returns the
    best-beam generated token ids. Beam search is more robust than greedy to
    bf16 tie-breaking noise so it helps separate "modeling bug" from
    "greedy-decode-tie avalanche".
    """
    prompt_ids = prompt_ids_cpu.to(device)
    if hasattr(model, "_gen_state"):
        model._gen_state.reset()
    with autocast_context(device, dtype):
        output = model.generate(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            early_stopping=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=False,
            pad_token_id=getattr(model.config, "pad_token_id", None)
            or getattr(model.config, "eos_token_id", None),
        )
    if hasattr(model, "_gen_state"):
        model._gen_state.reset()

    prompt_len = prompt_ids.shape[1]
    seq_len = output.sequences.shape[1]
    print(
        f"[Beam sanity{(' ' + tag) if tag else ''}] "
        f"num_beams={num_beams}, prompt_len={prompt_len}, sequence_len={seq_len}, "
        f"new_tokens={seq_len - prompt_len}"
    )

    generated_ids = output.sequences[0, prompt_len:].detach().cpu()
    return generated_ids


@torch.no_grad()
def run_hsa_batch_consistency(
    model, tokenizer, device, dtype, max_new_tokens=100,
    short_len=10, long_len=300,
):
    """Mimic OpenCompass HF gen-mode batched inference and compare against
    per-sample bsz=1 generate. Reproduces opencompass huggingface_above_v4_33.py:
        - tokenizer.padding_side='left', truncation_side='left'
        - batch_encode_plus(messages, padding=True, truncation=True,
                            add_special_tokens=True, return_tensors='pt')
        - model.generate(**tokens, do_sample=False, num_beams=1,
                         pad_token_id=tokenizer.pad_token_id, max_new_tokens=...)
        - outputs = outputs[:, tokens['input_ids'].shape[1]:]
    No external manual padding is required: HF generate handles left-padded
    batches internally via attention_mask.

    Two prompts: one short (~short_len tokens), one long (~long_len tokens).
    """
    # Build two prompts of distinct lengths from the existing default texts.
    base_text = " ".join([DEFAULT_DECODE_PROMPT, DEFAULT_TEXT])
    long_ids = tokenizer(base_text, add_special_tokens=True)["input_ids"]
    while len(long_ids) < long_len:
        long_ids = long_ids + long_ids
    long_ids = long_ids[:long_len]
    short_ids = long_ids[:short_len]
    long_text = tokenizer.decode(long_ids, skip_special_tokens=False)
    short_text = tokenizer.decode(short_ids, skip_special_tokens=False)
    messages = [short_text, long_text]
    print(
        f"\n[Batch consistency] prompts: short~{short_len} tok, long~{long_len} tok, "
        f"max_new_tokens={max_new_tokens}"
    )

    # Ensure pad_token_id exists; fallback to eos.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = (
            getattr(model.config, "pad_token_id", None)
            or getattr(model.config, "eos_token_id", None)
            or tokenizer.eos_token_id
        )
    pad_id = tokenizer.pad_token_id

    # Save and override tokenizer padding side (opencompass default).
    saved_padding_side = tokenizer.padding_side
    saved_truncation_side = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        pad_token_id=pad_id,
    )

    try:
        # Path A: bsz=1 per-sample reference.
        per_sample_outs = []
        for i, msg in enumerate(messages):
            tokens = tokenizer(msg, return_tensors="pt", add_special_tokens=False)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            if hasattr(model, "_gen_state"):
                model._gen_state.reset()
            with autocast_context(device, dtype):
                out = model.generate(**tokens, **gen_kwargs)
            if hasattr(model, "_gen_state"):
                model._gen_state.reset()
            new_tokens = out[0, tokens["input_ids"].shape[1]:].detach().cpu()
            per_sample_outs.append((tokens["input_ids"].shape[1], new_tokens))
            print(
                f"[Batch consistency / bsz=1] sample {i}: prompt_len="
                f"{tokens['input_ids'].shape[1]}, gen_len={new_tokens.shape[0]}"
            )

        # Path B: bsz=N batched generate (opencompass-style).
        tokens = tokenizer.batch_encode_plus(
            messages, return_tensors="pt", padding=True,
            truncation=True, add_special_tokens=False,
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        if hasattr(model, "_gen_state"):
            model._gen_state.reset()
        with autocast_context(device, dtype):
            batch_out = model.generate(**tokens, **gen_kwargs)
        if hasattr(model, "_gen_state"):
            model._gen_state.reset()
        prompt_len_padded = tokens["input_ids"].shape[1]
        batch_new = batch_out[:, prompt_len_padded:].detach().cpu()
        print(
            f"[Batch consistency / bsz={len(messages)}] padded_prompt_len="
            f"{prompt_len_padded}, gen_len={batch_new.shape[1]}"
        )
    finally:
        tokenizer.padding_side = saved_padding_side
        tokenizer.truncation_side = saved_truncation_side

    # Compare per-sample.
    print(f"[Batch consistency] bsz=1 vs bsz={len(messages)} comparison:")
    for i, (ref_prompt_len, ref_new) in enumerate(per_sample_outs):
        bsz_new = batch_new[i]
        n = min(ref_new.shape[0], bsz_new.shape[0])
        match = (ref_new[:n] == bsz_new[:n])
        match_count = int(match.sum().item())
        first_diff = -1 if match.all() else int((~match).nonzero()[0].item())
        print(
            f"  sample {i} (real_prompt_len={ref_prompt_len}): "
            f"match {match_count}/{n} ({100.0 * match_count / max(n, 1):.2f}%)"
            + (f", first_diff_at={first_diff}" if first_diff >= 0 else "")
        )
        print(f"    bsz=1 ids: {ref_new[:n].tolist()}")
        print(f"    bsz=N ids: {bsz_new[:n].tolist()}")
        try:
            print(f"    bsz=1 text: {tokenizer.decode(ref_new[:n].tolist())!r}")
            print(f"    bsz=N text: {tokenizer.decode(bsz_new[:n].tolist())!r}")
        except Exception:
            pass


@torch.no_grad()
def run_external_lmk_forward_baseline(
    model, prompt_ids_cpu, generated_ids_cpu, device, dtype, tag=""
):
    """Mirror unittests/test_generate_olmo.py: use HSA generated tokens,
    externally insert LMK, disable auto_insert_lmk, full forward, externally
    strip LMK logits, then compare shifted logits with generate scores.
    """
    prompt_ids = prompt_ids_cpu.to(device)
    generated_ids = generated_ids_cpu.to(device)
    if generated_ids.dim() == 1:
        generated_ids = generated_ids.unsqueeze(0)

    chunk_size = model.chunk_size
    lmk_id = model.lmk_id
    prompt_len = prompt_ids.shape[1]
    gen_len = generated_ids.shape[1]
    full_ids = torch.cat([prompt_ids, generated_ids], dim=1)
    full_len = full_ids.shape[1]

    full_ids_with_lmk = insert_special_tokens(full_ids, lmk_id, chunk_size)
    position_ids = create_position_ids_with_landmarks(
        None, full_len, chunk_size, device
    )
    pos_indices = torch.arange(full_ids_with_lmk.shape[1], device=device)
    non_lmk_mask = ~(pos_indices % chunk_size == chunk_size - 1)

    saved_auto_insert_lmk = getattr(model, "auto_insert_lmk", None)
    if hasattr(model, "_gen_state"):
        model._gen_state.reset()
    model.auto_insert_lmk = False

    try:
        with autocast_context(device, dtype):
            outputs = model(
                input_ids=full_ids_with_lmk,
                position_ids=position_ids,
                use_cache=True,
                attention_mask=None,
            )
    finally:
        model.auto_insert_lmk = saved_auto_insert_lmk
        if hasattr(model, "_gen_state"):
            model._gen_state.reset()

    logits = outputs.logits
    logits_no_lmk = logits[:, non_lmk_mask, :]
    if logits_no_lmk.shape[1] != full_len:
        raise RuntimeError(
            f"[External LMK {tag}] stripped logits length {logits_no_lmk.shape[1]} "
            f"!= raw full length {full_len}"
        )

    step_scores = logits_no_lmk[
        0, prompt_len - 1 : prompt_len + gen_len - 1, :
    ].detach().float().cpu()
    step_argmax = step_scores.argmax(dim=-1)
    print(
        f"[External LMK{(' ' + tag) if tag else ''}] raw_len={full_len}, "
        f"with_lmk_len={full_ids_with_lmk.shape[1]}, gen_len={gen_len}, "
        f"scores_shape={tuple(step_scores.shape)}"
    )
    return step_argmax, step_scores


@torch.no_grad()
def run_teacher_forced_decode(model, prompt_ids_cpu, full_ids_cpu, device, dtype, tag=""):
    """Teacher-forced "decode" comparison.

    Both models receive the SAME ground-truth token sequence, so per-step
    next-token logits can be compared without any divergence caused by
    each model emitting different tokens. We feed
        full_ids = [prompt | target_new_tokens]
    once with use_cache=True and read shifted logits at positions
    [prompt_len-1 : prompt_len-1 + n_new] as the per-step next-token logits.

    This is independent from the model's own generate path, so it isolates
    "is the modeling forward correct on these positions?" from
    "is generate's KV-cache state machine correct?".
    """
    full_ids = full_ids_cpu.to(device)
    prompt_len = prompt_ids_cpu.shape[1]
    n_new = full_ids.shape[1] - prompt_len
    if n_new <= 0:
        raise ValueError("full_ids must be longer than prompt_ids for teacher-forced decode")

    if hasattr(model, "_gen_state"):
        model._gen_state.reset()
    with autocast_context(device, dtype):
        out = model(input_ids=full_ids, use_cache=True)
    if hasattr(model, "_gen_state"):
        model._gen_state.reset()

    logits = out.logits  # [1, T, V] (HSA already strips LMK; same length as full_ids)
    if logits.shape[1] != full_ids.shape[1]:
        raise RuntimeError(
            f"[TF Decode {tag}] logits length {logits.shape[1]} != input length "
            f"{full_ids.shape[1]}; HSA must strip LMK before lm_head"
        )

    # logits[i] predicts token at position i+1, so per-step next-token logits
    # for the n_new "decode" positions are at logits[:, prompt_len-1 : prompt_len-1+n_new].
    step_scores = logits[0, prompt_len - 1 : prompt_len - 1 + n_new].detach().float().cpu()
    step_argmax = step_scores.argmax(dim=-1)
    return step_argmax, step_scores


@torch.no_grad()
def run_incremental_forced_decode(model, prompt_ids_cpu, target_ids_cpu, device, dtype, tag=""):
    """Step-by-step KV-cache decode while forcing a fixed token sequence.

    This mirrors generate(): call prepare_inputs_for_generation on the full
    sequence seen so far, run one forward pass with past_key_values, record the
    next-token logits, then append the fixed target token. Comparing these
    logits to run_teacher_forced_decode on the same [prompt|target] sequence
    checks HSA prefill-vs-decode consistency.
    """
    prompt_ids = prompt_ids_cpu.to(device)
    target_ids = target_ids_cpu.to(device)
    if target_ids.dim() == 1:
        target_ids = target_ids.unsqueeze(0)

    if hasattr(model, "_gen_state"):
        model._gen_state.reset()

    past_key_values = None
    input_ids_so_far = prompt_ids
    scores = []
    with autocast_context(device, dtype):
        for step in range(target_ids.shape[1]):
            model_inputs = model.prepare_inputs_for_generation(
                input_ids_so_far,
                past_key_values=past_key_values,
                use_cache=True,
            )
            outputs = model(**model_inputs)
            step_logits = outputs.logits[:, -1, :]
            scores.append(step_logits.detach().float().cpu())
            past_key_values = outputs.past_key_values
            input_ids_so_far = torch.cat(
                [input_ids_so_far, target_ids[:, step : step + 1]], dim=1
            )

    if hasattr(model, "_gen_state"):
        model._gen_state.reset()

    scores = torch.cat(scores, dim=0)  # [new_tokens, vocab]
    argmax_ids = scores.argmax(dim=-1)
    print(
        f"[Inc Decode sanity{(' ' + tag) if tag else ''}] "
        f"prompt_len={prompt_ids.shape[1]}, forced_steps={target_ids.shape[1]}, "
        f"scores_shape={tuple(scores.shape)}"
    )
    return argmax_ids, scores


def compare_decode(
    hsa_ids, hsa_scores, hf_ids, hf_scores, tokenizer, print_text=False,
    left_name="HSA", right_name="HF",
):
    gen_len = min(hsa_ids.shape[0], hf_ids.shape[0])
    if gen_len == 0:
        print("No tokens generated; skip decode comparison.")
        return

    hsa_ids = hsa_ids[:gen_len]
    hf_ids = hf_ids[:gen_len]
    hsa_scores = hsa_scores[:gen_len]
    hf_scores = hf_scores[:gen_len]

    match = (hsa_ids == hf_ids)
    match_count = int(match.sum().item())
    print(f"\n[Decode] generated {gen_len} tokens, exact match {match_count}/{gen_len} "
          f"({match_count / gen_len * 100:.2f}%)")

    diff = (hsa_scores - hf_scores).abs()
    print(f"[Decode] per-step logits max_abs_diff: {diff.max().item():.6f}")
    print(f"[Decode] per-step logits mean_abs_diff: {diff.mean().item():.6f}")

    hsa_argmax = hsa_scores.argmax(dim=-1)
    hf_argmax = hf_scores.argmax(dim=-1)
    argmax_match = (hsa_argmax == hf_argmax).float().mean().item()
    print(f"[Decode] per-step argmax token match: {argmax_match * 100:.2f}%")

    if print_text:
        print(f"[Decode] {left_name} text:")
        print(tokenizer.decode(hsa_ids.tolist()))
        print(f"[Decode] {right_name} text:")
        print(tokenizer.decode(hf_ids.tolist()))

    if match_count == gen_len:
        return

    bad = (~match).nonzero(as_tuple=False).squeeze(-1).tolist()
    first_bad = bad[: min(10, len(bad))]
    print(f"[Decode] mismatched positions: {bad}")
    print(f"[Decode] showing up to {len(first_bad)} mismatches with top1/top2 logits:")

    def tok_str(idx):
        try:
            return repr(tokenizer.decode([idx]))
        except Exception:
            return ""

    for idx in first_bad:
        h_logit = hsa_scores[idx]
        f_logit = hf_scores[idx]
        h_top = torch.topk(h_logit, k=2)
        f_top = torch.topk(f_logit, k=2)
        h_gap = (h_top.values[0] - h_top.values[1]).item()
        f_gap = (f_top.values[0] - f_top.values[1]).item()
        h_id1, h_id2 = int(h_top.indices[0]), int(h_top.indices[1])
        f_id1, f_id2 = int(f_top.indices[0]), int(f_top.indices[1])
        print(
            f"  step {idx}: emitted {left_name}={int(hsa_ids[idx])}{tok_str(int(hsa_ids[idx]))} "
            f"vs {right_name}={int(hf_ids[idx])}{tok_str(int(hf_ids[idx]))}\n"
            f"    {left_name} top1={h_id1}{tok_str(h_id1)} {h_top.values[0].item():.4f} | "
            f"top2={h_id2}{tok_str(h_id2)} {h_top.values[1].item():.4f} | gap={h_gap:.4f}\n"
            f"    {right_name} top1={f_id1}{tok_str(f_id1)} {f_top.values[0].item():.4f} | "
            f"top2={f_id2}{tok_str(f_id2)} {f_top.values[1].item():.4f} | gap={f_gap:.4f}"
        )


def release_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def compare_logits(hsa_logits, hf_logits, tokenizer=None, max_examples=10):
    if hsa_logits.shape != hf_logits.shape:
        print(f"Logit shape mismatch: HSA={tuple(hsa_logits.shape)}, HF={tuple(hf_logits.shape)}")
        return

    diff = (hsa_logits - hf_logits).abs()
    hsa_argmax = hsa_logits.argmax(dim=-1)
    hf_argmax = hf_logits.argmax(dim=-1)
    match = (hsa_argmax == hf_argmax)
    print(f"logits max_abs_diff: {diff.max().item():.6f}")
    print(f"logits mean_abs_diff: {diff.mean().item():.6f}")
    print(f"argmax token match: {match.float().mean().item() * 100:.2f}%")

    # Inspect mismatches with top1/top2 logits to see whether they are caused
    # by near-ties rather than real disagreements.
    if match.all():
        return
    mismatch = (~match).nonzero(as_tuple=False)
    total_mismatch = mismatch.shape[0]
    print(f"argmax mismatches: {total_mismatch} (showing up to {max_examples})")
    n_show = min(max_examples, total_mismatch)
    for i in range(n_show):
        coords = tuple(int(x) for x in mismatch[i].tolist())  # (b, t) for [B,T,V]
        h_logit = hsa_logits[coords]
        f_logit = hf_logits[coords]
        h_top = torch.topk(h_logit, k=2)
        f_top = torch.topk(f_logit, k=2)
        h_gap = (h_top.values[0] - h_top.values[1]).item()
        f_gap = (f_top.values[0] - f_top.values[1]).item()

        h_id1, h_id2 = int(h_top.indices[0]), int(h_top.indices[1])
        f_id1, f_id2 = int(f_top.indices[0]), int(f_top.indices[1])

        def tok_str(idx):
            if tokenizer is None:
                return ""
            try:
                return repr(tokenizer.decode([idx]))
            except Exception:
                return ""

        print(
            f"  pos {coords}:\n"
            f"    HSA top1={h_id1} {tok_str(h_id1)} {h_top.values[0].item():.4f} | "
            f"top2={h_id2} {tok_str(h_id2)} {h_top.values[1].item():.4f} | gap={h_gap:.4f}\n"
            f"    HF  top1={f_id1} {tok_str(f_id1)} {f_top.values[0].item():.4f} | "
            f"top2={f_id2} {tok_str(f_id2)} {f_top.values[1].item():.4f} | gap={f_gap:.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--hsa_config", type=str, default=str(DEFAULT_HSA_CONFIG))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_3")
    parser.add_argument("--max_length", type=int, default=500)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--text_file", type=str, default=None)
    parser.add_argument("--no_add_special_tokens", action="store_true")
    parser.add_argument("--no_compare_logits", action="store_true")
    parser.add_argument(
        "--decode_new_tokens",
        type=int,
        default=100,
        help="Number of greedy-decode tokens to compare. Set 0 to skip the decode test. "
             "Keep prompt_len + decode_new_tokens within the local sliding window.",
    )
    parser.add_argument(
        "--decode_prompt_len",
        type=int,
        default=-1,
        help="If > 0, truncate decode prompt to this many tokens after tokenization. "
             "Defaults to use the full dedicated decode prompt.",
    )
    parser.add_argument(
        "--decode_text",
        type=str,
        default=None,
        help="Dedicated prompt for greedy decode. If unset, use a non-repetitive default prompt.",
    )
    parser.add_argument(
        "--decode_text_file",
        type=str,
        default=None,
        help="Read dedicated decode prompt from this text file.",
    )
    parser.add_argument(
        "--decode_mode",
        choices=["greedy", "teacher_forced", "both"],
        default="both",
        help="greedy: each model decodes on its own (any divergence cascades). "
             "teacher_forced: feed both models the SAME ground-truth sequence "
             "(HF greedy output) and compare per-step next-token logits. "
             "both: run both for cross-checking.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=0,
        help="If > 1, also run beam-search decode on both models and compare "
             "the best-beam output. Useful to check whether divergence is due "
             "to greedy bf16 tie avalanche (beam search is far more robust).",
    )
    parser.add_argument(
        "--run_batch_consistency",
        action="store_true",
        help="On HSA model, compare bsz=1 per-sample generate vs bsz=N batched "
             "generate (opencompass HF gen-mode style: left padding, "
             "attention_mask, no external manual padding).",
    )
    parser.add_argument(
        "--batch_max_new_tokens",
        type=int,
        default=100,
        help="max_new_tokens for the batch consistency test.",
    )
    parser.add_argument(
        "--batch_short_len",
        type=int,
        default=10,
        help="Short prompt length (in tokens) for the batch consistency test.",
    )
    parser.add_argument(
        "--batch_long_len",
        type=int,
        default=300,
        help="Long prompt length (in tokens) for the batch consistency test.",
    )
    args = parser.parse_args()

    if args.max_length > 500:
        raise ValueError("This test intentionally limits input length to <= 500 tokens.")

    device = get_device(args.device)
    dtype = parse_dtype(args.dtype)
    compare = not args.no_compare_logits

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    input_ids = build_input_ids(tokenizer, args)
    print(f"ckpt_path: {args.ckpt_path}")
    print(f"hsa_config: {args.hsa_config}")
    print(f"device: {device}, dtype: {dtype}, attn_implementation: {args.attn_implementation}")
    print(f"input token length: {input_ids.shape[1]}")

    # Build a dedicated prompt for decode so generation quality is easier to inspect.
    decode_prompt = None
    if args.decode_new_tokens > 0:
        if args.decode_text_file is not None:
            decode_text = Path(args.decode_text_file).read_text(encoding="utf-8")
        elif args.decode_text is not None:
            decode_text = args.decode_text
        else:
            decode_text = DEFAULT_DECODE_PROMPT

        decode_encoded = tokenizer(
            decode_text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=not args.no_add_special_tokens,
        )
        decode_ids = decode_encoded["input_ids"]
        if args.decode_prompt_len > 0:
            decode_ids = decode_ids[:, : args.decode_prompt_len]
        # Reserve room so prompt + new tokens still fit under --max_length (<=500).
        max_prompt_len = args.max_length - args.decode_new_tokens
        if max_prompt_len < 1:
            print("[Decode] Not enough room for prompt under --max_length; skip decode test.")
        else:
            decode_prompt = decode_ids[:, :max_prompt_len]
            print(
                f"[Decode] prompt_len={decode_prompt.shape[1]}, "
                f"new_tokens={args.decode_new_tokens}, "
                f"total<=({decode_prompt.shape[1] + args.decode_new_tokens})"
            )
            print("[Decode] prompt text:")
            print(tokenizer.decode(decode_prompt[0].tolist()))

    # Run HF first so we can use its greedy output as the reference token
    # sequence for teacher-forced decoding, then release HF before loading HSA
    # so we never need to fit two 7B models in GPU memory at once.
    print("\nLoading official Transformers OLMo3 modeling...")
    hf_model = load_hf_model(args, device, dtype)
    hf_loss, hf_ppl, hf_logits = compute_ppl(
        hf_model, input_ids, device, dtype, compare, tag="HF"
    )
    print(f"HF  loss: {hf_loss:.6f}, HF  ppl: {hf_ppl:.6f}")
    hf_decode_ids = None
    hf_decode_scores = None
    hf_tf_argmax = None
    hf_tf_scores = None
    hf_beam_ids = None
    ref_ids = None
    full_ids = None
    if decode_prompt is not None and args.decode_mode in ("greedy", "both"):
        hf_decode_ids, hf_decode_scores = run_greedy_decode(
            hf_model, decode_prompt, device, dtype, args.decode_new_tokens, tag="HF"
        )
        print(f"HF  decode generated {hf_decode_ids.shape[0]} tokens")
    if decode_prompt is not None and args.num_beams > 1:
        hf_beam_ids = run_beam_decode(
            hf_model, decode_prompt, device, dtype, args.decode_new_tokens,
            num_beams=args.num_beams, tag="HF",
        )
        print(f"HF  beam decode generated {hf_beam_ids.shape[0]} tokens")
    if decode_prompt is not None and args.decode_mode in ("teacher_forced", "both"):
        if hf_decode_ids is None:
            hf_decode_ids, _ = run_greedy_decode(
                hf_model, decode_prompt, device, dtype, args.decode_new_tokens, tag="HF-ref"
            )
        ref_ids = hf_decode_ids
        full_ids = torch.cat([decode_prompt, ref_ids.unsqueeze(0)], dim=1)
        print(
            f"\n[TF Decode] prompt_len={decode_prompt.shape[1]}, "
            f"new_tokens={ref_ids.shape[0]}, full_len={full_ids.shape[1]}"
        )
        hf_tf_argmax, hf_tf_scores = run_teacher_forced_decode(
            hf_model, decode_prompt, full_ids, device, dtype, tag="HF"
        )
    release_model(hf_model)

    print("\nLoading custom HSA OLMo modeling...")
    hsa_model = load_hsa_model(args, device, dtype)
    hsa_loss, hsa_ppl, hsa_logits = compute_ppl(
        hsa_model, input_ids, device, dtype, compare, tag="HSA"
    )
    print(f"HSA loss: {hsa_loss:.6f}, HSA ppl: {hsa_ppl:.6f}")
    hsa_decode_ids = None
    hsa_decode_scores = None
    hsa_tf_argmax = None
    hsa_tf_scores = None
    hsa_inc_argmax = None
    hsa_inc_scores = None
    hsa_ext_argmax = None
    hsa_ext_scores = None
    hsa_beam_ids = None
    if decode_prompt is not None and args.decode_mode in ("greedy", "both"):
        hsa_decode_ids, hsa_decode_scores = run_greedy_decode(
            hsa_model, decode_prompt, device, dtype, args.decode_new_tokens, tag="HSA"
        )
        print(f"HSA decode generated {hsa_decode_ids.shape[0]} tokens")
    if decode_prompt is not None and args.num_beams > 1:
        hsa_beam_ids = run_beam_decode(
            hsa_model, decode_prompt, device, dtype, args.decode_new_tokens,
            num_beams=args.num_beams, tag="HSA",
        )
        print(f"HSA beam decode generated {hsa_beam_ids.shape[0]} tokens")
    if hsa_decode_ids is not None and getattr(hsa_model, 'insert_landmarks', False):
        hsa_ext_argmax, hsa_ext_scores = run_external_lmk_forward_baseline(
            hsa_model, decode_prompt, hsa_decode_ids, device, dtype, tag="HSA-generate"
        )
    else:
        hsa_ext_argmax, hsa_ext_scores = None, None
    if decode_prompt is not None and full_ids is not None:
        hsa_tf_argmax, hsa_tf_scores = run_teacher_forced_decode(
            hsa_model, decode_prompt, full_ids, device, dtype, tag="HSA-prefill"
        )
        hsa_inc_argmax, hsa_inc_scores = run_incremental_forced_decode(
            hsa_model, decode_prompt, ref_ids, device, dtype, tag="HSA-decode"
        )
    if args.run_batch_consistency:
        run_hsa_batch_consistency(
            hsa_model, tokenizer, device, dtype,
            max_new_tokens=args.batch_max_new_tokens,
            short_len=args.batch_short_len,
            long_len=args.batch_long_len,
        )
    release_model(hsa_model)

    print("\nSummary")
    print(f"loss abs diff: {abs(hsa_loss - hf_loss):.6f}")
    print(f"ppl  abs diff: {abs(hsa_ppl - hf_ppl):.6f}")
    if compare:
        compare_logits(hsa_logits, hf_logits, tokenizer=tokenizer)

    if hsa_decode_ids is not None and hf_decode_ids is not None:
        compare_decode(
            hsa_decode_ids, hsa_decode_scores, hf_decode_ids, hf_decode_scores, tokenizer,
            print_text=True,
        )

    if hsa_beam_ids is not None and hf_beam_ids is not None:
        # Beam search comparison: only compare token ids since scores aren't easily aligned
        # across beams. Bf16 tie-breaking has much weaker effect under beam search.
        n = min(hsa_beam_ids.shape[0], hf_beam_ids.shape[0])
        match = (hsa_beam_ids[:n] == hf_beam_ids[:n]).sum().item()
        print(f"\n[Beam Decode] num_beams={args.num_beams} -> exact token match {match}/{n} ({100.0*match/n:.2f}%)")
        print("[Beam Decode] HSA text:")
        print(tokenizer.decode(hsa_beam_ids.tolist()))
        print("[Beam Decode] HF  text:")
        print(tokenizer.decode(hf_beam_ids.tolist()))
        if match < n:
            first_div = int((hsa_beam_ids[:n] != hf_beam_ids[:n]).nonzero()[0].item())
            print(f"[Beam Decode] first divergence at step {first_div}: "
                  f"HSA={hsa_beam_ids[first_div].item()!r} vs HF={hf_beam_ids[first_div].item()!r}")

    if hsa_decode_ids is not None and hsa_ext_argmax is not None:
        print("\n[HSA Generate vs External-LMK Prefill] same logic as unittests/test_generate_olmo.py")
        compare_decode(
            hsa_decode_ids, hsa_decode_scores, hsa_ext_argmax, hsa_ext_scores, tokenizer,
            left_name="Generate", right_name="ExternalPrefill",
        )
        gen_loss, gen_ppl = compute_ppl_from_step_scores(hsa_decode_scores, hsa_decode_ids)
        ext_loss, ext_ppl = compute_ppl_from_step_scores(hsa_ext_scores, hsa_decode_ids)
        print(
            f"[HSA Generate vs External-LMK Prefill] "
            f"gen_nll={gen_loss:.6f}, gen_ppl={gen_ppl:.6f}, "
            f"prefill_nll={ext_loss:.6f}, prefill_ppl={ext_ppl:.6f}, "
            f"nll_abs_diff={abs(gen_loss - ext_loss):.6f}, "
            f"ppl_abs_diff={abs(gen_ppl - ext_ppl):.6f}"
        )

    if hsa_tf_argmax is not None and hf_tf_argmax is not None:
        # Teacher-forced comparison: both models saw the SAME inputs at every
        # step, so any divergence here is real (not cascading from a tie).
        # We synthesize "emitted ids" as each model's own argmax at each step.
        print("\n[TF Decode] both models received identical inputs at every step.")
        compare_decode(
            hsa_tf_argmax, hsa_tf_scores, hf_tf_argmax, hf_tf_scores, tokenizer
        )

    if hsa_inc_argmax is not None and hsa_tf_argmax is not None:
        # HSA self-consistency: incremental KV-cache decode vs full prefill on
        # the same [prompt|target] sequence. This is the train/infer check.
        print("\n[HSA Self] incremental decode vs full prefill on identical tokens.")
        compare_decode(
            hsa_inc_argmax, hsa_inc_scores, hsa_tf_argmax, hsa_tf_scores, tokenizer
        )


if __name__ == "__main__":
    main()




"""

python code_exp/compare_olmo_hsa_with_trm.py \
    --run_batch_consistency --attn_implementation sdpa --dtype fp32 \
    --batch_max_new_tokens 100 \
    --batch_short_len 10 --batch_long_len 300 \
    --decode_new_tokens 0 --no_compare_logits


"""