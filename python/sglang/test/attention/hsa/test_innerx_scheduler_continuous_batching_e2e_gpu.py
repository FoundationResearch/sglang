import json
import os
import tempfile

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU-only test: CUDA not available."
)

_HSA_VERBOSE = os.getenv("SGLANG_HSA_TEST_VERBOSE", "0") not in ("", "0", "false", "False")


def _vprint(*args):
    if _HSA_VERBOSE:
        print(*args, flush=True)


def test_scheduler_continuous_batching_works_with_innerx_hsa_cuda():
    """
    True scheduler-level E2E test for continuous batching with FlashHSA (InnerX):

    - Start a real `Scheduler` (single process, TP=PP=1) using a tiny InnerX config and
      `load_format=dummy` so no real checkpoint is required.
    - Enqueue 3 requests while max_running_requests=2.
    - Run the scheduler step-by-step (no infinite event loop).
    - Assert decode batches exhibit continuous batching member replacement:
        {r1, r2} ... then later {r2, r3}
      where r1 is forced to finish first via max_new_tokens=1.
    """
    from sglang.srt.distributed.parallel_state import cleanup_dist_env_and_memory
    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.server_args import PortArgs, ServerArgs, set_global_server_args_for_scheduler

    # This test starts a real Scheduler/ModelRunner, which initializes torch.distributed
    # and model-parallel global groups. Other tests in the suite may also init these
    # globals without tearing them down, so we defensively clean up before/after.
    cleanup_dist_env_and_memory()

    # Build a minimal HF-like config folder for FlashHSA InnerX.
    # This keeps the test self-contained (no external checkpoint).
    with tempfile.TemporaryDirectory(prefix="sglang_innerx_sched_e2e_") as tmp:
        cfg = {
            "model_type": "flash_hsa_innerx",
            "architectures": ["HSAForCausalLM"],
            "vocab_size": 256,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 1,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 16,
            "rms_norm_eps": 1e-6,
            "attention_bias": False,
            "chunk_size": 4,
            "hsa_topk": 2,
            "hsa_mode": "sparse",
            "full_attn_interleave": 1,
            "hsa_heads": 4,
            "hsa_qk_ratio": 4,
            "enable_gate": False,
            "use_sliding_window_merging": True,
            "sliding_window_merging_size": 4,
            "use_sliding_window_attention": False,
            "tie_word_embeddings": False,
        }
        with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f)

        # Configure server args to make scheduling deterministic and lightweight.
        server_args = ServerArgs(model_path=tmp)
        server_args.load_format = "dummy"
        server_args.device = "cuda"
        server_args.dtype = "bfloat16"
        server_args.trust_remote_code = True
        server_args.skip_tokenizer_init = True
        server_args.disable_cuda_graph = True
        server_args.disable_overlap_schedule = True
        # Keep KV cache pool small for unit tests (avoid allocating most of GPU memory).
        server_args.mem_fraction_static = 0.02
        server_args.max_total_tokens = 256

        # Continuous batching capacity control (critical for this test).
        server_args.max_running_requests = 2
        server_args.max_prefill_tokens = 64
        server_args.max_queued_requests = 16
        server_args.context_length = 64

        # InnerX/FlashHSA runtime knobs.
        server_args.attention_backend = "hsa"
        server_args.page_size = 4
        server_args.hsa_lmk_id = int(cfg["vocab_size"])

        # Keep background features off.
        server_args.enable_metrics = False
        server_args.enable_trace = False
        server_args.sleep_on_idle = False
        server_args.disable_radix_cache = True
        server_args.speculative_algorithm = None
        server_args.enable_dp_attention = False
        server_args.torchao_config = ""
        server_args.tp_size = 1
        server_args.pp_size = 1
        server_args.dp_size = 1
        server_args.ep_size = 1

        # Some subsystems read global server args.
        set_global_server_args_for_scheduler(server_args)
        port_args = PortArgs.init_new(server_args)

        sched = None
        try:
            sched = Scheduler(
                server_args=server_args,
                port_args=port_args,
                gpu_id=0,
                tp_rank=0,
                moe_ep_rank=0,
                pp_rank=0,
                dp_rank=None,
            )

            # Enqueue three requests. r1 finishes first (max_new_tokens=1).
            def _mk_req(rid: str, max_new_tokens: int):
                return TokenizedGenerateReqInput(
                    rid=rid,
                    http_worker_ipc=None,
                    input_text="",
                    input_ids=[1, 2, 3, 4],
                    mm_inputs=None,
                    sampling_params=SamplingParams(
                        max_new_tokens=max_new_tokens,
                        stop=[],
                        stop_regex=[],
                        temperature=0.0,
                        top_p=1.0,
                    ),
                    return_logprob=False,
                    logprob_start_len=-1,
                    top_logprobs_num=0,
                    token_ids_logprob=[],
                    stream=False,
                )

            # Phase 1: enqueue r1/r2 first. We'll enqueue r3 later to exercise
            # "continuous batching" dynamics (r3 arrives while decode is running).
            for req in (_mk_req("r1", 2), _mk_req("r2", 6)):
                sched.handle_generate_request(req)

            # Grab Req objects for completion checks.
            rid_to_req = {r.rid: r for r in list(sched.waiting_queue)}
            assert {"r1", "r2"} <= set(rid_to_req.keys())

            decode_rid_sets = []
            max_steps = 128
            enqueued_r3 = False
            for _ in range(max_steps):
                batch = sched.get_next_batch_to_run()
                sched.cur_batch = batch
                if batch is not None:
                    if batch.forward_mode is not None and batch.forward_mode.is_decode():
                        rid_set = frozenset(r.rid for r in batch.reqs)
                        decode_rid_sets.append(rid_set)
                        # Enqueue r3 after we have observed the first decode step of {r1,r2}.
                        if (not enqueued_r3) and rid_set == frozenset({"r1", "r2"}):
                            sched.handle_generate_request(_mk_req("r3", 4))
                            # The new request should enter waiting_queue immediately.
                            for r in list(sched.waiting_queue):
                                if r.rid == "r3":
                                    rid_to_req["r3"] = r
                                    break
                            assert "r3" in rid_to_req, "Failed to capture Req object for r3."
                            enqueued_r3 = True

                    result = sched.run_batch(batch)
                    sched.process_batch_result(batch, result)
                else:
                    # idle
                    pass

                # Mirror the real scheduler event loop bookkeeping.
                sched.last_batch = batch

                # Stop when all finished.
                if enqueued_r3:
                    if all(rid_to_req[rid].finished() for rid in ("r1", "r2", "r3")):
                        break
                else:
                    if all(rid_to_req[rid].finished() for rid in ("r1", "r2")):
                        break

            _vprint("### scheduler continuous batching (decode rid sets)")
            _vprint(decode_rid_sets)

            assert decode_rid_sets, "No decode batch observed; scheduler did not run decode."

            # Must see replacement: {r1,r2} first, then later {r2,r3}.
            i12 = next(
                (i for i, s in enumerate(decode_rid_sets) if s == frozenset({"r1", "r2"})),
                None,
            )
            i23 = next(
                (i for i, s in enumerate(decode_rid_sets) if s == frozenset({"r2", "r3"})),
                None,
            )
            assert i12 is not None, f"Did not observe decode batch {{r1,r2}} in {decode_rid_sets}"
            assert i23 is not None, f"Did not observe decode batch {{r2,r3}} in {decode_rid_sets}"
            assert i12 < i23, f"Expected {{r1,r2}} before {{r2,r3}} but got {i12=} {i23=}"

            assert rid_to_req["r1"].finished()
            assert rid_to_req["r2"].finished()
            assert rid_to_req["r3"].finished()
        finally:
            # Ensure we don't leak global dist/model-parallel state across tests.
            cleanup_dist_env_and_memory()

