"""
Debug: capture SWA and HSA head outputs separately during decode
to isolate which head group causes the 3.0 error.
"""
import os, sys, types, logging, tempfile, json

# veomni mock
def _mock():
    import torch.nn as nn
    for sub in ["veomni","veomni.distributed","veomni.distributed.parallel_state",
                "veomni.distributed.sequence_parallel","veomni.utils","veomni.utils.logging",
                "veomni.utils.import_utils","veomni.models","veomni.models.module_utils","veomni.models.loader"]:
        sys.modules[sub] = types.ModuleType(sub)
    class _PS: sp_enabled=False
    sys.modules["veomni.distributed.parallel_state"].get_parallel_state=lambda:_PS()
    sys.modules["veomni.distributed.sequence_parallel"].slice_position_embedding=lambda x,**kw:x
    def _gl(n=None):
        l=logging.getLogger(n)
        if not hasattr(l,"info_rank0"):l.info_rank0=l.info
        if not hasattr(l,"warning_rank0"):l.warning_rank0=l.warning
        return l
    sys.modules["veomni.utils.logging"].get_logger=_gl
    sys.modules["veomni.utils.import_utils"].is_liger_kernel_available=lambda:True
    sys.modules["veomni.utils.import_utils"].is_torch_npu_available=lambda:False
    sys.modules["veomni.utils.import_utils"].is_transformers_version_greater_or_equal_to=lambda v:True
    sys.modules["veomni.models.module_utils"].GradientCheckpointingLayer=nn.Module
    class _FR:
        def register(self,n):return lambda f:f
    sys.modules["veomni.models.loader"].MODELING_REGISTRY=_FR()
    sys.modules["veomni.models.loader"].MODEL_CONFIG_REGISTRY=_FR()
_mock()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "InfiniteLongLM"))
for _n in ["models.SWANGPT","models.DRT","models.SWANNSA"]: sys.modules[_n]=types.ModuleType(_n)

import torch, torch.distributed as dist
from models.FlashHSA.modeling_hsa_lmk import HSAForCausalLM as OfficialModel, HierarchicalSparseAttention
from models.FlashHSA.configuration_hsa import HSAConfig
from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks

# Patches
_oi=HierarchicalSparseAttention.__init__
def _pi(s,c,l):_oi(s,c,l);s.num_key_value_groups=c.num_attention_heads//c.num_key_value_heads if not hasattr(s,"num_key_value_groups") else None
HierarchicalSparseAttention.__init__=_pi
import ops.topk_group as _tm, ops.hsa_fwd_bwd_group as _hm
_ot=_tm.online_topk_group; _oh=_hm.HSA_block_M_group
def _d(t):return torch.empty(t.shape,dtype=t.dtype,device=t.device).copy_(t)
_tm.online_topk_group=lambda q,k,*a,**kw:_ot(_d(q),_d(k),*a,**kw)
_hm.HSA_block_M_group=lambda q,k,v,*a,**kw:_oh(_d(q),_d(k),_d(v),*a,**kw)
from sglang.srt.layers import dp_attention as dpa; dpa.get_attention_tp_size=lambda:1

def _cmp(name, a, b):
    if a is None or b is None: print(f"  {name}: SKIP (None)"); return
    if a.shape!=b.shape:
        # Try to align
        if a.dim()==3 and b.dim()==2: b=b.unsqueeze(0)
        elif a.dim()==2 and b.dim()==3: a=a.unsqueeze(0)
        if a.shape!=b.shape: print(f"  {name}: shape {a.shape} vs {b.shape}"); return
    d=(a.float()-b.float()).abs()
    mx=d.max().item(); mn=d.mean().item()
    s="OK" if mx<0.05 else ("CLOSE" if mx<0.5 else "DIVERGED")
    print(f"  {name}: max={mx:.6f} mean={mn:.6f} [{s}]")

def main():
    device=torch.device("cuda"); dtype=torch.bfloat16; torch.manual_seed(42)
    CD=dict(vocab_size=256,hidden_size=1024,intermediate_size=4096,num_hidden_layers=1,
            num_attention_heads=16,num_key_value_heads=4,head_dim=64,rms_norm_eps=1e-6,
            chunk_size=64,hsa_topk=2,hsa_mode="sparse",full_attn_interleave=1,
            hsa_heads=8,hsa_qk_ratio=4,use_sliding_window=True,sliding_window=64,
            tie_word_embeddings=False,rope_theta=10000.0,hidden_act="silu")
    PS=64; VS=256; real_tokens=list(range(5,15))  # 10 tokens = 0 completed pages = SWA-only

    # ---- Official ----
    print("="*60); print("OFFICIAL MODEL (10 tokens, SWA-only)")
    oc=HSAConfig(**CD); oc._attn_implementation="eager"
    om=OfficialModel(oc).to(device=device,dtype=dtype); om.eval()
    for m in om.modules():
        if isinstance(m,HierarchicalSparseAttention):
            _tf=m.topk_func;_hf=m.hsa_func
            m.topk_func=lambda q,k,*a,_f=_tf,**kw:_f(_d(q),_d(k),*a,**kw)
            m.hsa_func=lambda q,k,v,*a,_f=_hf,**kw:_f(_d(q),_d(k),_d(v),*a,**kw)

    # Hook official HSA attention to capture internal values
    oa=None
    for m in om.modules():
        if isinstance(m,HierarchicalSparseAttention): oa=m; break

    off_cap={}
    _orig_off_fwd = oa.forward
    def _hook_off_fwd(hidden_states, attention_mask=None, position_ids=None,
                      past_key_value=None, output_attentions=None, use_cache=False,
                      cache_position=None, position_embeddings=None, shared_kv=None, **kwargs):
        # Call original
        result = _orig_off_fwd(hidden_states, attention_mask=attention_mask,
                                position_ids=position_ids, past_key_value=past_key_value,
                                output_attentions=output_attentions, use_cache=use_cache,
                                cache_position=cache_position, position_embeddings=position_embeddings,
                                shared_kv=shared_kv, **kwargs)
        # Capture the o_proj input (before projection)
        # We can't easily hook inside, but we can check if result matches expected patterns
        off_cap["attn_output"] = result[0].detach().clone() if isinstance(result, tuple) else result.detach().clone()
        return result
    oa.forward = _hook_off_fwd

    # Prefill
    ids_raw=torch.tensor([real_tokens],device=device,dtype=torch.long)
    ids_lmk=insert_special_tokens(ids_raw,VS,PS)
    pos_ids=create_position_ids_with_landmarks(len(real_tokens),PS,device)
    mask=torch.ones((1,ids_lmk.shape[1]),dtype=torch.long,device=device)
    with torch.no_grad(): off_pf=om(input_ids=ids_lmk,position_ids=pos_ids,attention_mask=mask,use_cache=True)

    # Decode
    di=torch.tensor([[42]],device=device,dtype=torch.long)
    L=ids_lmk.shape[1]; dp_val=L-(L//PS)
    dp_ids=torch.tensor([[dp_val]],device=device,dtype=torch.long)
    dm=torch.ones((1,L+1),dtype=torch.long,device=device)
    with torch.no_grad(): off_dc=om(input_ids=di,position_ids=dp_ids,attention_mask=dm,past_key_values=off_pf.past_key_values,use_cache=True)
    off_dec_logits=off_dc.logits[:,-1,:VS].float()
    off_attn=off_cap.get("attn_output")
    print(f"  Decode logits argmax: {off_dec_logits.argmax().item()}")
    print(f"  Attn output norm: {off_attn.float().norm().item():.4f}" if off_attn is not None else "  No attn captured")

    # ---- sglang ----
    print("\n"+"="*60); print("SGLANG MODEL (10 tokens, SWA-only)")
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.configs.flash_hsa import FlashHSAConfig
    from sglang.srt.layers.attention.hsa_backend import HSAAttnBackend
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch,ForwardMode,compute_position,compute_decode_positions_landmark
    from sglang.srt.models.flash_hsa import HSAForCausalLM as SM, FlashHSAInnerXHierarchicalSparseAttention as SHSA
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs,set_global_server_args_for_scheduler

    if not dist.is_initialized():
        _,p=tempfile.mkstemp(prefix="dbg_",suffix=".t")
        dist.init_process_group(backend="gloo",init_method=f"file://{p}",rank=0,world_size=1)
    if not ps.model_parallel_is_initialized():
        ps._WORLD=ps.init_world_group(ranks=[0],local_rank=0,backend="gloo")
        ps._TP=ps.init_model_parallel_group(group_ranks=[[0]],local_rank=0,backend="gloo",use_custom_allreduce=False,use_mscclpp_allreduce=False,use_torch_symm_mem_allreduce=False,group_name="tp")
        ps._PP=ps.init_model_parallel_group(group_ranks=[[0]],local_rank=0,backend="gloo",use_custom_allreduce=False,use_mscclpp_allreduce=False,use_torch_symm_mem_allreduce=False,group_name="pp")
    if getattr(dpa,"_ATTN_TP_RANK",None) is None:
        dpa._ATTN_TP_RANK=0;dpa._ATTN_TP_SIZE=1;dpa._ATTN_DP_RANK=0;dpa._ATTN_DP_SIZE=1
        dpa._LOCAL_ATTN_DP_RANK=0;dpa._LOCAL_ATTN_DP_SIZE=1;dpa._ENABLE_DP_ATTENTION_FLAG=False
        dpa._ATTN_TP_GROUP=ps.get_tp_group()
    try:
        sa=ServerArgs(model_path="dummy");sa.attention_backend="hsa";sa.enable_dp_lm_head=False
        set_global_server_args_for_scheduler(sa)
    except:pass

    cfg=FlashHSAConfig(model_type="flash_hsa_innerx",architectures=["HSAForCausalLM"],
        vocab_size=256,hidden_size=1024,intermediate_size=4096,num_hidden_layers=1,
        num_attention_heads=16,num_key_value_heads=4,head_dim=64,rms_norm_eps=1e-6,
        chunk_size=64,hsa_topk=2,hsa_mode="sparse",full_attn_interleave=1,hsa_heads=8,
        hsa_qk_ratio=4,use_sliding_window_merging=True,sliding_window_merging_size=64,
        tie_word_embeddings=False,rope_theta=10000.0,hidden_act="silu")
    sm=SM(cfg).to(device=device,dtype=dtype); sm.eval()
    # Transfer weights
    osd=om.state_dict();ssd=sm.state_dict()
    for n,p in ssd.items():
        if n in osd:
            s=osd[n]
            if s.shape==p.shape: p.data.copy_(s)
            elif "embed_tokens" in n or "lm_head" in n: p.data[:min(s.shape[0],p.shape[0])].copy_(s[:min(s.shape[0],p.shape[0])])
        elif "gate_up_proj" in n:
            gn=n.replace("gate_up_proj","gate_proj");un=n.replace("gate_up_proj","up_proj")
            if gn in osd and un in osd:
                f=torch.cat([osd[gn],osd[un]],dim=0)
                if f.shape==p.shape: p.data.copy_(f)
    for m in sm.modules():
        if hasattr(m,"cos_sin_cache") and m.cos_sin_cache is not None: m.cos_sin_cache=m.cos_sin_cache.to(torch.float32)

    lmk_id=256; mc=256
    r2t=torch.zeros((1,mc),dtype=torch.int32,device=device)
    pool=MHATokenToKVPool(size=512,page_size=PS,dtype=dtype,head_num=4,head_dim=64,layer_num=1,device=device,enable_memory_saver=False,enable_alt_stream=False)
    mr=types.SimpleNamespace(device=device,page_size=PS,sliding_window_size=None,model=sm,
        model_config=types.SimpleNamespace(is_encoder_decoder=False,context_len=mc,num_attention_heads=16,get_num_kv_heads=lambda tp:4//tp),
        hybrid_gdn_config=None,kimi_linear_config=None,gpu_id=0,
        server_args=types.SimpleNamespace(attention_backend="hsa",speculative_num_draft_tokens=0,speculative_num_steps=0,
            triton_attention_num_kv_splits=8,triton_attention_split_tile_size=None,enable_deterministic_inference=False,
            hsa_topk=None,hsa_selection_strategy=None,hsa_layers=None,hsa_window_size=None,hsa_enable_swa_merging=None,hsa_lmk_id=lmk_id))
    mr.req_to_token_pool=types.SimpleNamespace(size=1,req_to_token=r2t)
    mr.token_to_kv_pool=pool; mr.token_to_kv_pool_allocator=object()
    be=HSAAttnBackend(mr)

    # Prefill
    fi=Req._hsa_insert_lmk_prompt(real_tokens,page_size=PS,lmk_id=lmk_id)
    pl=len(fi)
    tl=torch.arange(0,mc,dtype=torch.int32,device=device); r2t[0,:pl]=tl[:pl]
    ep=torch.tensor([0],device=device,dtype=torch.int32);es=torch.tensor([pl],device=device,dtype=torch.int32)
    pos_s,esl=compute_position("hsa",ep,es,pl,page_size=PS,enable_landmark_positions=True)
    fb=ForwardBatch(forward_mode=ForwardMode.EXTEND,batch_size=1,
        input_ids=torch.tensor(fi,device=device,dtype=torch.int64),
        req_pool_indices=torch.tensor([0],device=device,dtype=torch.int32),
        seq_lens=torch.tensor([pl],device=device,dtype=torch.int32),
        out_cache_loc=tl[:pl].to(torch.int64),seq_lens_sum=pl,
        seq_lens_cpu=torch.tensor([pl],device="cpu",dtype=torch.int32),
        positions=pos_s,extend_prefix_lens=ep,extend_seq_lens=es,extend_start_loc=esl,
        extend_prefix_lens_cpu=[0],extend_seq_lens_cpu=[pl],
        req_to_token_pool=mr.req_to_token_pool,token_to_kv_pool=pool,attn_backend=be)
    be.init_forward_metadata(fb)
    with torch.no_grad(): sm.model(fb.input_ids,fb.positions,fb)

    # Decode - hook the RadixAttention to capture Q,K,V and output
    sg_attn=None
    for m in sm.modules():
        if isinstance(m,SHSA): sg_attn=m; break
    sg_radix=sg_attn.attn
    sg_cap={}
    def _h_radix(mod,args,kwargs,out):
        sg_cap["q"]=args[0].detach().clone()
        sg_cap["k"]=args[1].detach().clone()
        sg_cap["v"]=args[2].detach().clone()
        sg_cap["out"]=out.detach().clone()
    hr=sg_radix.register_forward_hook(_h_radix,with_kwargs=True)

    # Also hook the o_proj
    sg_oproj=sg_attn.o_proj
    def _h_oproj(mod,args,out):
        inp=args[0] if args else None
        sg_cap["oproj_in"]=inp.detach().clone() if inp is not None else None
        o=out[0] if isinstance(out,(tuple,list)) else out
        sg_cap["oproj_out"]=o.detach().clone()
    ho=sg_oproj.register_forward_hook(_h_oproj)

    dsl=pl+1; r2t[0,pl]=tl[pl]
    dp2=compute_decode_positions_landmark(torch.tensor([dsl],device=device,dtype=torch.int32),page_size=PS)
    fbd=ForwardBatch(forward_mode=ForwardMode.DECODE,batch_size=1,
        input_ids=torch.tensor([42],device=device,dtype=torch.int64),
        req_pool_indices=torch.tensor([0],device=device,dtype=torch.int32),
        seq_lens=torch.tensor([dsl],device=device,dtype=torch.int32),
        out_cache_loc=tl[pl:pl+1].to(torch.int64),seq_lens_sum=dsl,
        seq_lens_cpu=torch.tensor([dsl],device="cpu",dtype=torch.int32),
        positions=dp2,req_to_token_pool=mr.req_to_token_pool,token_to_kv_pool=pool,attn_backend=be)
    be.init_forward_metadata(fbd)
    with torch.no_grad(): sg_out=sm.model(fbd.input_ids,fbd.positions,fbd)
    hr.remove(); ho.remove()

    if isinstance(sg_out,tuple): sg_out=sg_out[0]
    sg_out=sm.model.norm(sg_out)
    sg_logits=(sg_out@sm.lm_head.weight.t())[:,:VS].float()
    print(f"  Decode logits argmax: {sg_logits.argmax().item()}")

    # ---- Compare per-head ----
    print("\n"+"="*60)
    print("HEAD-LEVEL COMPARISON")
    print("="*60)

    # sglang captures
    sq=sg_cap.get("q"); sk=sg_cap.get("k"); sv=sg_cap.get("v"); so=sg_cap.get("out")
    oproj_in=sg_cap.get("oproj_in")

    if sq is not None:
        print(f"\n  sglang RadixAttn input:")
        print(f"    q: {sq.shape} norm={sq.float().norm():.4f}")
        print(f"    k: {sk.shape} norm={sk.float().norm():.4f}")
        print(f"    v: {sv.shape} norm={sv.float().norm():.4f}")
        print(f"    out: {so.shape} norm={so.float().norm():.4f}")

        # Split Q into SWA and HSA heads
        HQ_swa=8; HQ_hsa=8; D=64
        q3=sq.view(-1,16,D)
        out3=so.view(-1,16,D)
        print(f"\n  sglang per-head-group output norms:")
        print(f"    SWA heads (0-7):  {out3[:,:HQ_swa,:].float().norm():.4f}")
        print(f"    HSA heads (8-15): {out3[:,HQ_swa:,:].float().norm():.4f}")

    if oproj_in is not None:
        print(f"\n  sglang o_proj input: {oproj_in.shape} norm={oproj_in.float().norm():.4f}")
        # This is the concatenated [SWA|HSA] output before projection
        opi3=oproj_in.view(-1,16,D)
        print(f"    SWA heads (0-7):  {opi3[:,:HQ_swa,:].float().norm():.4f}")
        print(f"    HSA heads (8-15): {opi3[:,HQ_swa:,:].float().norm():.4f}")

    # Compare logits
    print(f"\n  Logits comparison:")
    _cmp("decode logits", off_dec_logits.unsqueeze(0), sg_logits.unsqueeze(0))
    print(f"  Official argmax: {off_dec_logits.argmax().item()}")
    print(f"  sglang  argmax: {sg_logits.argmax().item()}")

if __name__=="__main__":
    main()
