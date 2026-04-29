"""Compare official model with tilelang kernels vs PyTorch reference kernels."""
import sys,os,types,logging
for sub in ['veomni','veomni.distributed','veomni.distributed.parallel_state','veomni.distributed.sequence_parallel','veomni.utils','veomni.utils.logging','veomni.utils.import_utils','veomni.models','veomni.models.module_utils','veomni.models.loader']:sys.modules[sub]=types.ModuleType(sub)
class _PS:sp_enabled=False
sys.modules['veomni.distributed.parallel_state'].get_parallel_state=lambda:_PS()
sys.modules['veomni.distributed.sequence_parallel'].slice_position_embedding=lambda x,**kw:x
def _gl(n=None):l=logging.getLogger(n);l.info_rank0=l.info;l.warning_rank0=l.warning;return l
sys.modules['veomni.utils.logging'].get_logger=_gl
sys.modules['veomni.utils.import_utils'].is_liger_kernel_available=lambda:True
sys.modules['veomni.utils.import_utils'].is_torch_npu_available=lambda:False
sys.modules['veomni.utils.import_utils'].is_transformers_version_greater_or_equal_to=lambda v:True
import torch.nn as nn;sys.modules['veomni.models.module_utils'].GradientCheckpointingLayer=nn.Module
class FR:
    def register(s,n):return lambda f:f
sys.modules['veomni.models.loader'].MODELING_REGISTRY=FR();sys.modules['veomni.models.loader'].MODEL_CONFIG_REGISTRY=FR()
for n in ['models.SWANGPT','models.DRT','models.SWANNSA']:sys.modules[n]=types.ModuleType(n)
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'InfiniteLongLM'))

import torch
from models.FlashHSA.modeling_hsa_lmk import HSAForCausalLM as OM,HierarchicalSparseAttention as HA
from models.FlashHSA.configuration_hsa import HSAConfig as OC
from utils.landmark_utils import insert_special_tokens,create_position_ids_with_landmarks
from ops.hsa_fwd_bwd_group import hsa_torch_ref
from ops.topk_group import ref_topk_forward_with_grad

_oi=HA.__init__
def _pi(s,c,l):_oi(s,c,l);s.num_key_value_groups=c.num_attention_heads//c.num_key_value_heads if not hasattr(s,'num_key_value_groups') else None
HA.__init__=_pi

def _d(t):return torch.empty(t.shape,dtype=t.dtype,device=t.device).copy_(t)
import ops.topk_group as _tm,ops.hsa_fwd_bwd_group as _hm
_ot=_tm.online_topk_group;_oh=_hm.HSA_block_M_group
_tm.online_topk_group=lambda q,k,*a,**kw:_ot(_d(q),_d(k),*a,**kw)
_hm.HSA_block_M_group=lambda q,k,v,*a,**kw:_oh(_d(q),_d(k),_d(v),*a,**kw)

torch.manual_seed(42);device=torch.device('cuda');dtype=torch.bfloat16;PS=64;VS=256
cd=dict(vocab_size=256,hidden_size=1024,intermediate_size=4096,num_hidden_layers=1,
        num_attention_heads=16,num_key_value_heads=4,head_dim=64,rms_norm_eps=1e-6,
        chunk_size=64,hsa_topk=2,hsa_mode='sparse',full_attn_interleave=1,hsa_heads=8,
        hsa_qk_ratio=4,use_sliding_window=True,sliding_window=64,tie_word_embeddings=False,
        rope_theta=10000.0,hidden_act='silu')
oc=OC(**cd);oc._attn_implementation='eager'

om_tl=OM(oc).to(device=device,dtype=dtype);om_tl.eval()
om_pt=OM(oc).to(device=device,dtype=dtype);om_pt.eval()
om_pt.load_state_dict(om_tl.state_dict())

for mod in om_tl.modules():
    if isinstance(mod,HA):
        _tf=mod.topk_func;_hf=mod.hsa_func
        mod.topk_func=lambda q,k,*a,_f=_tf,**kw:_f(_d(q),_d(k),*a,**kw)
        mod.hsa_func=lambda q,k,v,*a,_f=_hf,**kw:_f(_d(q),_d(k),_d(v),*a,**kw)

for mod in om_pt.modules():
    if isinstance(mod,HA):
        def _ref_topk(q,k,topk,block_size,window_size,is_causal=False,memory_window_size=-1,q_offset=0):
            S=k.shape[1]
            eff_topk=min(topk,S)
            idx,sc=ref_topk_forward_with_grad(q,k,eff_topk,block_size,window_size,is_causal,q.dtype,memory_window_size=memory_window_size,q_offset=q_offset)
            if eff_topk<topk:
                B,L,H,_=idx.shape;dev=idx.device
                pad_i=torch.full((B,L,H,topk-eff_topk),-1,dtype=idx.dtype,device=dev)
                pad_s=torch.full((B,L,H,topk-eff_topk),float('-inf'),dtype=sc.dtype,device=dev)
                idx=torch.cat([idx,pad_i],dim=-1);sc=torch.cat([sc,pad_s],dim=-1)
            return idx,sc
        mod.topk_func=_ref_topk
        _cs=mod.chunk_size;_sc=mod.scaling
        def _ref_hsa(q,k,v,weights,indices,block_size,mask_last_token=False,_cs=_cs,_sc=_sc):
            # weights has K+1 (or K+2) entries; indices has K. Slice weights to match.
            K=indices.shape[-1]
            w=weights[...,:K]
            out=hsa_torch_ref(q,k,v,w,indices,chunk_size=block_size,sm_scale=_sc,block_q=1,mask_last_token=mask_last_token)
            return out.to(q.dtype)
        mod.hsa_func=_ref_hsa

real_tokens=list(range(5,70))  # 65 real
ids=insert_special_tokens(torch.tensor([real_tokens]),VS,PS)
pos=create_position_ids_with_landmarks(len(real_tokens),PS,device)
mask=torch.ones((1,ids.shape[1]),dtype=torch.long,device=device)

with torch.no_grad():
    out_tl=om_tl(input_ids=ids.to(device),position_ids=pos.to(device),attention_mask=mask,use_cache=False)
    out_pt=om_pt(input_ids=ids.to(device),position_ids=pos.to(device),attention_mask=mask,use_cache=False)

l_tl=out_tl.logits[0,:,:VS].float()
l_pt=out_pt.logits[0,:,:VS].float()
diff=(l_tl-l_pt).abs()
print(f'TILELANG vs PYTORCH-REF (65 real -> {ids.shape[1]} eng tokens)')
print(f'  Overall: max={diff.max().item():.6f} mean={diff.mean().item():.6f}')
for ps2 in range(0,l_tl.shape[0],64):
    pe=min(ps2+64,l_tl.shape[0])
    pd=diff[ps2:pe]
    print(f'  Page {ps2//64} (pos {ps2}-{pe-1}): max={pd.max().item():.6f}')
for i in [62,63,64,65]:
    if i<l_tl.shape[0]:
        m='OK' if l_tl[i].argmax()==l_pt[i].argmax() else 'MISS'
        print(f'  Pos {i}: tl={l_tl[i].argmax().item()} pt={l_pt[i].argmax().item()} {m} err={diff[i].max().item():.6f}')
