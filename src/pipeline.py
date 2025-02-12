import gc, os, torch
from PIL.Image import Image as _I
from diffusers import FluxPipeline as _F, FluxTransformer2DModel as _M, AutoencoderKL as _A
from huggingface_hub.constants import HF_HUB_CACHE as _C
from pipelines.models import TextToImageRequest as _T
from torch import Generator as _G
from transformers import T5EncoderModel as _5, CLIPTextModel as _L
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe as _P

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
cp, rv = "black-forest-labs/FLUX.1-schnell", "741f7c3ce8b383c54771c7003378a50191e9efe9"

def A():
    a=_L.from_pretrained(cp,revision=rv,subfolder="text_encoder",local_files_only=True,torch_dtype=torch.bfloat16)
    b=_5.from_pretrained(cp,revision=rv,subfolder="text_encoder_2",local_files_only=True,torch_dtype=torch.bfloat16)
    c=_A.from_pretrained(cp,revision=rv,subfolder="vae",local_files_only=True,torch_dtype=torch.bfloat16)
    d=os.path.join(_C,"models--barneystinson--FLUX.1-schnell-int8wo/snapshots/b9fa75333f9319a48b411a2618f6f353966be599")
    e=_M.from_pretrained(d,torch_dtype=torch.bfloat16,use_safetensors=False)
    f=_F.from_pretrained(cp,revision=rv,local_files_only=True,text_encoder=a,text_encoder_2=b,transformer=e,vae=c,torch_dtype=torch.bfloat16).to("cuda")
    _P(f,residual_diff_threshold=1)
    f("")
    return f

def B(r, p):
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    g=_G(p.device).manual_seed(r.seed)
    return p(r.prompt,generator=g,guidance_scale=0.0,num_inference_steps=4,max_sequence_length=256,height=r.height,width=r.width).images[0]
