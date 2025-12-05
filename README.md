---
license: apache-2.0
language:
- en
pipeline_tag: text-to-image
library_name: diffusers
tags:
- next-dit
- text-to-image
- transformer
- image-generation
- Anime
base_model:
- NewBie-AI/NewBie-image-Exp0.1
---


<h1 align="center">NewBie image Exp0.1<br><sub><sup>Efficient Image Generation Base Model Based on Next-DiT</sup></sub></h1>

<div align="center">

[![GitHub-NewBie](https://img.shields.io/badge/GitHub-NewBie%20image%20Exp0.1-181717?logo=github&logoColor=white)](https://github.com/NewBieAI-Lab/NewBie-image-Exp0.1)&#160;
[![GitHub - LoRa Trainer](https://img.shields.io/badge/GitHub-LoRa%20Trainer-181717?logo=github&logoColor=white)](https://github.com/NewBieAI-Lab/NewbieLoraTrainer)&#160;
[![GitHub - ComfyUI-NewBie](https://img.shields.io/badge/GitHub-ComfyUI--NewBie-181717?logo=github&logoColor=white)](https://github.com/NewBieAI-Lab/ComfyUI-Newbie-V0.1)&#160;
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint-NewBie%20image%20Exp0.1-yellow)](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1)&#160;
[![MS](https://img.shields.io/badge/🤖%20Checkpoint-NewBie%20image%20Exp0%2E1-624aff)](https://www.modelscope.cn/models/NewBieAi-lab/NewBie-image-Exp0.1)
![C5BDBA2F0B1D85D81D3A9DCADF6DED1F](https://cdn-uploads.huggingface.co/production/uploads/67fdc3911c5d7301352a0507/qB2wyVrTuYBtg_ToRb2wP.jpeg)
</div>

## 🧱Exp0.1 Base
**NewBie image Exp0.1** is a **3.5B** parameter DiT model developed through research on the Lumina architecture.
Building on these insights, it adopts Next-DiT as the foundation to design a new NewBie architecture tailored for text-to-image generation.
The *NewBie image Exp0.1* model is trained within this newly constructed system, representing the first experimental release of the NewBie text-to-image generation framework.
#### Text Encoder
We use Gemma3-4B-it as the primary text encoder (conditioning on its second-to-last layer token embeddings), and extract Jina CLIP v2 pooled text features that are projected and fused into the model conditioning path. (into the time/AdaLN conditioning pathway).
Benefiting from Gemma3-4B-it and Jina CLIP v2, NewBie image Exp0.1 achieves strong prompt understanding and instruction-following capability.
#### VAE
Use the FLUX.1-dev 16channel VAE to encode images into latents, delivering richer, smoother color rendering and finer texture detail helping safeguard the stunning visual quality of NewBie image Exp0.1.

## 🖼️Task type
<div align="center">
  
**NewBie image Exp0.1** is pretrain on a large corpus of high-quality anime data, enabling the model to generate remarkably detailed and visually striking anime style images.
![NewBie image preview](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1/resolve/main/image/newbie_image.png)
We reformatted the dataset text into an **XML structured format** for our experiments. Empirically, this improved attention binding and attribute/element disentanglement, and also led to faster convergence.

Besides that, It also supports natural language and tags inputs.

**In multi character scenes, using XML structured prompt typically leads to more accurate image generation results.**
  </div>

<div style="display:flex; gap:16px; align-items:flex-start;">

  <div style="flex:1; min-width:0;">
    <details open style="box-sizing:border-box; border:1px solid #e5e7eb; border-radius:10px; padding:12px; height:260px; overflow:auto;">
      <summary><b>XML structured prompt</b></summary>

```prompt
  <character_1>
  <n>$character_1$</n>
  <gender>1girl</gender>
  <appearance>chibi, red_eyes, blue_hair, long_hair, hair_between_eyes, head_tilt, tareme, closed_mouth</appearance>
  <clothing>school_uniform, serafuku, white_sailor_collar, white_shirt, short_sleeves, red_neckerchief, bow, blue_skirt, miniskirt, pleated_skirt, blue_hat, mini_hat, thighhighs, grey_thighhighs, black_shoes, mary_janes</clothing>
  <expression>happy, smile</expression>
  <action>standing, holding, holding_briefcase</action>
  <position>center_left</position>
  </character_1>

  <character_2>
  <n>$character_2$</n>
  <gender>1girl</gender>
  <appearance>chibi, red_eyes, pink_hair, long_hair, very_long_hair, multi-tied_hair, open_mouth</appearance>
  <clothing>school_uniform, serafuku, white_sailor_collar, white_shirt, short_sleeves, red_neckerchief, bow, red_skirt, miniskirt, pleated_skirt, hair_bow, multiple_hair_bows, white_bow, ribbon_trim, ribbon-trimmed_bow, white_thighhighs, black_shoes, mary_janes, bow_legwear, bare_arms</clothing>
  <expression>happy, smile</expression>
  <action>standing, holding, holding_briefcase, waving</action>
  <position>center_right</position>
  </character_2>

  <general_tags>
  <count>2girls, multiple_girls</count>
  <style>anime_style, digital_art</style>
  <background>white_background, simple_background</background>
  <atmosphere>cheerful</atmosphere>
  <quality>high_resolution, detailed</quality>
  <objects>briefcase</objects>
  <other>alternate_costume</other>
  </general_tags>
```
</details> 

</div>
<div style="box-sizing:border-box; width:260px; height:260px; flex:0 0 260px; border:1px solid #e5e7eb; border-radius:10px; padding:12px; display:flex; align-items:center; justify-content:center;"> 
  <img src="https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1/resolve/main/image/XML_prompt_image.png" alt="XML prompt image" style="max-width:100%; max-height:100%; object-fit:contain; display:block;" /> 
</div>
</div>
<h1 align="center"><br><sub><sup>XML structured prompt and attribute/element disentanglement showcase</sup></sub></h1>
</div>

## 🧰Model Zoo
| Model | Hugging Face | ModelScope |
| :--- | :--- | :--- |
| **NewBie image Exp0.1** | [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint-NewBie%20image%20Exp0%2E1-yellow)](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1) | [![MS](https://img.shields.io/badge/🤖%20Checkpoint-NewBie%20image%20Exp0%2E1-624aff)](https://www.modelscope.cn/models/NewBieAi-lab/NewBie-image-Exp0.1) |
| **Gemma3-4B-it** | [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint-Gemma3--4B--it-yellow)](https://huggingface.co/google/gemma-3-4b-it) | [![MS](https://img.shields.io/badge/🤖%20Checkpoint-Gemma3--4B--it-624aff)](https://www.modelscope.cn/models/google/gemma-3-4b-it) |
| **Jina CLIP v2** | [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint-Jina%20CLIP%20v2-yellow)](https://huggingface.co/jinaai/jina-clip-v2) | [![MS](https://img.shields.io/badge/🤖%20Checkpoint-Jina%20CLIP%20v2-624aff)](https://www.modelscope.cn/models/jinaai/jina-clip-v2) |
| **FLUX.1-dev VAE** | [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20Checkpoint-FLUX%2E1--dev%20VAE-yellow)](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/vae/diffusion_pytorch_model.safetensors) | [![MS](https://img.shields.io/badge/🤖%20Checkpoint-FLUX%2E1--dev%20VAE-624aff)](https://www.modelscope.cn/models/black-forest-labs/FLUX.1-dev/tree/master/vae) |

## 🚀Quickstart
```bash
pip install diffusers transformers accelerate safetensors torch --upgrade
# Recommended: install FlashAttention and Triton according to your operating system.
```
```python
import torch
from diffusers import NewbiePipeline

def main():
    model_id = "NewBie-AI/NewBie-image-Exp0.1"

    # Load pipeline
    pipe = NewbiePipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
  # use float16 if your GPU does not support bfloat16

    prompt = "1girl"

    image = pipe(
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=28,
    ).images[0]

    image.save("newbie_sample.png")
    print("Saved to newbie_sample.png")

if __name__ == "__main__":
    main()
```

## 💪Training procedure
![NewBie image preview](https://huggingface.co/NewBie-AI/NewBie-image-Exp0.1/resolve/main/image/NewBie_image_Exp0.1_Training.png)