import argparse
import os
import builtins
import json
import math
import multiprocessing as mp
import os
import random
import socket
import traceback

import gradio as gr
import numpy as np
from safetensors.torch import load_file
import torch
from torchvision.transforms.functional import to_pil_image

from imgproc import generate_crop_size_list
import models
from transport import Sampler, create_transport

from multiprocessing import Process,Queue,set_start_method,get_context

class ModelFailure:
    pass

def encode_prompt(prompt_batch, text_encoder, tokenizer, clip_model, clip_tokenizer, proportion_empty_prompts, is_train=True, clip_captions=None):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    if clip_captions is None:
        clip_captions = captions

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding=True,
            pad_to_multiple_of=8,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask

        prompt_embeds = text_encoder(
            input_ids=text_input_ids.cuda(),
            attention_mask=prompt_masks.cuda(),
            output_hidden_states=True,
        ).hidden_states[-2]

        clip_inputs = clip_tokenizer(
            clip_captions,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        ).to(clip_model.device)

        clip_text_pooled, clip_text_sequence = clip_model.get_text_features(input_ids=clip_inputs)

        if clip_text_sequence is not None:
            clip_text_sequence = clip_text_sequence.clone()
        if clip_text_pooled is not None:
            clip_text_pooled = clip_text_pooled.clone()

        clip_mask = clip_inputs.attention_mask

    return prompt_embeds, prompt_masks, clip_text_sequence, clip_text_pooled, clip_mask



@torch.no_grad()
def model_main(args, master_port, rank, request_queue, response_queue, mp_barrier):
    # import here to avoid huggingface Tokenizer parallelism warnings
    from diffusers.models import AutoencoderKL
    from transformers import AutoModel, AutoTokenizer, AutoConfig

    # override the default print function since the delay can be large for child process
    original_print = builtins.print

    # Redefine the print function with flush=True by default
    def print(*args, **kwargs):
        kwargs.setdefault("flush", True)
        original_print(*args, **kwargs)

    # Override the built-in print with the new version
    builtins.print = print

    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.num_gpus)

    train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"), weights_only=False)
    print("Loaded model arguments:", json.dumps(train_args.__dict__, indent=2))
    print(f"Creating lm: Gemma-3-4B")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]

    text_encoder = AutoModel.from_pretrained(
        "google/gemma-3-4b-it", torch_dtype=dtype, token=args.hf_token
    ).to("cuda:0").eval()
    cap_feat_dim = text_encoder.config.text_config.hidden_size

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it", token=args.hf_token)
    tokenizer.padding_side = "right"

    print(f"Loading CLIP model: jinaai/jina-clip-v2")
    clip_config = AutoConfig.from_pretrained(
        "jinaai/jina-clip-v2",
        trust_remote_code=True
    )
    clip_tokenizer = AutoTokenizer.from_pretrained(
        "jinaai/jina-clip-v2",
        trust_remote_code=True
    )

    print("Loading CLIP model...")

    clip_model = AutoModel.from_pretrained(
        "jinaai/jina-clip-v2",
        torch_dtype=dtype,
        config=clip_config,
        trust_remote_code=True,
        token=args.hf_token
    ).to("cuda:0").eval()

    if args.num_gpus > 1:
        raise NotImplementedError("Inference with >1 GPUs not yet supported")

    vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae", token=args.hf_token).cuda()

    print(f"Creating DiT: {train_args.model}")
    model = models.__dict__[train_args.model](
        in_channels=16,
        qk_norm=train_args.qk_norm,
        cap_feat_dim=cap_feat_dim,
    )
    model.eval().to("cuda", dtype=dtype)

    assert train_args.model_parallel_size == args.num_gpus
    if args.ema:
        print("Loading ema model.")
    print('load model')
    ckpt_path = os.path.join(
        args.ckpt,
        f"consolidated{'_ema' if args.ema else ''}.{rank:02d}-of-{args.num_gpus:02d}.safetensors",
    )
    if os.path.exists(ckpt_path):
        ckpt = load_file(ckpt_path)
    else:
        ckpt_path = os.path.join(
            args.ckpt,
            f"consolidated{'_ema' if args.ema else ''}.{rank:02d}-of-{args.num_gpus:02d}.pth",
        )
        assert os.path.exists(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(ckpt, strict=True)
    print('load model finish')
    mp_barrier.wait()

    with torch.autocast("cuda", dtype):
        while True:
            (
                cap,
                neg_cap,
                system_type,
                resolution,
                num_sampling_steps,
                cfg_scale,
                cfg_trunc,
                renorm_cfg,
                solver,
                t_shift,
                seed,
                scaling_method,
                scaling_watershed,
                proportional_attn,
            ) = request_queue.get()

            system_prompt = system_type
            original_cap = cap
            original_neg_cap = neg_cap

            cap = system_prompt + cap
            if neg_cap != "":
                neg_cap = system_prompt + neg_cap

            metadata = dict(
                real_cap=cap,
                real_neg_cap=neg_cap,
                system_type=system_type,
                resolution=resolution,
                num_sampling_steps=num_sampling_steps,
                cfg_scale=cfg_scale,
                cfg_trunc=cfg_trunc,
                renorm_cfg=renorm_cfg,
                solver=solver,
                t_shift=t_shift,
                seed=seed,
                scaling_method=scaling_method,
                scaling_watershed=scaling_watershed,
                proportional_attn=proportional_attn,
            )
            print("> params:", json.dumps(metadata, indent=2))

            try:
                resolution = resolution.split(" ")[-1]
                w, h = resolution.split("x")
                w, h = int(w), int(h)
                latent_w, latent_h = w // 8, h // 8
                if int(seed) != 0:
                    torch.random.manual_seed(int(seed))
                z = torch.randn([1, 16, latent_h, latent_w], device="cuda").to(dtype)
                z = z.repeat(2, 1, 1, 1)  # [cond, uncond]

                if neg_cap != "":
                    cap_feats, cap_mask, clip_text_sequence, clip_text_pooled, clip_mask = encode_prompt(
                        [cap] + [neg_cap], text_encoder, tokenizer, clip_model, clip_tokenizer, 0.0,
                        clip_captions=[original_cap] + [original_neg_cap]
                    )
                else:
                    cap_feats, cap_mask, clip_text_sequence, clip_text_pooled, clip_mask = encode_prompt(
                        [cap] + [""], text_encoder, tokenizer, clip_model, clip_tokenizer, 0.0,
                        clip_captions=[original_cap] + [""]
                    )

                cap_mask = cap_mask.to(cap_feats.device)

                model_kwargs = dict(
                    cap_feats=cap_feats,
                    cap_mask=cap_mask,
                    cfg_scale=cfg_scale,
                    cfg_trunc=cfg_trunc,
                    renorm_cfg=renorm_cfg,
                    clip_text_sequence=clip_text_sequence,
                    clip_text_pooled=clip_text_pooled,
                    clip_img_pooled=None
                )

                # begin sampler
                if solver == "dpm":
                    transport = create_transport(
                        "Linear",
                        "velocity",
                    )
                    sampler = Sampler(transport)
                    sample_fn = sampler.sample_dpm(
                        model.forward_with_cfg,
                        model_kwargs=model_kwargs,
                    )
                else:
                    transport = create_transport(
                        args.path_type,
                        args.prediction,
                        args.loss_weight,
                        args.train_eps,
                        args.sample_eps,
                    )
                    sampler = Sampler(transport)
                    sample_fn = sampler.sample_ode(
                        sampling_method=solver,
                        num_steps=num_sampling_steps,
                        atol=args.atol,
                        rtol=args.rtol,
                        reverse=args.reverse,
                        time_shifting_factor=t_shift,
                    )
                    # end sampler

                print(f"> caption: {cap}")
                print(f"> num_sampling_steps: {num_sampling_steps}")
                print(f"> cfg_scale: {cfg_scale}")
                print("> start sample")
                if solver == "dpm":
                    samples = sample_fn(z, steps=num_sampling_steps, order=2, skip_type="time_uniform_flow", method="multistep", flow_shift=t_shift)
                else:
                    samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
                samples = samples[:1]
                print("smaple_dtype", samples.dtype)

                vae_scale = {
                    "sdxl": 0.13025,
                    "sd3": 1.5305,
                    "ema": 0.18215,
                    "mse": 0.18215,
                    "cogvideox": 1.15258426,
                    "flux": 0.3611,
                }["flux"]
                vae_shift = {
                    "sdxl": 0.0,
                    "sd3": 0.0609,
                    "ema": 0.0,
                    "mse": 0.0,
                    "cogvideox": 0.0,
                    "flux": 0.1159,
                }["flux"]
                print(f"> vae scale: {vae_scale}, shift: {vae_shift}")
                print("samples.shape", samples.shape)
                samples = vae.decode(samples / vae_scale + vae_shift).sample
                samples = (samples + 1.0) / 2.0
                samples.clamp_(0.0, 1.0)

                img = to_pil_image(samples[0, :].float())
                print("> generated image, done.")

                if response_queue is not None:
                    response_queue.put((img, metadata))

            except Exception:
                print(traceback.format_exc())
                response_queue.put(ModelFailure())


def none_or_str(value):
    if value == "None":
        return None
    return value


def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument(
        "--path-type",
        type=str,
        default="Linear",
        choices=["Linear", "GVP", "VP"],
        help="the type of path for transport: 'Linear', 'GVP' (Geodesic Vector Pursuit), or 'VP' (Vector Pursuit).",
    )
    group.add_argument(
        "--prediction",
        type=str,
        default="velocity",
        choices=["velocity", "score", "noise"],
        help="the prediction model for the transport dynamics.",
    )
    group.add_argument(
        "--loss-weight",
        type=none_or_str,
        default=None,
        choices=[None, "velocity", "likelihood"],
        help="the weighting of different components in the loss function, can be 'velocity' for dynamic modeling, 'likelihood' for statistical consistency, or None for no weighting.",
    )
    group.add_argument("--sample-eps", type=float, help="sampling in the transport model.")
    group.add_argument("--train-eps", type=float, help="training to stabilize the learning process.")


def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for the ODE solver.",
    )
    group.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for the ODE solver.",
    )
    group.add_argument("--reverse", action="store_true", help="run the ODE solver in reverse.")
    group.add_argument(
        "--likelihood",
        action="store_true",
        help="Enable calculation of likelihood during the ODE solving process.",
    )


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default='', required=False)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--hf_token", type=str, default=None, help="huggingface read token for accessing gated repo.")
    parser.add_argument("--res", type=int, default=1024, choices=[256, 512, 1024])
    parser.add_argument("--port", type=int, default=5995)

    parse_transport_args(parser)
    parse_ode_args(parser)

    args = parser.parse_known_args()[0]

    if args.num_gpus != 1:
        raise NotImplementedError("Multi-GPU Inference is not yet supported")

    master_port = find_free_port()
    processes = []
    request_queues = []
    response_queue = mp.Queue()
    mp_barrier = mp.Barrier(args.num_gpus + 1)
    for i in range(args.num_gpus):
        request_queues.append(mp.Queue())
        p = mp.Process(
            target=model_main,
            args=(
                args,
                master_port,
                i,
                request_queues[i],
                response_queue if i == 0 else None,
                mp_barrier,
            ),
        )
        p.start()
        processes.append(p)

    description = args.ckpt.split('/')[-1]

    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                cap = gr.Textbox(
                    lines=2,
                    label="Caption",
                    interactive=True,
                    value="A charcoal sketch of Istanbul with the iconic Hagia Sophia Mosque. The city streets wind through the landscape, with bright houses, trees, and flowers. There are puddles with raindrops and reflections of light and shadow. The Hagia Sophia stands tall and is crafted with precision. The background contains mountains and a bridge. The artist's signature 'brewozxy7' and the date 'October 2024' are in the lower left corner.",
                    placeholder="Enter a caption.",
                )
                neg_cap = gr.Textbox(
                    lines=2,
                    label="Negative Caption",
                    interactive=True,
                    value="",
                    placeholder="Enter a negative caption.",
                )
                default_value = "You are an assistant designed to generate high-quality images based on user prompts."
                system_type = gr.Dropdown(
                    value=default_value,
                    choices=[
                        "You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts.",
                        "You are an assistant designed to generate high-quality images based on user prompts.",
                        "You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts.",
                    ],
                    label="System Type",
                )
                with gr.Row():
                    res_choices = [f"{w}x{h}" for w, h in generate_crop_size_list((args.res // 64) ** 2, 64)]
                    default_value = "1024x1024"

                    resolution = gr.Dropdown(
                        value=default_value, choices=res_choices, label="Resolution"
                    )
                with gr.Row():
                    num_sampling_steps = gr.Slider(
                        minimum=1,
                        maximum=70,
                        value=18,
                        step=1,
                        interactive=True,
                        label="Sampling steps",
                    )
                    seed = gr.Slider(
                        minimum=0,
                        maximum=int(1e5),
                        value=0,
                        step=1,
                        interactive=True,
                        label="Seed (0 for random)",
                    )
                    cfg_trunc = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.25,
                        step=0.01,
                        interactive=True,
                        label="CFG Truncation",
                    )
                with gr.Row():
                    solver = gr.Dropdown(
                        value="midpoint",
                        choices=["euler", "midpoint", "rk4", "dpm"],
                        label="solver",
                    )
                    t_shift = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=6,
                        step=1,
                        interactive=True,
                        label="Time shift",
                    )
                    cfg_scale = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=5.0,
                        interactive=True,
                        label="CFG scale",
                    )
                with gr.Row():
                    renorm_cfg = gr.Dropdown(
                        value="0.0",
                        choices=["1.0", "0.0", "2.0"],
                        label="CFG Renorm",
                    )
                with gr.Accordion("Advanced Settings for Resolution Extrapolation", open=False):
                    with gr.Row():
                        scaling_method = gr.Dropdown(
                            value="Time-aware",
                            choices=["Time-aware", "None"],
                            label="RoPE scaling method",
                        )
                        scaling_watershed = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.3,
                            interactive=True,
                            label="Linear/NTK watershed",
                        )
                    with gr.Row():
                        proportional_attn = gr.Checkbox(
                            value=True,
                            interactive=True,
                            label="Proportional attention",
                        )
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
            with gr.Column():
                output_img = gr.Image(
                    label="Generated image",
                    interactive=False,
                )
                with gr.Accordion(label="Generation Parameters", open=True):
                    gr_metadata = gr.JSON(label="metadata", show_label=False)

        with gr.Row():
            prompts = [
                "Close-up portrait of a young woman with light brown hair, looking to the right, illuminated by warm, golden sunlight. Her hair is gently tousled, catching the light and creating a halo effect around her head. She wears a white garment with a V-neck, visible in the lower left of the frame. The background is dark and out of focus, enhancing the contrast between her illuminated face and the shadows. Soft, ethereal lighting, high contrast, warm color palette, shallow depth of field, natural backlighting, serene and contemplative mood, cinematic quality, intimate and visually striking composition.",
                "一个剑客，武侠风，红色腰带，戴着斗笠，低头，盖住眼睛，白色背景，细致，精品，杰作，水墨画，墨烟，墨云，泼墨，色带，墨水，墨黑白莲花，光影艺术，笔触。",
                "Aesthetic photograph of a bouquet of pink and white ranunculus flowers in a clear glass vase, centrally positioned on a wooden surface. The flowers are in full bloom, displaying intricate layers of petals with a soft gradient from pale pink to white. The vase is filled with water, visible through the clear glass, and the stems are submerged. In the background, a blurred vase with green stems is partially visible, adding depth to the composition. The lighting is warm and natural, casting soft shadows and highlighting the delicate textures of the petals. The scene is serene and intimate, with a focus on the organic beauty of the flowers. Photorealistic, shallow depth of field, soft natural lighting, warm color palette, high contrast, glossy texture, tranquil, visually balanced."
            ]
            prompts = [[_] for _ in prompts]
            gr.Examples(
                prompts,
                [cap],
                label="Examples",
            )

        def on_submit(*args):
            for q in request_queues:
                q.put(args)
            result = response_queue.get()
            if isinstance(result, ModelFailure):
                raise RuntimeError
            img, metadata = result

            return img, metadata

        submit_btn.click(
            on_submit,
            [
                cap,
                neg_cap,
                system_type,
                resolution,
                num_sampling_steps,
                cfg_scale,
                cfg_trunc,
                renorm_cfg,
                solver,
                t_shift,
                seed,
                scaling_method,
                scaling_watershed,
                proportional_attn,
            ],
            [output_img, gr_metadata],
        )

        def show_scaling_watershed(scaling_m):
            return gr.update(visible=scaling_m == "Time-aware")

        scaling_method.change(show_scaling_watershed, scaling_method, scaling_watershed)

    mp_barrier.wait()
    demo.queue().launch(
        share=True,
        server_name="0.0.0.0",
        server_port=args.port
    )


if __name__ == "__main__":
    mp.set_start_method("fork")
    main()