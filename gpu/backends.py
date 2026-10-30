"""Hardware-specific SDXL inference used by the outbound GPU worker.

This module deliberately has no FastAPI, database, or MinIO dependency.  It
loads a model and produces JPEG bytes; the worker process is responsible for
talking to the app backend.
"""

from __future__ import annotations

import asyncio
import io
import platform
import sys
from collections.abc import Awaitable, Callable

import numpy as np
from PIL import Image as PILImg
from tqdm.std import tqdm

from gpu.errors import InferenceCancelled

IS_APPLE_METAL = sys.platform == "darwin" and platform.machine().lower() == "arm64"

if IS_APPLE_METAL:
    import mlx.core as mx  # type: ignore

    from gpu.imagen_metal.stable_diffusion import StableDiffusionXL  # type: ignore
else:
    import torch  # type: ignore
    from diffusers import (  # type: ignore
        AutoencoderKL,
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        HeunDiscreteScheduler,
        KDPM2AncestralDiscreteScheduler,
        KDPM2DiscreteScheduler,
        LMSDiscreteScheduler,
        StableDiffusionXLPipeline,
        UniPCMultistepScheduler,
    )
    from transformers import CLIPTextModel, CLIPTextModelWithProjection  # type: ignore
    from transformers.modeling_outputs import BaseModelOutputWithPooling  # type: ignore


StepCallback = Callable[[int, int], Awaitable[bool]]


async def _report_step(callback: StepCallback | None, step: int, total: int) -> None:
    if callback is not None:
        keep_going = await callback(step, total)
        if not keep_going:
            raise InferenceCancelled()


class LoadedSDXL:
    def __init__(self, model: str):
        self._sdxl = None
        if IS_APPLE_METAL:
            mx.set_cache_limit(1 * 1024 * 1024 * 1024)
            self._sdxl = StableDiffusionXL(model=model, float16=True)
            self._sdxl.ensure_models_are_loaded()
        else:
            torch.use_deterministic_algorithms(True, warn_only=True)
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "xpu"
                if hasattr(torch, "xpu") and torch.xpu.is_available()
                else "cpu"
            )
            self._schedulers = {
                "euler": EulerDiscreteScheduler,
                "euler_a": EulerAncestralDiscreteScheduler,
                "unipc": UniPCMultistepScheduler,
                "dpm": DPMSolverMultistepScheduler,
                "dpm_singlestep": DPMSolverSinglestepScheduler,
                "heun": HeunDiscreteScheduler,
                "lms": LMSDiscreteScheduler,
                "kdpm2": KDPM2DiscreteScheduler,
                "kdpm2_a": KDPM2AncestralDiscreteScheduler,
            }

            text_encoder = CLIPTextModel.from_pretrained(
                model, subfolder="text_encoder", dtype=torch.float32
            ).to(self.device)  # type: ignore
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                model, subfolder="text_encoder_2", dtype=torch.float32
            ).to(self.device)  # type: ignore
            vae = AutoencoderKL.from_pretrained(
                model, subfolder="vae", torch_dtype=torch.float32
            ).to(self.device)  # type: ignore
            self._sdxl = StableDiffusionXLPipeline.from_pretrained(
                model,
                torch_dtype=torch.float16,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                vae=vae,
            ).to(self.device)

    def _tokenize_text(self, text: str, tokenizer):
        raw_text_chunks = [chunk.strip(" ,") for chunk in text.split("BREAK")]
        token_chunks_list = tokenizer(
            raw_text_chunks,
            truncation=False,
            add_special_tokens=False,
        ).input_ids
        token_chunks_without_special_tokens = [
            token_chunks[i : i + 75]
            for token_chunks in token_chunks_list
            for i in range(0, len(token_chunks), 75)
        ]
        token_chunks_with_special_tokens = [
            [tokenizer.bos_token_id]
            + token_chunk
            + [tokenizer.eos_token_id]
            + [tokenizer.pad_token_id] * (75 - len(token_chunk))
            for token_chunk in token_chunks_without_special_tokens
        ]
        input_ids = torch.tensor(
            token_chunks_with_special_tokens,
            dtype=torch.long,
            device=self.device,
        )
        return input_ids

    def _encode_text(self, input_ids, encoder, clip_skip: int):
        with torch.no_grad():
            outputs = encoder(input_ids, output_hidden_states=True)
        text_embeds = (
            None
            if isinstance(outputs, BaseModelOutputWithPooling)
            else outputs.text_embeds[0:1]
        )
        hidden_states = (
            encoder.text_model.final_layer_norm(outputs.hidden_states[-2])
            if clip_skip == 1
            else outputs.hidden_states[-clip_skip]
        )
        hidden_states = hidden_states.view(1, -1, hidden_states.size(-1))
        return hidden_states, text_embeds

    async def infer(
        self,
        promise_id: str,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        steps: int,
        seed: int,
        width: int,
        height: int,
        clip_skip: int,
        scheduler: str,
        on_step: StepCallback | None = None,
    ) -> bytes:
        if IS_APPLE_METAL:
            return await self._infer_metal(
                prompt,
                negative_prompt,
                guidance_scale,
                steps,
                seed,
                width,
                height,
                on_step,
            )
        return await self._infer_torch(
            prompt,
            negative_prompt,
            guidance_scale,
            steps,
            seed,
            width,
            height,
            clip_skip,
            scheduler,
            on_step,
        )

    async def _infer_metal(
        self,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        steps: int,
        seed: int,
        width: int,
        height: int,
        on_step: StepCallback | None,
    ) -> bytes:
        if not self._sdxl:
            raise ValueError("model is not loaded")
        sdxl: StableDiffusionXL = self._sdxl  # type: ignore
        latents = sdxl.generate_latents(
            prompt,
            negative_text=negative_prompt,
            n_images=1,
            cfg_weight=guidance_scale,
            num_steps=steps,
            seed=seed,
            latent_size=(height // 8, width // 8),
            orig_size=(height, width),
        )
        final_latent = None
        i = 1
        for latent_t in tqdm(latents, "iteration", total=steps):
            mx.eval(latent_t)
            final_latent = latent_t
            await _report_step(on_step, i, steps)
            i += 1
            await asyncio.sleep(0)
        if final_latent is None:
            raise RuntimeError("model produced no latents")
        decoded = sdxl.decode(final_latent)
        mx.eval(decoded)
        await asyncio.sleep(0)
        img_array = (decoded * 255).astype(mx.uint8)
        np_array = np.array(img_array)
        pil_img = PILImg.fromarray(np_array[-1])
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()

    async def _infer_torch(
        self,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        steps: int,
        seed: int,
        width: int,
        height: int,
        clip_skip: int,
        scheduler: str,
        on_step: StepCallback | None,
    ) -> bytes:
        if not self._sdxl:
            raise ValueError("model is not loaded")

        Scheduler = self._schedulers.get(scheduler)
        if not Scheduler:
            raise ValueError(f"No scheduler found: {scheduler}")
        local_scheduler = Scheduler.from_config(self._sdxl.scheduler.config)

        sdxl: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pipe(
            self._sdxl, scheduler=local_scheduler, torch_dtype=torch.float16
        )

        tokenized_prompt = self._tokenize_text(prompt, sdxl.tokenizer)
        tokenized_prompt_2 = self._tokenize_text(prompt, sdxl.tokenizer_2)
        tokenized_neg_prompt = self._tokenize_text(negative_prompt, sdxl.tokenizer)
        tokenized_neg_prompt_2 = self._tokenize_text(negative_prompt, sdxl.tokenizer_2)

        padding_size = abs(tokenized_prompt.shape[0] - tokenized_neg_prompt.shape[0])
        if padding_size > 0:
            padding_tensor_1 = (
                torch.tensor([sdxl.tokenizer.eos_token_id] * padding_size * 77)
                .reshape(padding_size, 77)
                .to(self.device)
            )
            padding_tensor_2 = (
                torch.tensor([sdxl.tokenizer_2.eos_token_id] * padding_size * 77)
                .reshape(padding_size, 77)
                .to(self.device)
            )
            should_neg_pad = tokenized_prompt.shape[0] > tokenized_neg_prompt.shape[0]
            if should_neg_pad:
                tokenized_neg_prompt = torch.cat(
                    [tokenized_neg_prompt, padding_tensor_1], dim=0
                )
                tokenized_neg_prompt_2 = torch.cat(
                    [tokenized_neg_prompt_2, padding_tensor_2], dim=0
                )
            else:
                tokenized_prompt = torch.cat(
                    [tokenized_prompt, padding_tensor_1], dim=0
                )
                tokenized_prompt_2 = torch.cat(
                    [tokenized_prompt_2, padding_tensor_2], dim=0
                )

        prompt_embeds, _ = self._encode_text(
            tokenized_prompt, sdxl.text_encoder, clip_skip
        )
        prompt_embeds_2, cond_pooled_text_embeds = self._encode_text(
            tokenized_prompt_2, sdxl.text_encoder_2, clip_skip
        )
        cond_prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)

        neg_prompt_embeds, _ = self._encode_text(
            tokenized_neg_prompt, sdxl.text_encoder, clip_skip
        )
        neg_prompt_embeds_2, uncond_pooled_text_embeds = self._encode_text(
            tokenized_neg_prompt_2, sdxl.text_encoder_2, clip_skip
        )
        uncond_prompt_embeds = torch.cat(
            [neg_prompt_embeds, neg_prompt_embeds_2], dim=-1
        )

        crossattend_text_embeds = torch.cat(
            [uncond_prompt_embeds, cond_prompt_embeds], dim=0
        )
        pooled_text_embeds = torch.cat(
            [uncond_pooled_text_embeds, cond_pooled_text_embeds], dim=0
        )
        time_ids = torch.tensor(
            [[height, width, 0, 0, height, width]],
            dtype=torch.float16,
            device=self.device,
        )
        time_ids = torch.cat([time_ids, time_ids], dim=0)
        added_cond_kwargs = {
            "text_embeds": pooled_text_embeds,
            "time_ids": time_ids,
        }

        sdxl.unet.set_default_attn_processor()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noisy_latents = torch.randn(
            (1, 4, height // 8, width // 8),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
        noisy_latents = noisy_latents.to(device=self.device, dtype=torch.float16)
        sdxl.scheduler.set_timesteps(steps, device=self.device)
        noisy_latents *= sdxl.scheduler.init_noise_sigma
        step = 1

        await asyncio.sleep(0)

        for t in tqdm(
            sdxl.scheduler.timesteps,
            desc="generating",
            total=steps,
            file=sys.stdout,
        ):
            latent_model_input = torch.cat([noisy_latents] * 2)
            latent_model_input = sdxl.scheduler.scale_model_input(latent_model_input, t)

            with torch.no_grad():
                output = sdxl.unet(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=crossattend_text_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                )

            noise_pred_uncond, noise_pred_text = output.sample.chunk(2)
            predicted_noise = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            try:
                noisy_latents = sdxl.scheduler.step(
                    predicted_noise, t, noisy_latents, generator=generator
                ).prev_sample
            except TypeError:
                noisy_latents = sdxl.scheduler.step(
                    predicted_noise, t, noisy_latents
                ).prev_sample

            match self.device:
                case "cuda":
                    torch.cuda.synchronize()
                case "mps":
                    torch.mps.synchronize()
                case "xpu":
                    torch.xpu.synchronize()

            await _report_step(on_step, step, steps)
            step += 1
            sys.stdout.flush()
            await asyncio.sleep(0)

        def vae_decode(vae, latents):
            latents = latents.to(device=vae.device, dtype=vae.dtype)

            with torch.no_grad():
                unscaled_latents = latents / vae.config.scaling_factor
                decoded_output = vae.decode(unscaled_latents).sample

            reconstructed_tensor = (decoded_output + 1.0) / 2.0
            reconstructed_tensor = reconstructed_tensor.clamp(0, 1)
            reconstructed_tensor = (
                reconstructed_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
            )
            reconstructed_array = (reconstructed_tensor[0] * 255).astype(np.uint8)

            return PILImg.fromarray(reconstructed_array)

        image = vae_decode(sdxl.vae, noisy_latents)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
