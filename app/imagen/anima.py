import asyncio
import io
import gc

from fastapi import FastAPI
from tqdm import tqdm
from PIL import Image

from app.imagen.sdxl import DedicatedSDXLWorker, LoadedSDXL, IS_APPLE_METAL

if not IS_APPLE_METAL:
    import torch
    from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler


class DedicatedAnimaWorker(DedicatedSDXLWorker):
    def __init__(self, app: FastAPI):
        super().__init__(app, "")
        self._sdxl = LoadedNovaAnimeAM(app)

    async def _imagen(self, item: dict) -> None:
        await super()._imagen(item)
        self._sdxl.clear_cache()


class LoadedNovaAnimeAM(LoadedSDXL):
    def __init__(self, app: FastAPI):
        if IS_APPLE_METAL:
            raise NotImplementedError()

        self._app = app
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "xpu"
            if torch.xpu.is_available()
            else "cpu"
        )

        repo = "wssdaas/nova-anime-am-v4.0"
        self._sdxl = DiffusionPipeline.from_pretrained(
            repo,
            trust_remote_code=True,
            custom_pipeline=repo,
            torch_dtype=torch.bfloat16,
        ).to(self.device)

        self._diffusers_lock: asyncio.Lock = asyncio.Lock()

    async def _update_inf_step(
        self, worker: DedicatedAnimaWorker, promise_id: str, step: int
    ):
        if worker.inference_requests:
            inf = worker.inference_requests.get(promise_id)
            if inf:
                inf.steps_completed = step
                await worker._set_inference_req(promise_id, inf)

    def clear_cache(self):
        gc.collect()
        match self.device:
            case "xpu":
                torch.xpu.empty_cache()

    async def infer(
        self,
        worker: DedicatedAnimaWorker,
        promise_id: str,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        steps: int,
        seed: int,
        width: int,
        height: int,
        clip_skip: int,
        scheduler,
    ):
        if not self._sdxl:
            raise RuntimeError()
        sdxl = self._sdxl
        device = self.device
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            sdxl.scheduler.config,
        )

        # embed.shape = [batch, token, 1024]
        inputs = sdxl.tokenizer(
            prompt, return_tensors="pt", padding="max_length", max_length=512
        ).to(device)
        async with self._diffusers_lock:
            with torch.no_grad():
                cond_embeds = sdxl.text_encoder(inputs.input_ids)[0]
        cond_attn_mask = inputs.attention_mask.clone()

        inputs = sdxl.tokenizer(
            negative_prompt, return_tensors="pt", padding="max_length", max_length=512
        ).to(device)
        async with self._diffusers_lock:
            with torch.no_grad():
                uncond_embeds = sdxl.text_encoder(inputs.input_ids)[0]
        uncond_attn_mask = inputs.attention_mask.clone()

        # t5_token input_ids: [batch, token]
        inputs = sdxl.t5_tokenizer(
            prompt, return_tensors="pt", padding="max_length", max_length=512
        ).to(device)
        cond_t5_ids = inputs.input_ids
        cond_t5_attn_mask = inputs.attention_mask

        inputs = sdxl.t5_tokenizer(
            negative_prompt, return_tensors="pt", padding="max_length", max_length=512
        ).to(device)
        uncond_t5_ids = inputs.input_ids
        uncond_t5_attn_mask = inputs.attention_mask

        # prompt_embeds.shape = [batch, token, 2048]
        async with self._diffusers_lock:
            with torch.no_grad():
                cond_embeds = sdxl.text_conditioner(
                    source_hidden_states=cond_embeds,
                    target_input_ids=cond_t5_ids,
                    source_attention_mask=cond_attn_mask,
                    target_attention_mask=cond_t5_attn_mask,
                )
                uncond_embeds = sdxl.text_conditioner(
                    source_hidden_states=uncond_embeds,
                    target_input_ids=uncond_t5_ids,
                    source_attention_mask=uncond_attn_mask,
                    target_attention_mask=uncond_t5_attn_mask,
                )

        await asyncio.sleep(0)
        prompt_embeds = torch.cat((uncond_embeds, cond_embeds), dim=0)

        del (
            inputs,
            cond_embeds,
            uncond_embeds,
            cond_attn_mask,
            uncond_attn_mask,
            cond_t5_ids,
            cond_t5_attn_mask,
            uncond_t5_ids,
            uncond_t5_attn_mask,
        )
        self.clear_cache()

        # noisy_latents.shape = [batch, channel, frame, height, width]
        latent_height, latent_width = height // 8, width // 8
        in_channels = sdxl.transformer.config.in_channels
        dtype_dit = sdxl.transformer.dtype
        noisy_latents = torch.randn(
            (1, in_channels, 1, latent_height, latent_width),
            dtype=dtype_dit,
            device=device,
        )
        padding_mask = torch.zeros(
            (1, 1, latent_height, latent_width),
            device=noisy_latents.device,
            dtype=noisy_latents.dtype,
        )

        # scheduler
        scheduler.set_timesteps(steps, device=device)
        scheduler.set_begin_index(0)

        # denoise
        for i, t in enumerate(
            tqdm(scheduler.timesteps, total=steps, desc="Denoising"), start=1
        ):
            if guidance_scale == 1.0:
                input_latents = noisy_latents
                encoder_hidden_states = cond_embeds
            else:
                input_latents = torch.cat([noisy_latents] * 2, dim=0)
                encoder_hidden_states = prompt_embeds
            timestep = (
                t.expand(input_latents.shape[0]) / scheduler.config.num_train_timesteps
            )
            async with self._diffusers_lock:
                with torch.no_grad():
                    outputs = sdxl.transformer(
                        hidden_states=input_latents,
                        encoder_hidden_states=encoder_hidden_states,
                        timestep=timestep,
                        padding_mask=padding_mask,
                    ).sample
            await asyncio.sleep(0)

            if guidance_scale == 1.0:
                predicted_noise = outputs
            else:
                uncond_noise, cond_noise = outputs.chunk(2)
                predicted_noise = uncond_noise + guidance_scale * (
                    cond_noise - uncond_noise
                )

            noisy_latents = scheduler.step(
                predicted_noise, t, noisy_latents
            ).prev_sample

            await self._update_inf_step(worker, promise_id, i)
            if self.device == "xpu":
                torch.xpu.synchronize()

            # if callback_on_step_end is not None:
            #     callback_kwargs = {}
            #     if "latents" in callback_on_step_end_tensor_inputs:
            #         callback_kwargs["latents"] = noisy_latents
            #
            #     callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
            #
            #     if "latents" in callback_outputs:
            #         noisy_latents = callback_outputs["latents"]

        latents_device = noisy_latents.device
        latents_dtype = noisy_latents.dtype
        latents_mean = (
            torch.tensor(sdxl.vae.config.latents_mean)
            .view(1, sdxl.vae.config.z_dim, 1, 1, 1)
            .to(latents_device, latents_dtype)
        )
        latents_std = (
            torch.tensor(sdxl.vae.config.latents_std)
            .view(1, sdxl.vae.config.z_dim, 1, 1, 1)
            .to(latents_device, latents_dtype)
        )
        latents_std = 1.0 / latents_std

        final_latents = noisy_latents / latents_std + latents_mean
        final_latents = final_latents.to(dtype=sdxl.vae.dtype)
        async with self._diffusers_lock:
            with torch.no_grad():
                images = sdxl.vae.decode(final_latents, return_dict=False)[0][:, :, 0]
        await asyncio.sleep(0)

        images = (images / 2.0 + 0.5).clamp(0.0, 1.0)
        images = images.permute(0, 2, 3, 1)

        images = (images * 255).round().to("cpu", torch.uint8)
        image = images[0].numpy()
        image = Image.fromarray(image)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        return buffer.getvalue()
