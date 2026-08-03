import inspect
import copy
from dataclasses import dataclass
import json
import sys
import threading
import asyncio
import platform
from time import sleep

from fastapi import FastAPI
from sqlalchemy import select
from tqdm.std import tqdm
import numpy as np
import io
from PIL import Image as PILImg
from diffusers import EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection

from app.database import db_session
from app.database.minio import upload_image
from app.database.models import Image

IS_APPLE_METAL = platform.machine().lower() == "arm64"

if IS_APPLE_METAL:
    from imagen.metal.stable_diffusion import StableDiffusionXL  # type ignore
    import mlx.core as mx  # type ignore
else:
    import torch  # type: ignore
    from diffusers import StableDiffusionXLPipeline  # type: ignore
    from transformers.modeling_outputs import BaseModelOutputWithPooling  # type: ignore


def init_sdxl(app: FastAPI, model: str):
    app.state.sdxl = DedicatedSDXLWorker(app, model)
    app.state.sdxl.start()


@dataclass
class InferenceRequest:
    total_steps: int
    steps_completed: int


class DedicatedSDXLWorker:
    def __init__(self, app: FastAPI, model: str):
        self._queue: asyncio.Queue | None = None
        self._ack_queue: asyncio.Queue | None = None
        self._event_loop = asyncio.new_event_loop()
        self._worker_thread = threading.Thread(target=self._threading_loop, daemon=True)
        self._sdxl: LoadedSDXL | None = None
        self._app: FastAPI = app
        self._model: str = model
        self._semaphore: asyncio.Semaphore | None = None
        self._inference_requests: dict[str, InferenceRequest] | None = None
        self._inference_requests_lock: asyncio.Lock | None = None

    @property
    def inference_requests(self) -> dict[str, InferenceRequest] | None:
        if not self._inference_requests:
            return None
        copied_requests = self._inference_requests.copy()
        return copied_requests

    def _threading_loop(self):
        asyncio.set_event_loop(self._event_loop)
        self._event_loop.run_until_complete(self._async_loop())

    async def _set_inference_req(self, promise_id: str, req: InferenceRequest):
        async def _():
            if not self._inference_requests_lock:
                raise ValueError()
            if self._inference_requests is None:
                raise ValueError()
            async with self._inference_requests_lock:
                self._inference_requests[promise_id] = req

        if asyncio.get_event_loop() == self._event_loop:
            await _()
        else:
            tf = asyncio.run_coroutine_threadsafe(_(), self._event_loop)
            await asyncio.wrap_future(tf)

    async def _pop_inference_req(self, promise_id) -> InferenceRequest | None:
        async def _():
            if not self._inference_requests_lock:
                return None
            if self._inference_requests is None:
                return None
            async with self._inference_requests_lock:
                return self._inference_requests.pop(promise_id)

        tf = asyncio.run_coroutine_threadsafe(_(), self._event_loop)
        return await asyncio.wrap_future(tf)

    async def _imagen(self, item: dict) -> None:
        """for each imagen req"""
        if not self._sdxl:
            return
        # TODO: Pass enceyption info here in future, zero_knowledge is given as stringified json
        zk: dict | None = None
        if "zero_knowledge" in item.keys():
            zk: dict | None = json.loads(item.pop("zero_knowledge"))
        promise_id: str | None = item.get("promise_id")
        if not promise_id:
            return
        # generate image
        await self._set_inference_req(promise_id, InferenceRequest(item["steps"], 0))
        image_array = await self._sdxl.infer(self, **item)
        if not image_array:
            return None
        # TODO: encryption
        if zk:
            pass

        async def _db_query():
            await upload_image(self._app, io.BytesIO(image_array), promise_id)
            async with db_session() as session:
                stmt = select(Image).where(Image.promise_id == promise_id)
                results = await session.scalars(stmt)
                img = results.first()
                if not img:
                    return
                result_steps = await self._pop_inference_req(promise_id)
                if not result_steps:
                    return
                img.steps_status = (
                    f"{result_steps.steps_completed}/{result_steps.total_steps}"
                )
                await session.commit()

        sb_schedule = asyncio.run_coroutine_threadsafe(
            _db_query(),
            self._app.state.event_loop,
        )
        awaitable_db_task = asyncio.wrap_future(sb_schedule)
        await awaitable_db_task

    async def _async_loop(self):
        self._sdxl = LoadedSDXL(self._app, self._model)
        self._queue = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(4)
        self._inference_requests = {}
        self._inference_requests_lock = asyncio.Lock()
        while True:
            item = await self._queue.get()
            if not item:
                break
            asyncio.create_task(self._imagen(item))

    def queue_inference(self, promise_id: str, inference_params: dict):
        async def _queue():
            if not self._sdxl or not self._queue:
                return None
            inference_params["promise_id"] = promise_id
            await self._queue.put(inference_params)

        asyncio.run_coroutine_threadsafe(_queue(), self._event_loop)

    def get_inference_status(self, promise_id: str) -> str | None:
        """
        Will always return in format: "<current_step>/<scheduled_steps>"
        Otherwise None if no inf found
        """
        if not self.inference_requests:
            return None
        inf = self.inference_requests.get(promise_id)
        return None if not inf else f"{inf.steps_completed}/{inf.total_steps}"

    def cancel_inference():
        pass

    def start(self):
        self._worker_thread.start()

    def interrupt(self):
        """
        Terminates this worker thread and wait
        """
        pass


# pyright: reportUnboundVariable=false
class LoadedSDXL:
    def __init__(self, app: FastAPI, model: str):
        self._sdxl = None
        if IS_APPLE_METAL:
            mx.set_cache_limit(1 * 1024 * 1024 * 1024)
            self._sdxl = StableDiffusionXL(model=model, float16=True)
            self._sdxl.ensure_models_are_loaded()
        else:
            import torch  # type: ignore
            from diffusers import StableDiffusionXLPipeline  # type: ignore

            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "xpu"
                if torch.xpu.is_available()
                else "cpu"
            )

            revision = "fix/corrupted-text-encoder"
            text_encoder = CLIPTextModel.from_pretrained(
                model, subfolder="text_encoder", dtype=torch.float32, revision=revision
            ).to(self.device)
            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                model, subfolder="text_encoder_2", dtype=torch.float32
            ).to(self.device)
            self._sdxl = StableDiffusionXLPipeline.from_pretrained(
                model,
                torch_dtype=torch.float16,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
            ).to(self.device)
            self._diffusers_lock: asyncio.Lock = asyncio.Lock()

    async def infer(
        self,
        worker: DedicatedSDXLWorker,
        promise_id: str,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        steps: int,
        seed: int,
        width: int,
        height: int,
        clip_skip: int,
    ):
        if IS_APPLE_METAL:
            if not self._sdxl:
                raise ValueError()
            sdxl: StableDiffusionXL = self._sdxl  # type: ignore
            latents = sdxl.generate_latents(
                prompt,
                negative_text=negative_prompt,
                n_images=1,
                cfg_weight=guidance_scale,
                num_steps=steps,
                seed=seed,
                latent_size=(height // 8, width // 8),
            )
            final_latent = None
            i = 1
            for latent_t in tqdm(latents, "iteration", total=steps):
                mx.eval(latent_t)
                final_latent = latent_t
                if worker.inference_requests:
                    inf = worker.inference_requests.get(promise_id)
                    if inf:
                        inf.steps_completed = i
                        await worker._set_inference_req(promise_id, inf)
                i += 1
                await asyncio.sleep(0)
            decoded = sdxl.decode(final_latent)
            mx.eval(decoded)
            await asyncio.sleep(0)
            img_array = (decoded * 255).astype(mx.uint8)

            np_array = np.array(img_array)
            pil_img = PILImg.fromarray(np_array[-1])
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")  # Or "JPEG"
            encoded_bytes = buffer.getvalue()
            return encoded_bytes
        else:
            if not self._sdxl:
                raise ValueError()

            def _tokenize_text(text: str, tokenizer):
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

            def _encode_text(input_ids, encoder):
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

            local_scheduler = self._sdxl.scheduler.from_config(
                self._sdxl.scheduler.config
            )
            sdxl: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pipe(
                self._sdxl, scheduler=local_scheduler, torch_dtype=torch.float16
            )
            sdxl.scheduler = local_scheduler

            # Text Tokenizing
            tokenized_prompt = _tokenize_text(prompt, sdxl.tokenizer)
            tokenized_prompt_2 = _tokenize_text(prompt, sdxl.tokenizer_2)
            tokenized_neg_prompt = _tokenize_text(negative_prompt, sdxl.tokenizer)
            tokenized_neg_prompt_2 = _tokenize_text(negative_prompt, sdxl.tokenizer_2)

            # padding
            padding_size = abs(
                tokenized_prompt.shape[0] - tokenized_neg_prompt.shape[0]
            )
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
                should_neg_pad = (
                    tokenized_prompt.shape[0] > tokenized_neg_prompt.shape[0]
                )
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

            prompt_embeds, _ = _encode_text(tokenized_prompt, sdxl.text_encoder)
            prompt_embeds_2, cond_pooled_text_embeds = _encode_text(
                tokenized_prompt_2, sdxl.text_encoder_2
            )
            cond_prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)

            neg_prompt_embeds, _ = _encode_text(
                tokenized_neg_prompt,
                sdxl.text_encoder,
            )
            neg_prompt_embeds_2, uncond_pooled_text_embeds = _encode_text(
                tokenized_neg_prompt_2, sdxl.text_encoder_2
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
                [[1024, 1024, 0, 0, 1024, 1024]],
                dtype=torch.float16,
                device=self.device,
            )
            time_ids = torch.cat([time_ids, time_ids], dim=0)
            added_cond_kwargs = {
                "text_embeds": pooled_text_embeds,
                "time_ids": time_ids,
            }

            # initialize random latent
            generator = torch.Generator(device=self.device).manual_seed(seed)
            noisy_latents = torch.randn(
                (1, 4, height // 8, width // 8),
                generator=generator,
                device=self.device,
                dtype=torch.float16,
            )
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
                latent_model_input = sdxl.scheduler.scale_model_input(
                    latent_model_input, t
                )

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

                if worker.inference_requests:
                    inf = worker.inference_requests.get(promise_id)
                    if inf:
                        inf.steps_completed = step
                        await worker._set_inference_req(promise_id, inf)
                step += 1
                await asyncio.sleep(0)
                sys.stdout.flush()

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

            sdxl.vae = sdxl.vae.to(dtype=torch.float32)
            image = vae_decode(sdxl.vae, noisy_latents)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")

            return buffer.getvalue()
