import platform

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from tqdm.std import tqdm

IS_APPLE_METAL = platform.machine().lower() == "arm64"


async def init_sdxl(app: FastAPI, model: str):
    app.state.sdxl = LoadedSDXL(app, model)


class LoadedSDXL:
    def __init__(self, app: FastAPI, model: str):
        self._sdxl = None
        if IS_APPLE_METAL:
            from imagen.metal.stable_diffusion import StableDiffusionXL

            self._sdxl = StableDiffusionXL(model=model, float16=True)
            self._sdxl.ensure_models_are_loaded()

    def infer_and_wait(
        self,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        steps: int,
        seed: int,
        width: int,
        height: int,
    ) -> bytes | None:
        if IS_APPLE_METAL:
            from imagen.metal.stable_diffusion import StableDiffusionXL
            import mlx.core as mx

            mx.set_default_stream(mx.default_stream(mx.default_device()))
            if not self._sdxl:
                raise ValueError()
            sdxl: StableDiffusionXL = self._sdxl
            latents = sdxl.generate_latents(
                prompt,
                negative_text=negative_prompt,
                n_images=1,
                cfg_weight=guidance_scale,
                num_steps=steps,
                seed=seed,
                latent_size=(width // 8, height // 8),
            )
            final_latent = None
            for latent_t in tqdm(latents, total=steps):
                mx.eval(latent_t)
                final_latent = latent_t
            decoded = sdxl.decode(final_latent)
            mx.eval(decoded)
            img_array = (decoded * 255).astype(mx.uint8)
            return bytes(img_array)

    async def infer(
        self,
        prompt: str,
        negative_prompt: str,
        guidance_scale: float,
        steps: int,
        seed: int,
        width: int,
        height: int,
    ):
        await run_in_threadpool(
            self.infer_and_wait,
            prompt,
            negative_prompt,
            guidance_scale,
            steps,
            seed,
            width,
            height,
        )
