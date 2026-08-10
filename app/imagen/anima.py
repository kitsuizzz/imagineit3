import asyncio
import io

from fastapi import FastAPI

from app.imagen.sdxl import DedicatedSDXLWorker, LoadedSDXL, IS_APPLE_METAL

if not IS_APPLE_METAL:
    import torch
    from diffusers import DiffusionPipeline


class DedicatedAnimaWorker(DedicatedSDXLWorker):
    def __init__(self, app: FastAPI):
        super().__init__(app, "")
        self._sdxl = LoadedNovaAnimeAM(app)


class LoadedNovaAnimeAM(LoadedSDXL):
    def __init__(self, app: FastAPI):
        if IS_APPLE_METAL:
            raise NotImplementedError()
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "xpu"
            if torch.xpu.is_available()
            else "cpu"
        )

        repo = "wssdaas/nova-anime-am-v4.0"
        self._sdxl = DiffusionPipeline.from_pretrained(
            repo, trust_remote_code=True, custom_pipeline=repo, dtype=torch.bfloat16
        ).to(self.device)

    async def _update_inf_step(
        worker: DedicatedAnimaWorker, promise_id: str, step: int
    ):
        if worker.inference_requests:
            inf = worker.inference_requests.get(promise_id)
            if inf:
                inf.steps_completed = step
                await worker._set_inference_req(promise_id, inf)

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
    ):
        if not self._sdxl:
            raise RuntimeError()

        def _callback(a, b, c, d):
            match self.device:
                case "cuda":
                    torch.cuda.synchronize()
                case "mps":
                    torch.mps.synchronize()
                case "xpu":
                    torch.xpu.synchronize()
            t_future = asyncio.run_coroutine_threadsafe(
                _update_inf_step(worker, promise_id, b), app.state.event_loop
            )
            t_future.result()
            return d

        image = self._sdxl(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=torch.generator(device="cpu").manual_seed(seed),
            width=width,
            height=height,
            callback_on_step_end=_callback,
        )

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")

        return buffer.getvalue()
