from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import numpy as np
import torch
from metamotivo.fb_cpr.huggingface import FBcprModel


DEFAULT_S1_MODEL_ID = "facebook/metamotivo-S-1"
DEFAULT_S1_BUFFER = Path("/home/yhzhu/AI/humenv/metamotivo-S-1-datasets/data/buffer_inference_500000.hdf5")


@dataclass
class MetamotivoS1Policy:
    """Frozen MetaMotivo S-1 policy used as the human base controller."""

    task: str = "move-ego-0-2"
    model_id: str = DEFAULT_S1_MODEL_ID
    buffer_path: Path = DEFAULT_S1_BUFFER
    device: str = "cpu"
    num_samples_per_inference: int = 100_000
    max_workers: int = 12
    mean_action: bool = True
    latent_cache_dir: Path = Path(".cache/hit_exo_humenv/s1_latents")

    def __post_init__(self) -> None:
        self.buffer_path = Path(self.buffer_path)
        self.latent_cache_dir = Path(self.latent_cache_dir)
        self._model = FBcprModel.from_pretrained(self.model_id, device=self.device, local_files_only=True)
        self._z: torch.Tensor | None = None

    def z_for_task(self, task: str) -> torch.Tensor:
        cache_path = self._latent_cache_path(task)
        if cache_path.exists():
            return torch.load(cache_path, map_location=self.device)

        from metamotivo.buffers.buffers import DictBuffer
        from metamotivo.wrappers.humenvbench import RewardWrapper

        print(f"[INFO] Inferring S-1 latent for {task} using {self.num_samples_per_inference} samples...")
        with h5py.File(self.buffer_path, "r") as hf:
            data = {key: value[:] for key, value in hf.items()}
        buffer = DictBuffer(capacity=data["qpos"].shape[0], device="cpu")
        buffer.extend(data)
        reward_policy = RewardWrapper(
            model=self._model,
            inference_dataset=buffer,
            num_samples_per_inference=self.num_samples_per_inference,
            inference_function="reward_wr_inference",
            max_workers=self.max_workers,
        )
        z = reward_policy.reward_inference(task, num_envs=1, state_init="Default")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(z.detach().cpu(), cache_path)
        print(f"[INFO] Cached S-1 latent: {cache_path}")
        return z

    def z_for_tracking_motion(self, motion_file: str | Path, episode: str = "ep_0") -> torch.Tensor:
        motion_file = Path(motion_file)
        cache_path = self._tracking_latent_cache_path(motion_file, episode)
        if cache_path.exists():
            return torch.load(cache_path, map_location=self.device)

        from metamotivo.wrappers.humenvbench import TrackingWrapper

        with h5py.File(motion_file, "r") as hf:
            ep = hf[episode]
            if "observation" not in ep:
                raise KeyError(f"{motion_file}:{episode} is missing required 'observation' dataset")
            observation = np.asarray(ep["observation"][:], dtype=np.float32)
        if observation.shape[0] < 2:
            raise ValueError(f"{motion_file}:{episode} must contain at least two observation frames")

        tracker = TrackingWrapper(model=self._model, numpy_output=False)
        next_obs = torch.as_tensor(observation[1:], dtype=torch.float32, device=self.device)
        print(f"[INFO] Inferring S-1 tracking latents for {motion_file}:{episode}...")
        with torch.inference_mode():
            z = tracker.tracking_inference(next_obs=next_obs).detach().cpu()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(z, cache_path)
        print(f"[INFO] Cached S-1 tracking latents: {cache_path}")
        return z

    @torch.inference_mode()
    def act_tensor(self, proprio_obs: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        obs = proprio_obs.to(device=self.device, dtype=torch.float32)
        if z is None:
            if self._z is None:
                self._z = self.z_for_task(self.task)
            z = self._z.expand(obs.shape[0], -1)
        else:
            z = z.to(device=self.device, dtype=torch.float32)
        if self.mean_action:
            return self._act_mean(obs, z)
        return self._model.act(obs=obs, z=z, mean=False)

    @torch.inference_mode()
    def __call__(self, proprio_obs: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(
            np.asarray(proprio_obs, dtype=np.float32).reshape(1, -1),
            device=self.device,
        )
        action = self.act_tensor(obs)
        return np.asarray(action, dtype=np.float64).reshape(-1)

    def _latent_cache_path(self, task: str | None = None) -> Path:
        safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", task or self.task)
        return self.latent_cache_dir / f"{safe_task}_{self.num_samples_per_inference}.pt"

    def _tracking_latent_cache_path(self, motion_file: Path, episode: str) -> Path:
        safe_motion = re.sub(r"[^A-Za-z0-9_.-]+", "_", motion_file.stem)
        safe_episode = re.sub(r"[^A-Za-z0-9_.-]+", "_", episode)
        return self.latent_cache_dir / f"tracking_{safe_motion}_{safe_episode}.pt"

    def _act_mean(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Return the deterministic S-1 action without constructing a distribution."""
        norm_obs = self._model._normalize(obs)
        actor = self._model._actor
        z_embedding = actor.embed_z(torch.cat([norm_obs, z], dim=-1))
        s_embedding = actor.embed_s(norm_obs)
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        return torch.tanh(actor.policy(embedding))
