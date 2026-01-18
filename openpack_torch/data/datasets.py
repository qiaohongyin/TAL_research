from logging import getLogger
from typing import Any,Tuple
import numpy as np
import torch
from pathlib import Path
from omegaconf import DictConfig, open_dict


log = getLogger(__name__)

class OpenPackkd(torch.utils.data.Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        user_session_list: tuple[tuple[str, str], ...] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg


        self.imu_list = []
        self.skel_list = []
        self.dist_list = []
        self.t_list = []
        self.ts_list = []


        if user_session_list:
            print(f"📂 准备加载 {len(user_session_list)} 个 Session...")
            self.load_dataset(user_session_list)
            
            if len(self.imu_list) > 0:
                self.x_imu = np.concatenate(self.imu_list, axis=0)
                self.x_skel = np.concatenate(self.skel_list, axis=0)
                self.t = np.concatenate(self.t_list, axis=0)
                self.ts = np.concatenate(self.ts_list, axis=0)
                
                if len(self.dist_list) > 0 and self.dist_list[0] is not None:
                    self.x_dist = np.concatenate(self.dist_list, axis=0)
                else:
                    self.x_dist = None
                    
                print(f"✅ 数据集加载完成! 总样本数: {self.x_imu.shape[0]}")
            else:
                raise RuntimeError("未加载到任何数据，请检查 user_session_list 或文件路径。")
        else:
            print("⚠️ Warning: user_session_list 为空，未加载数据。")

    def load_dataset(self, user_session_list: tuple[tuple[str, str], ...]) -> None:
        for seq_idx, (user, session) in enumerate(user_session_list):
            with open_dict(self.cfg):
                self.cfg.user = {"name": user}
                self.cfg.session = session

            try:
                imu_arr, kpt_arr, label_arr, ts_arr, dist_arr = load_wrapper(self.cfg)
                
                self.imu_list.append(imu_arr)
                self.skel_list.append(kpt_arr)
                self.t_list.append(label_arr)
                self.ts_list.append(ts_arr)
                if dist_arr is not None:
                    self.dist_list.append(dist_arr)
                    
            except Exception as e:
                print(f"⚠️ 跳过 Session {user}-{session}: {e}")
                continue

    def __len__(self) -> int:
        return self.x_imu.shape[0]

    def __getitem__(self, idx: int) -> dict[str, Any]:

        imu_np = self.x_imu[idx]
        x_imu = torch.from_numpy(imu_np).float().unsqueeze(-1)

        # 2. Skeleton: (3, 450, 17) -> 直接转 float
        skel_np = self.x_skel[idx]
        x_skel = torch.from_numpy(skel_np).float()

        # 3. Label: (1, 450) -> (450,) 去除多余维度
        t = torch.from_numpy(self.t[idx]).long().squeeze(0)

        # 4. Timestamp: (1, 450) -> (450,)
        ts = torch.from_numpy(self.ts[idx]).long().squeeze(0)

        if self.x_dist is not None:
            x_dist = torch.from_numpy(self.x_dist[idx]).float().squeeze(0)
        else:
            x_dist = torch.zeros_like(ts).float()

        return {
            "skeleton": x_skel,          # (3, T, V)
            "imu_teacher": x_imu,        # (6, T, 1)
            "imu_student": x_imu[:3],    # (3, T, 1) 
            "t": t,                      # (T,)
            "ts": ts,                    # (T,)
            "wrist_dist": x_dist,        # (T,)
        }
    
# -----------------------------------------------------------------------------
def load_wrapper(cfg: DictConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = Path(
        cfg.dataset.stream_k4a.spec.kinect2dKpt.spec.path.dir,
        cfg.dataset.stream_k4a.spec.kinect2dKpt.spec.path.fname,
    )
    data = np.load(path)
    
    x_imu = data['x_imu']   # Shape: (N, 6, 450)
    x_sk  = data['x_skel']  # Shape: (N, 3, 450, 17)
    t     = data['t']       # Shape: (N, 1, 450)
    ts    = data['ts']      # Shape: (N, 1, 450)
    dist = data['x_dist'] # Shape: (N, 1, 450)


    return x_imu, x_sk, t, ts, dist