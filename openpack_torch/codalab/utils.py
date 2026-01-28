from logging import getLogger
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from .eval import eval_operation_segmentation

logger = getLogger(__name__)

import numpy as np
import pandas as pd
from typing import Dict, Tuple
# -----------------------------------------------------------------------
# 1. Select the final observation within every one-second window.
# -----------------------------------------------------------------------
def resample_to_1hz_last_point(
    ts: np.ndarray, 
    y: np.ndarray, 
    t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    seconds = ts // 1000 
    
    next_seconds = np.append(seconds[1:], seconds[-1] + 1)
    is_last_point_of_second = (seconds != next_seconds)
    
    y_1hz = y[is_last_point_of_second]
    t_1hz = t[is_last_point_of_second]
    
    return y_1hz, t_1hz

# -----------------------------------------------------------------------
def eval_operation_segmentation_wrapper(
    cfg: DictConfig,
    outputs: Dict[str, Dict[str, np.ndarray]],
    classes_tuple: Tuple[Tuple[int, str], ...],
) -> pd.DataFrame:
    
    """
    Converts 30Hz model outputs to 1Hz (taking the last point per second),
    calculates metrics per session (e.g., F1-score), and aggregates global 
    results across all sessions.
    """

    ignore_class_id = -1
    for i, (cls_id, cls_name) in enumerate(classes_tuple):
        if cls_name == "Null":
            ignore_class_id = i 
            break
            
    df_scores = []
    t_id_concat, y_id_concat = [], []


    for key, d in outputs.items():
        if d["y"].ndim == 3:
            y_pred_30hz = d["y"][0].argmax(axis=0)
        else:
            y_pred_30hz = d["y"].argmax(axis=0) if d["y"].ndim == 2 else d["y"]
            
        t_gt_30hz = d["t_idx"]  
        ts_30hz = d["unixtime"]  

        y_pred_1hz, t_gt_1hz = resample_to_1hz_last_point(ts_30hz, y_pred_30hz, t_gt_30hz)

        t_id_concat.append(t_gt_1hz)
        y_id_concat.append(y_pred_1hz)

        df_tmp = eval_operation_segmentation(
            t_gt_1hz,       
            y_pred_1hz,
            classes=classes_tuple,
            ignore_class_id=ignore_class_id,
        )
        df_tmp["key"] = key
        df_scores.append(df_tmp.reset_index(drop=False))

    df_tmp = eval_operation_segmentation(
        np.concatenate(t_id_concat, axis=0),
        np.concatenate(y_id_concat, axis=0),
        classes=classes_tuple,
        ignore_class_id=ignore_class_id,
    )
    df_tmp["key"] = "all"
    df_scores.append(df_tmp.reset_index(drop=False))

    df_scores = pd.concat(df_scores, axis=0, ignore_index=True)
    return df_scores