import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pprint import pprint


class PatternTransformer:
    """Learns a transformation from input_like -> output_like and applies it to new inputs.

    The transformer currently supports:
    - Per-value color remapping (0..9)
    - Rotations (0, 90, 180, 270 degrees)
    - Flips (horizontal, vertical)
    - Integer zooming (nearest-neighbor upscaling)
    - Integer shrinking (block-mode downscaling)
    Fallbacks keep the input unchanged when no reliable rule is learned.
    """

    def __init__(self):
        # Learned parameters
        self.color_map = None  # dict[int,int]
        self.rotation_k = 0  # number of 90deg rotations
        self.flip_h = False
        self.flip_v = False
        # Upscaling factors (>=1). When ==1 and pool factors >1, that means shrinking.
        self.row_repeat = 1
        self.col_repeat = 1
        # Downscaling factors (>=1). When >1 we will pool blocks with mode.
        self.row_pool = 1
        self.col_pool = 1
        # Translation placement offsets into target output canvas
        self.offset_row = 0
        self.offset_col = 0
        # Learned target output shape (from training output_like)
        self.target_h = None
        self.target_w = None
        self.learned_background = None  # optional background color inferred from output_like

    @staticmethod
    def _mode(values: np.ndarray) -> int:
        if values.size == 0:
            return 0
        vals, counts = np.unique(values, return_counts=True)
        return int(vals[np.argmax(counts)])

    def fit(self, input_like, output_like) -> None:
        input_like = np.array(input_like)
        output_like = np.array(output_like)

        # Default background as the most common color in output_like
        self.learned_background = self._mode(output_like)

        in_h, in_w = input_like.shape
        out_h, out_w = output_like.shape
        self.target_h = out_h
        self.target_w = out_w

        # Search over geometric transforms and integer scaling to best align to output_like
        rotations = [0, 1, 2, 3]  # multiples of 90 degrees
        flip_options = [(False, False), (True, False), (False, True), (True, True)]

        def apply_geom(arr: np.ndarray, rot_k: int, fh: bool, fv: bool) -> np.ndarray:
            x = np.rot90(arr, k=rot_k)
            if fh:
                x = np.fliplr(x)
            if fv:
                x = np.flipud(x)
            return x

        def upsample(arr: np.ndarray, rr: int, cc: int) -> np.ndarray:
            if rr == 1 and cc == 1:
                return arr
            # nearest-neighbor upscaling using np.kron
            return np.kron(arr, np.ones((rr, cc), dtype=arr.dtype))

        def block_mode_reduce(arr: np.ndarray, pr: int, pc: int) -> np.ndarray:
            if pr == 1 and pc == 1:
                return arr
            h, w = arr.shape
            assert h % pr == 0 and w % pc == 0
            new_h = h // pr
            new_w = w // pc
            # reshape into blocks and take mode per block
            reshaped = arr.reshape(new_h, pr, new_w, pc).swapaxes(1, 2).reshape(new_h * new_w, pr * pc)
            out_vals = []
            for i in range(reshaped.shape[0]):
                block = reshaped[i]
                vals, counts = np.unique(block, return_counts=True)
                out_vals.append(int(vals[np.argmax(counts)]))
            return np.array(out_vals, dtype=arr.dtype).reshape(new_h, new_w)

        def score_map(a: np.ndarray, b: np.ndarray) -> float:
            # percentage of equal cells
            if a.shape != b.shape:
                return -1.0
            return float(np.mean(a == b))

        best = {
            "score": -1.0,
            "rot": 0,
            "fh": False,
            "fv": False,
            "rr": 1,
            "cc": 1,
            "pr": 1,
            "pc": 1,
            "dy": 0,
            "dx": 0,
            "map": None,
        }

        for rot in rotations:
            for fh, fv in flip_options:
                geom = apply_geom(input_like, rot, fh, fv)
                gh, gw = geom.shape

                # Determine scaling to reach output shape: either up or down per axis
                # Rows
                rr, pr = 1, 1
                if out_h % gh == 0:
                    rr = out_h // gh
                elif gh % out_h == 0:
                    pr = gh // out_h
                else:
                    continue  # cannot match rows exactly

                # Cols
                cc, pc = 1, 1
                if out_w % gw == 0:
                    cc = out_w // gw
                elif gw % out_w == 0:
                    pc = gw // out_w
                else:
                    continue  # cannot match cols exactly

                # Apply scaling
                scaled = geom
                if rr > 1 or cc > 1:
                    scaled = upsample(scaled, rr, cc)
                if pr > 1 or pc > 1:
                    scaled = block_mode_reduce(scaled, pr, pc)

                # Placement search (translations, padding, cropping) into output_like.shape
                sh, sw = scaled.shape
                tgt_h, tgt_w = output_like.shape

                # Compute dy range
                if sh < tgt_h:
                    dy_range = range(0, tgt_h - sh + 1)
                elif sh > tgt_h:
                    dy_range = range(-(sh - tgt_h), 1)  # negative means cropping from top
                else:
                    k = min(3, max(0, tgt_h - 1))
                    dy_range = range(-k, k + 1)

                # Compute dx range
                if sw < tgt_w:
                    dx_range = range(0, tgt_w - sw + 1)
                elif sw > tgt_w:
                    dx_range = range(-(sw - tgt_w), 1)
                else:
                    k = min(3, max(0, tgt_w - 1))
                    dx_range = range(-k, k + 1)

                def place_with_offset(src: np.ndarray, dy: int, dx: int, target_shape, bg: int) -> np.ndarray:
                    th, tw = target_shape
                    canvas = np.full((th, tw), bg, dtype=src.dtype)
                    # Determine overlapping ranges
                    sy_start = max(0, -dy)
                    sx_start = max(0, -dx)
                    sy_end = min(src.shape[0], th - dy)
                    sx_end = min(src.shape[1], tw - dx)
                    if sy_start >= sy_end or sx_start >= sx_end:
                        return canvas
                    ty_start = max(0, dy)
                    tx_start = max(0, dx)
                    ty_end = ty_start + (sy_end - sy_start)
                    tx_end = tx_start + (sx_end - sx_start)
                    canvas[ty_start:ty_end, tx_start:tx_end] = src[sy_start:sy_end, sx_start:sx_end]
                    return canvas

                for dy in dy_range:
                    for dx in dx_range:
                        placed = place_with_offset(scaled, dy, dx, output_like.shape, self.learned_background)

                        # Learn color map from placed to output
                        cmap = {}
                        for v in range(10):
                            m = (placed == v)
                            if np.any(m):
                                cmap[v] = self._mode(output_like[m])
                        # Apply map to evaluate
                        tmp = placed.copy()
                        for v in range(10):
                            if v in cmap:
                                tmp[placed == v] = cmap[v]
                        s = score_map(tmp, output_like)
                        if s > best["score"]:
                            best.update({
                                "score": s,
                                "rot": rot,
                                "fh": fh,
                                "fv": fv,
                                "rr": rr,
                                "cc": cc,
                                "pr": pr,
                                "pc": pc,
                                "dy": dy,
                                "dx": dx,
                                "map": cmap,
                            })

        if best["score"] >= 0:
            self.rotation_k = best["rot"]
            self.flip_h = best["fh"]
            self.flip_v = best["fv"]
            self.row_repeat = best["rr"]
            self.col_repeat = best["cc"]
            self.row_pool = best["pr"]
            self.col_pool = best["pc"]
            self.offset_row = best["dy"]
            self.offset_col = best["dx"]
            self.color_map = best["map"]
            return

        # Fallbacks if nothing matched exactly
        # Try pure color map when shapes match
        if input_like.shape == output_like.shape:
            mapping = {}
            for v in range(10):
                mask = (input_like == v)
                if np.any(mask):
                    mapping[v] = self._mode(output_like[mask])
            self.color_map = mapping
            self.rotation_k = 0
            self.flip_h = False
            self.flip_v = False
            self.row_repeat = 1
            self.col_repeat = 1
            self.row_pool = 1
            self.col_pool = 1
            return

        # Last resort: overlapping region color map
        min_h = min(in_h, out_h)
        min_w = min(in_w, out_w)
        overlap_in = input_like[:min_h, :min_w]
        overlap_out = output_like[:min_h, :min_w]
        mapping = {}
        for v in range(10):
            mask = (overlap_in == v)
            if np.any(mask):
                mapping[v] = self._mode(overlap_out[mask])
        self.color_map = mapping if mapping else None
        self.rotation_k = 0
        self.flip_h = False
        self.flip_v = False
        self.row_repeat = 1
        self.col_repeat = 1
        self.row_pool = 1
        self.col_pool = 1

    def predict(self, input_matrix) -> np.ndarray:
        input_matrix = np.array(input_matrix)

        # 1) Geometric transform
        x = np.rot90(input_matrix, k=self.rotation_k)
        if self.flip_h:
            x = np.fliplr(x)
        if self.flip_v:
            x = np.flipud(x)

        # 2) Integer zoom (upsample)
        if self.row_repeat > 1 or self.col_repeat > 1:
            x = np.kron(x, np.ones((self.row_repeat, self.col_repeat), dtype=x.dtype))

        # 3) Integer shrink (block-mode reduce)
        if self.row_pool > 1 or self.col_pool > 1:
            h, w = x.shape
            # guard to avoid crash if not divisible; if not, crop from bottom/right minimally
            pr = self.row_pool
            pc = self.col_pool
            h_adj = (h // pr) * pr
            w_adj = (w // pc) * pc
            if h_adj != h or w_adj != w:
                x = x[:h_adj, :w_adj]
            new_h = x.shape[0] // pr
            new_w = x.shape[1] // pc
            reshaped = x.reshape(new_h, pr, new_w, pc).swapaxes(1, 2).reshape(new_h * new_w, pr * pc)
            out_vals = []
            for i in range(reshaped.shape[0]):
                block = reshaped[i]
                vals, counts = np.unique(block, return_counts=True)
                out_vals.append(int(vals[np.argmax(counts)]))
            x = np.array(out_vals, dtype=x.dtype).reshape(new_h, new_w)

        # 4) Placement into learned target canvas size using learned offsets
        if self.target_h is not None and self.target_w is not None:
            th, tw = self.target_h, self.target_w
            bg = self.learned_background if self.learned_background is not None else 0
            canvas = np.full((th, tw), bg, dtype=x.dtype)
            dy, dx = self.offset_row, self.offset_col
            sy_start = max(0, -dy)
            sx_start = max(0, -dx)
            sy_end = min(x.shape[0], th - dy)
            sx_end = min(x.shape[1], tw - dx)
            if sy_start < sy_end and sx_start < sx_end:
                ty_start = max(0, dy)
                tx_start = max(0, dx)
                ty_end = ty_start + (sy_end - sy_start)
                tx_end = tx_start + (sx_end - sx_start)
                canvas[ty_start:ty_end, tx_start:tx_end] = x[sy_start:sy_end, sx_start:sx_end]
            x = canvas

        # 5) Color mapping
        if self.color_map is not None:
            mapped = x.copy()
            for v in range(10):
                if v in self.color_map:
                    mapped[x == v] = self.color_map[v]
            x = mapped

        return x

def load_arc_data(json_path):
    """Load ARC AGI data from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    training_pairs = {}  # Dict of problem_id and corresponding (input_tensor, output_tensor)
    testing_inputs = {}  # Dict of problem_id and corresponding input_tensors

    first = True
    first_problem_id = None
    
    for problem_id, problem in data.items():
        # Train data
        for pair in problem['train']:
            if first:
                first_problem_id = problem_id
                # print("First Problem ID:", first_problem_id)
                # print("First Pair:")
                # pprint(pair)
            inp = pair['input']
            out = pair['output']
            if first:
                first = False
            training_pairs[problem_id] = (inp, out)
            if first:
                pprint(training_pairs)
        # Test data
        for test_case in problem['test']:
            inp = test_case['input']
            testing_inputs[problem_id] = inp

    return first_problem_id, training_pairs, testing_inputs

def build_output(input, input_like, output_like):
    """Deprecated: kept for compatibility; delegates to PatternTransformer."""
    transformer = PatternTransformer()
    transformer.fit(input_like, output_like)
    return transformer.predict(input)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading ARC AGI data...")
    first_problem_id, training_pairs, testing_inputs = load_arc_data('/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json')
    print(f"Loaded {len(training_pairs)} training pairs and {len(testing_inputs)} test inputs.")
    submission = {}
    counter = 0
    for key, val in training_pairs.items():
        print(key)
        test_input = testing_inputs[key]
        train_in, train_out = val

        # Learn transformation from training example
        transformer = PatternTransformer()
        transformer.fit(train_in, train_out)

        # Predict on the provided test input
        pred = transformer.predict(test_input)

        submission[key] = [{"attempt_1": pred.tolist() if isinstance(pred, np.ndarray) else []}]
        counter += 1
        print("#"*40)
        print("Success!")
        pprint(pred)
        print("#"*40)
    # Write the data to a file with pretty-printing
    with open("submission.json", "w") as submission_file:
        json.dump(submission, submission_file, indent=4)
    print(f"Finished - {counter}!")

if __name__ == "__main__":
    main()