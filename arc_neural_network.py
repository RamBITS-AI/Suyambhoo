import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt
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
        # Super Kron Broadcast
        self.skb = False
        # Fill Quadrilaterals
        self.have_to_fill_quadrilaterals = False
        # Tile and Rotate
        self.t_n_r = False
        self.tile_by = ()  # (3, 3)
        self.rows_to_roll = []  # [2, 3]
        # Fill Between Diagonals
        self.has_diagonal_fillings = False
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

    def apply_geom(self, arr: np.ndarray, rot_k: int, fh: bool, fv: bool) -> np.ndarray:
        x = np.rot90(arr, k=rot_k)
        if fh:
            x = np.fliplr(x)
        if fv:
            x = np.flipud(x)
        return x

    def super_kron_broadcast(self, arr: np.ndarray) -> np.ndarray:
        b = arr > 0
        if len(arr[b]) == 0:
            return arr
        b = arr[b][0]
        b = arr == b
        b_arr = arr.copy()
        b_arr[b] = 1
        arr = np.kron(arr, b_arr)
        return arr

    def tile_and_roll(self, arr: np.ndarray, shift: int = 1, axis: int = 1) -> np.ndarray:
        arr = np.tile(arr, self.tile_by)
        arr[self.rows_to_roll] = np.roll(arr[self.rows_to_roll], shift=shift, axis=axis)
        return arr

    def fill_between_diagonals(self, arr: np.ndarray, boundary_element: int = 3, fill_element: int = 4):
        """
        Fills positions 'in between' diagonals (or straight lines) formed by 3's with 4's.
        Uses local pattern matching for cross, and diagonal L-bends.
        """
        rows, cols = arr.shape
        output_arr = arr.copy()
        
        for r in range(rows):
            for c in range(cols):
                if output_arr[r, c] != 0:
                    continue
                # Get neighbors (True if 3)
                up = self.get_neighbor(output_arr, r, c, -1, 0, boundary_element)
                down = self.get_neighbor(output_arr, r, c, 1, 0, boundary_element)
                left = self.get_neighbor(output_arr, r, c, 0, -1, boundary_element)
                right = self.get_neighbor(output_arr, r, c, 0, 1, boundary_element)
                up_right = self.get_neighbor(output_arr, r, c, -1, 1, boundary_element)
                down_left = self.get_neighbor(output_arr, r, c, 1, -1, boundary_element)
                up_left = self.get_neighbor(output_arr, r, c, -1, -1, boundary_element)
                down_right = self.get_neighbor(output_arr, r, c, 1, 1, boundary_element)
                
                filled = False
                
                # Vertical cross (fully bounded orthogonally)
                if up and down and left and right:
                    filled = True
                
                # Diagonal L-bend type 1: bottom-left turn (\ direction)
                elif left and down and up_right and not up and not right:
                    filled = True
                
                # Diagonal L-bend type 2: top-right turn (/ direction)
                elif up and right and down_left and not down and not left:
                    filled = True
                
                # Mirror for bottom-right turn (for \ extensions)
                elif right and down and up_left and not up and not left:
                    filled = True
                
                # Mirror for top-left turn (for / extensions)
                elif up and left and down_right and not down and not right:
                    filled = True
                
                if filled:
                    output_arr[r, c] = fill_element
        
        return output_arr
    
    def get_neighbor(self, arr, r, c, dr, dc, boundary_element: int = 3):
        nr, nc = r + dr, c + dc
        if nr < arr.shape[0] and nc < arr.shape[1]:
            # print(r, c, nr, nc, arr[nr, nc], boundary_element, arr[nr, nc] == boundary_element)
            return arr[nr, nc] == boundary_element
        return False
    
    def has_fillings_between_diagonals(self, arr: np.ndarray, boundary_element: int = 3):
        # Apply the filling
        filled_arr = self.fill_between_diagonals(arr, boundary_element=3, fill_element=11)
        return filled_arr, not np.array_equal(arr, filled_arr)
    
    def find_quadrilaterals(self, arr: np.ndarray, boundary_element: int = 2, valid_sum_of_inner_elements: int = 2):
        """
        Identifies axis-aligned rectangular quadrilaterals where the perimeter is entirely 3's.
        
        Returns a list of dictionaries, each containing:
        - 'rows': tuple (start_row, end_row)
        - 'cols': tuple (start_col, end_col)
        - 'subarray': the subarray of the quadrilateral
        """
        rows, cols = arr.shape
        quadrilaterals = []
        
        for row_start in range(rows):
            for row_end in range(row_start + 1, rows):  # Ensure height >= 2
                for col_start in range(cols):
                    for col_end in range(col_start + 1, cols):  # Ensure width >= 2
                        # Check top row
                        if not np.all(arr[row_start, col_start:col_end+1] == boundary_element):
                            continue
                        # Check bottom row
                        if not np.all(arr[row_end, col_start:col_end+1] == boundary_element):
                            continue
                        # Check left column
                        if not np.all(arr[row_start:row_end+1, col_start] == boundary_element):
                            continue
                        # Check right column
                        if not np.all(arr[row_start:row_end+1, col_end] == boundary_element):
                            continue
                        # if inner values are non-zero by any chance (like in the case of a all 9 elements are 3's)
                        if np.isnan(valid_sum_of_inner_elements) and np.sum(arr[row_start+1:row_end, col_start+1:col_end]) != valid_sum_of_inner_elements:
                            continue
                        if row_end - row_start < 2 or col_end - col_start < 2:
                            continue
                        # Valid quadrilateral
                        quad = {
                            'rows': (row_start, row_end),
                            'cols': (col_start, col_end),
                            'subarray': arr[row_start:row_end+1, col_start:col_end+1],
                            'contentarray': arr[row_start+1:row_end, col_start+1:col_end]
                        }
                        quadrilaterals.append(quad)
        
        return quadrilaterals

    def find_quadrilaterals_with_missing_corners(self, arr: np.ndarray, boundary_element: int = 3, valid_sum_of_inner_elements = np.nan):
        """
        Identifies axis-aligned rectangular quadrilaterals (height/width >=3) where the perimeter is entirely 3's,
        allowing for missing 3's at the four corners.
        
        Returns a list of dictionaries, each containing:
        - 'rows': tuple (start_row, end_row)
        - 'cols': tuple (start_col, end_col)
        - 'subarray': the subarray of the quadrilateral
        """
        rows, cols = arr.shape
        quadrilaterals = []
        
        for row_start in range(rows):
            for row_end in range(row_start + 2, rows):  # Ensure height >= 3
                for col_start in range(cols):
                    for col_end in range(col_start + 2, cols):  # Ensure width >= 3
                        # Check top row, excluding corners
                        if not np.all(arr[row_start, col_start+1:col_end] == boundary_element):
                            continue
                        # Check bottom row, excluding corners
                        if not np.all(arr[row_end, col_start+1:col_end] == boundary_element):
                            continue
                        # Check left column, excluding corners
                        if not np.all(arr[row_start+1:row_end, col_start] == boundary_element):
                            continue
                        # Check right column, excluding corners
                        if not np.all(arr[row_start+1:row_end, col_end] == boundary_element):
                            continue
                        # if inner values are 3 by any chance (like in the case of a plus-like formation)
                        if np.sum(arr[row_start+1:row_end, col_start+1:col_end]) != valid_sum_of_inner_elements:
                            continue
                        if row_end - row_start < 2 or col_end - col_start < 2:
                            continue
                        # Valid quadrilateral
                        quad = {
                            'rows': (row_start, row_end),
                            'cols': (col_start, col_end),
                            'subarray': arr[row_start:row_end+1, col_start:col_end+1],
                            'contentarray': arr[row_start+1:row_end, col_start+1:col_end]
                        }
                        quadrilaterals.append(quad)
        
        return quadrilaterals

    def fit(self, input_like, output_like) -> None:
        try:
            self.input_like = np.array(input_like)
            self.output_like = np.array(output_like)
    
            # Default background as the most common color in self.output_like
            self.learned_background = self._mode(self.output_like)
    
            in_h, in_w = self.input_like.shape
            out_h, out_w = self.output_like.shape
            self.target_h = out_h
            self.target_w = out_w
    
            # Search over geometric transforms and integer scaling to best align to self.output_like
            rotations = [0, 1, 2, 3]  # multiples of 90 degrees
            flip_options = [(False, False), (True, False), (False, True), (True, True)]
    
            # for rot in rotations:
            #     for fh, fv in flip_options:
            #         geom = apply_geom(self.input_like, rot, fh, fv)
            #         gh, gw = geom.shape
    
            self.skb = np.array_equal(self.output_like, self.super_kron_broadcast(self.input_like))
    
            if not self.skb:
                self.have_to_fill_quadrilaterals = len(self.find_quadrilaterals(self.input_like, boundary_element=2, valid_sum_of_inner_elements=2)) > 0
                if not self.have_to_fill_quadrilaterals:
                    self.tile_by = (3, 3)
                    self.rows_to_roll = [2, 3]
                    self.t_n_r = np.array_equal(self.output_like, self.tile_and_roll(self.input_like))
                    if not self.t_n_r:
                        self.tile_by = ()
                        self.rows_to_roll = []
                        # Apply the filling
                        _, has_diagonal_fillings = self.has_fillings_between_diagonals(self.input_like, boundary_element=3)
                        self.has_diagonal_fillings = has_diagonal_fillings
        except Exception as e:
            print("Exception... - ", e)

    def predict(self, input_matrix) -> np.ndarray:
        output_like = self.output_like
        pred = np.zeros_like(output_like)
        try:
            input = np.array(input_matrix)
            input_like = self.input_like

            b = output_like == input_like[0][0]

            # 1) Super Kron Broadcast
            if self.skb:
                return self.super_kron_broadcast(input.copy())
            # 2) Fill Quadrilaterals
            elif self.have_to_fill_quadrilaterals:
                pred = input.copy()
                boundary_element = 2
                center_element = 2
                quads = self.find_quadrilaterals(pred, boundary_element=boundary_element, valid_sum_of_inner_elements=center_element)
                second_quads = self.find_quadrilaterals(output_like.copy(), boundary_element=boundary_element)
                if len(second_quads) > 0 and len(quads) > 0:
                    fill_element = second_quads[0]['contentarray'][0][0]
                    for quad in quads:
                        row_start, row_end = quad['rows']
                        col_start, col_end = quad['cols']
                        pred[row_start+1:row_end, col_start+1:col_end] = fill_element  # 8
                        if int((row_start + row_end) / 2) * 2 == row_start + row_end:
                            if int((col_start + col_end) / 2) * 2 == col_start + col_end:
                                pred[int((row_start + row_end) / 2), int((col_start + col_end) / 2)] = center_element
                return pred
            # 3) Tile And Roll
            elif self.t_n_r:
                return self.tile_and_roll(input.copy())
            # 4) Diagonal Fillings
            elif self.has_diagonal_fillings:
                pred = input.copy()
                pred = self.fill_between_diagonals(pred, boundary_element=3, fill_element=4)
                return pred
            elif input.shape == output_like.shape:
                for i in range(1, 10):
                    ci = input_like == i
                    if np.any(ci):
                        cix = ci.copy()
                        ccix = np.nan
                        ci = i
                        for j in range(1, 10):
                            if j == i:
                                continue
                            cci = input_like == j
                            if np.any(cci):
                                ccix = cci.copy()
                                cci = j
                                break
                        try:
                            do = output_like[cix]
                            do = do[0]
                            ddx = np.nan
                            if isinstance(ccix, np.ndarray):
                                ddo = output_like[ccix]
                                ddo = ddo[0]
                            else:
                                for j in range(1, 10):
                                    if j == ci:
                                        continue
                                    ddo = output_like == j
                                    if np.any(ddo):
                                        ddx = ddo.copy()
                                        ddo = j
                                        break
                            p = input == ci
                            if not isinstance(cci, np.ndarray):
                                pp = input == cci
                            pred = input.copy()
                            pred[p] = do
                            if not isinstance(cci, np.ndarray):
                                pred[pp] = ddo
                            elif isinstance(ddx, np.ndarray):
                                pred[ddx] = ddo
                        except Exception as e:
                            print("Exception...", e)
                        return pred
            elif input.shape[1] == output_like.shape[1]:
                for i in range(1, 10):
                    ci = input_like == i
                    if np.any(ci):
                        ci = np.vstack([ci, ci[:output_like.shape[0]-input.shape[0],:]])
                        if output_like.shape != ci.shape:
                            break
                        do = output_like[ci]
                        do = do[0]
                        for j in range(1, 10):
                            p = input == j
                            if output_like.shape[0] > input.shape[0]:
                                p = np.vstack([p, p[:output_like.shape[0]-input.shape[0],:]])
                            elif input.shape[0] > output_like.shape[0]:
                                p = p[:input.shape[0]-output_like.shape[0]]
                            if np.any(p):
                                pred[p] = do
                                break
                return pred
            elif input.shape[0] > output_like.shape[0] and input.shape[1] > output_like.shape[1]:
                pred = input.copy()
                # Find the quadrilaterals
                quads = self.find_quadrilaterals_with_missing_corners(pred, 3)
                for quad in quads:
                    row_start, row_end = quad['rows']
                    col_start, col_end = quad['cols']
                    pred[row_start+1:row_end, col_start+1:col_end] = 4
                return pred
            else:
                return np.zeros([self.target_h, self.target_w])
        except Exception as e:
            print("Exception! - ", e)
        return pred

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
        testing_inputs[problem_id] = []
        for test_case in problem['test']:
            inp = np.array(test_case['input'])
            testing_inputs[problem_id].append(inp)

        if len(testing_inputs[problem_id]) == 1: testing_inputs[problem_id] = testing_inputs[problem_id][0]

    return first_problem_id, training_pairs, testing_inputs

def build_output(input, input_like, output_like):
    """Deprecated: kept for compatibility; delegates to PatternTransformer."""
    transformer = PatternTransformer()
    transformer.fit(input_like, output_like)
    return transformer.predict(input)

def _prepare_submission(t_input, key, val, submission, solutions, index=-1):
    test_input = t_input
    train_in, train_out = val

    # if key != '00dbd492':
    #     continue

    # pprint(test_input)
    # pprint(train_in)
    # pprint(train_out)
    
    # Learn transformation from training example
    transformer = PatternTransformer()

    transformer.fit(train_in, train_out)

    # Predict on the provided test input
    pred = transformer.predict(test_input)

    test_input = np.array(test_input)

    train_in = np.array(train_in)

    train_out = np.array(train_out)
    
    pred = pred.tolist() if isinstance(pred, np.ndarray) else np.zeros_like(test_input)

    pred_2 = np.zeros_like(test_input) if test_input.shape == train_in.shape and train_in.shape == train_out.shape else np.zeros_like(train_out) if test_input.shape != train_in.shape and train_in.shape == train_out.shape else np.zeros_like(test_input)

    pred_2 = pred_2.tolist()
    
    submission[key].append({"attempt_1": pred, 'attempt_2': pred_2})

    try:
        if (np.array_equal(np.array(solutions[key][index]), pred)):
            print("#"*80)
            print("Success!")
            pprint(pred)
            print("#"*80)
    except Exception as e:
        print("Index:", index)
        print("EXX:", e)
        print("T-Input:", t_input)
        print("Solutions:")
        for i, solution in enumerate(solutions[key]):
            print(f"solution[{key}][{i}]:", solution)
        print("Preds:", pred)
    # if counter < 7 and counter > 8:
    #     continue

    # # plt.pcolor(test_input)
    # plt.imshow(test_input, cmap=plt.cm.hot)
    # plt.colorbar()
    # plt.savefig(f'{counter}_test_input.png', dpi=300, bbox_inches='tight')  # dpi for resolution, bbox_inches to fit content
    # plt.close()

    # # plt.pcolor(train_in)
    # plt.imshow(train_in, cmap=plt.cm.hot)
    # plt.colorbar()
    # plt.savefig(f'{counter}_train_in.png', dpi=300, bbox_inches='tight')  # dpi for resolution, bbox_inches to fit content
    # plt.close()

    # # plt.pcolor(train_out)
    # plt.imshow(train_out, cmap=plt.cm.hot)
    # plt.colorbar()
    # plt.savefig(f'{counter}_train_out.png', dpi=300, bbox_inches='tight')  # dpi for resolution, bbox_inches to fit content
    # plt.close()
    
def prepare_submission_list(submission, test_input_list, key, val, solutions):
    for i, test_input in enumerate(test_input_list):
        _prepare_submission(test_input, key, val, submission, solutions, i)

def prepare_submission(submission, test_input, key, val, solutions):
    _prepare_submission(test_input, key, val, submission, solutions)
    
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading ARC AGI data...")
    first_problem_id, training_pairs, testing_inputs = load_arc_data('training_challenges.json')
    print(f"Loaded {len(training_pairs)} training pairs and {len(testing_inputs)} test inputs.")
    submission = {}
    counter = 0

    # print(testing_inputs)

    # print("#"*80)
    # for key, testing_input in testing_inputs.items():
    #     print(key, f"--{len(np.array(testing_input).shape)}--")
    # print("#"*80)

    solutions = {}
    
    with open('training_solutions.json', 'r') as f:
        solutions = json.load(f)

    # with open('/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json', 'r') as f:
    #     tc = json.load(f)
    
    for key, val in training_pairs.items():
        print(key)
        test_input = testing_inputs[key]
        submission[key] = []
        # print(len(np.array(test_input).shape), "LEN!", np.array(test_input).shape[1])
        if isinstance(test_input, list):
            print("IsList = True")
            prepare_submission_list(submission, test_input, key, val, solutions)
        else:
            print("IsList = False")
            print(np.array(test_input).shape)
            prepare_submission(submission, test_input, key, val, solutions)
        counter += 1

    # Write the data to a file with pretty-printing
    with open("submission.json", "w") as submission_file:
        json.dump(submission, submission_file)
    # with open("tc.json", "w") as tc_file:
    #     json.dump(tc, tc_file)
    print(f"Finished - {counter}!")
    # print(submission)

if __name__ == "__main__":
    main()