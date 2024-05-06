# Codes are adapted from: https://github.com/zhengqili/CGIntrinsics

import json

import numpy as np
from skimage.transform import resize


def compute_whdr(reflectance, judgements, delta=0.1):
    points = judgements['intrinsic_points']
    comparisons = judgements['intrinsic_comparisons']
    id_to_points = {p['id']: p for p in points}
    rows, cols = reflectance.shape[0:2]

    error_sum = 0.0
    error_equal_sum = 0.0
    error_inequal_sum = 0.0

    weight_sum = 0.0
    weight_equal_sum = 0.0
    weight_inequal_sum = 0.0

    for c in comparisons:
        # "darker" is "J_i" in our paper
        darker = c['darker']
        if darker not in ('1', '2', 'E'):
            continue

        # "darker_score" is "w_i" in our paper
        weight = c['darker_score']
        if weight <= 0.0 or weight is None:
            continue

        point1 = id_to_points[c['point1']]
        point2 = id_to_points[c['point2']]
        if not point1['opaque'] or not point2['opaque']:
            continue

        # convert to grayscale and threshold
        l1 = max(1e-10, np.mean(reflectance[
                                    int(point1['y'] * rows), int(point1['x'] * cols), ...]))
        l2 = max(1e-10, np.mean(reflectance[
                                    int(point2['y'] * rows), int(point2['x'] * cols), ...]))

        # # convert algorithm value to the same units as human judgements
        if l2 / l1 > 1.0 + delta:
            alg_darker = '1'
        elif l1 / l2 > 1.0 + delta:
            alg_darker = '2'
        else:
            alg_darker = 'E'

        if darker == 'E':
            if darker != alg_darker:
                error_equal_sum += weight

            weight_equal_sum += weight
        else:
            if darker != alg_darker:
                error_inequal_sum += weight

            weight_inequal_sum += weight

        if darker != alg_darker:
            error_sum += weight

        weight_sum += weight

    if weight_sum:
        return (error_sum / weight_sum, weight_sum > 1e-5), \
               (error_equal_sum / max(weight_equal_sum, 1e-6), weight_equal_sum > 1e-5), \
               (error_inequal_sum / max(weight_inequal_sum, 1e-6), weight_inequal_sum > 1e-5)
    else:
        return None


def evaluate_WHDR(prediction_R, targets, eq_ratio=0.1):
    # num_images = prediction_S.size(0) # must be even number
    total_whdr = float(0)
    total_whdr_eq = float(0)
    total_whdr_ineq = float(0)

    count = float(0)
    count_eq = float(0)
    count_ineq = float(0)

    for i in range(0, prediction_R.size(0)):
        prediction_R_np = prediction_R[i ,: ,: ,:].cpu().numpy()
        # prediction_R_np = np.transpose(np.exp(prediction_R_np * 0.4545), (1 ,2 ,0))
        prediction_R_np = np.transpose(prediction_R_np, (1 ,2 ,0))

        # o_h = targets['oringinal_shape'][0].numpy()
        # o_w = targets['oringinal_shape'][1].numpy()

        # prediction_R_srgb_np = prediction_R_srgb.data[i,:,:,:].cpu().numpy()
        # prediction_R_srgb_np = np.transpose(prediction_R_srgb_np, (1,2,0))

        o_h = targets['original_shape'][0].numpy()
        o_w = targets['original_shape'][1].numpy()
        # resize to original resolution
        prediction_R_np = resize(prediction_R_np, (o_h[i] ,o_w[i]), order=1, preserve_range=True)

        # print(targets["judgements_path"][i])
        # load Json judgement
        judgements = json.load(open(targets["judgements_path"][i]))
        (whdr, _), (whdr_eq, valid_eq), (whdr_ineq, valid_ineq) = compute_whdr(prediction_R_np, judgements, eq_ratio)

        total_whdr += whdr
        count += 1.
        if valid_eq:
            total_whdr_eq += whdr_eq
            count_eq += 1.
        if valid_ineq:
            total_whdr_ineq += whdr_ineq
            count_ineq += 1.

    return (total_whdr, count), (total_whdr_eq, count_eq), (total_whdr_ineq, count_ineq)
