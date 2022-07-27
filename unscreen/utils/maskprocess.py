"""some functions for mask processing."""
import cv2
import numpy as np
import pdb


def dilate_mask(mask, kernelsize=5, iters=10):
    """dilate a mask.
    Args:
        mask (np.array<np.uint8>): the input mask
        kernelsize (int): the kernel size for dilation
        iters (int): the iterations for dilation
    Returns:
        dilated_mask(np.array<np.uint8>): the mask after dilation
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernelsize, kernelsize))
    dilated_mask = cv2.dilate(mask, kernel, iterations=iters)
    return dilated_mask


def erode_mask(mask, kernelsize=5, iters=10):
    """erode a mask.
    Args:
        mask (np.array<np.uint8>): the input mask
        kernelsize (int): the kernel size for erosion
        iters (int): the iterations for erosion
    Returns:
        dilated_mask (np.array<np.uint8>): the mask after erosion
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (kernelsize, kernelsize))
    eroded_mask = cv2.erode(mask, kernel, iterations=iters)
    return eroded_mask


def get_fgbox(fgmask, padsize=5):
    """get the bounding box of the foreground according to its mask.
    Args:
        fgmask (np.array<np.uint8>): the mask of foreground
        padsize (int): the pixels to pad outer the bounding box
    Returns:
        left (int): left of the foreground bounding box
        right (int): right of the foreground bounding box
        top (int): top of the foreground bounding box
        bottom (int): bottom of the foreground bounding box
    """
    h, w = fgmask.shape
    x, y = np.where(fgmask > 0)
    left, right, top, bottom = np.min(x), np.max(x), np.min(y), np.max(y)
    left, right = max(left - padsize, 0), min(right + padsize, h)
    top, bottom = max(top - padsize, 0), min(bottom + padsize, w)
    return left, right, top, bottom


def exist_foreground(mask, fg_exist_thr):
    h, w = mask.shape
    if (mask >= 128).sum() > fg_exist_thr * h * w:
        return True
    return False


def get_outer_boundary(mask, kernelsize=7, iters=10):
    """get the outer boundary of a mask.
    Args:
        mask (np.array<np.uint8>): the input mask
        kernelsize (int): the kernel size for dilation
        iters (int): the iterations for dilation
    Returns:
        boundary (np.array<np.uint8>): the boundary of the mask
    """
    dilated_mask = dilate_mask(mask, kernelsize, iters)
    boundary = np.clip(dilated_mask - mask, 0, 255)
    return boundary


def remove_invalid_objects(cfg, alpha,
                           segmask=None,
                           saliency_thr=0.001,
                           consensus_thr=0.5,
                           score_map=None,
                           score_map_center=(3. / 5, 1. / 2)):
    """remove some invalid objects in the foreground.
    First, all objects in the foreground are found by "findContours".
    Then each object is taken as invalid if:
        1. the saliency score is smaller than saliency_thr; or
        2. the consensus score is smaller than consensus_thr
    Here the saliency score is calculated by both the area and the location of
    the object. The larger the object is, the larger the saliency score is. The
    nearer to the center the object is, the larger the saliency score is. The
    consensus score is calculated by referring to the segmentation mask. If an
    object does not appeared in the segmentation mask, it would get a low
    consensus score.
    Args:
        alpha (np.array<np.uint8>): the predicted alpha channel to indicate the
            foreground
        segmask (np.array<np.uint8>): the segmentation mask
        saliency_thr (np.float): the threshold of saliency score
        consensus_thr (np.float): the threshold of consensus score
        score_map (np.array<np.float>, optional): a score map to encode the
            influence of the location each pixel is in [0, 1], the pixels that
            are nearer to cneter would have larger value. If the score_map is
            not given, the function would create a new map score with
            "score_map_center".
        score_map_center (Tuple[float], optional): the center of the score map
            for saliency estimation, the center would have maximum score, i.e.
            one, and the score would decrease from the center to the border
            linearly, utile zero. Here the center is represented as the ratio
            of the location, which is a float number between (0, 1).
    Returns:
        alpha (np.array<np.uint8>): the updated alpha channel
    """
    saliency_thr = cfg['objectremoval']['saliency_thr']
    consensus_thr = cfg['objectremoval']['consensus_thr']
    if segmask is None:
        segmask = alpha
    # buld score map
    h, w = alpha.shape
    score_map = build_score_map(h, w, cfg)
    if score_map is None:
        score_map = get_score_map((h, w), score_map_center)
    # get objects
    outFindContours = cv2.findContours(alpha, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
    if len(outFindContours) == 3:
        _, objects, _ = outFindContours
    else:
        objects, _ = outFindContours
    valid_objects = np.zeros_like(alpha)
    num_objects = len(objects)
    for i in range(num_objects):
        # exclude some noise for efficiency
        area = cv2.contourArea(objects[i])
        if area < 100:
            continue
        # calculate saliency score
        object_map = cv2.drawContours(
            np.zeros_like(alpha), objects, i, (255, 255, 255), cv2.FILLED)
        saliency_score = score_map[object_map > 0].sum() / float(h * w)
        # calculate consensus score
        consensus_score = segmask[object_map > 0].astype(
            np.float).mean() / 255.
        # check if valid
        # print('{:>10.1f} {:>5.4f} {:>5.4f} {:>5.4f} {:>5.4f}'.format(
        #  area, saliency_score, consensus_score, saliency_thr, consensus_thr))
        if ((saliency_score > saliency_thr and consensus_score > consensus_thr)
                or (saliency_score > saliency_thr * 10)):
            valid_objects = cv2.drawContours(valid_objects, objects, i,
                                             (255, 255, 255), cv2.FILLED)
    # remove invalid objects
    alpha[valid_objects == 0] = 0
    return alpha


def get_score_map(map_size, center):
    """get a score map according to the distance to the center.
    Args:
        map_size (Tuple[int]): the size of the score map.
        center (Tuple[float], optional): the center of the score map for
            saliency estimation, the center would have maximum score, i.e. one,
            and the score would decrease from the center to the border
            linearly, utile zero. Here the center is represented as the
            ratio of the location, which is a float number between (0, 1).
    Returns:
        score_map (np.array<np.float>): a score map to encode the influence of
            the location each pixel is in [0, 1], the pixels that are nearer to
            center would have larger value.
    """
    score_map = np.ones(map_size, np.float)
    h, w = map_size
    y, x = int(h * center[0]), int(w * center[1])
    score_map[:, x:w] = np.linspace(0, 1, w - x)[np.newaxis, ...]**2
    score_map[:, 0:x] = np.linspace(1, 0, x)[np.newaxis, ...]**2
    score_map[y:h] += np.linspace(0, 1, h - y)[..., np.newaxis]**2
    score_map[0:y] += np.linspace(1, 0, y)[..., np.newaxis]**2
    score_map = np.sqrt(score_map)
    score_map = (score_map.max() - score_map) / score_map.max()
    return score_map


def build_score_map(h, w, config):
    """build score map for object removal."""
    centers = config['objectremoval']['score_map_center']
    if w > h:
        center = centers['landscape']
    else:
        center = centers['portrait']
    score_map = get_score_map((h, w), center)
    return score_map
