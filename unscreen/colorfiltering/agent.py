import cv2
import numpy as np
import torch
from sklearn import mixture

from ..utils import dilate_mask, erode_mask, get_target_size


class ColorFilteringAgent():
    """Get alpha map by building GMM models for foreground and background.

    Description: Given a coarse mask of foreground, usually achieved by
        segmentation, ColorFilteringAgent aims to predict the a fine alpha map
        by estimating the probability of each pixels belonging to the
        foregournd. The estimation is done by building GMM models for each
        channel of the foreground and backgound in the HSV color space.

    Args:
        input_long_side (int): long side of the input image would be resize to
        bg_ncomp (Tuple[int]): shape (3,); number of components for (h, s, v)
            channels in GMM of background
        fg_ncomp (Tuple[int]): shape (3,); number of components for (h, s, v)
            channels in GMM of foreground
        max_num_samples (int): the maximum number of samples to feed to GMM for
            training, if samples in an image is larger than max_num_samples, we
            would uniformly sample from the image.
        color_prior_winsize (int, defaut=30):
            the pixels between
            (peak-color_prior_winsize//2, peak+color_prior_winsize//2)
            in h channel would be taken as background, by the observation that
            the green/blue screen would usually have a obvious peak in the h
            channel
        use_opencv_gmm (bool, default=False):
            if set as True, the GMM model in opencv would be used,
            otherwise, GMM in sklearn would be used.

    Attributes:
        input_long_side (int): long side of the input image would be resize to
        bg_ncomp (Tuple[int]): shape (3,);
            number of components for (h, s, v) channels in GMM of background
        fg_ncomp (Tuple[int]): shape (3,);
            number of components for (h, s, v) channels in GMM of foreground
        bg_gmms (List[sklearn.mixture.GaussianMixture or cv2.ml.EM]):
            shape (3,); GMM models for (h, s, v) channels of background
        fg_gmms (List[sklearn.mixture.GaussianMixture or cv2.ml.EM]):
            shape (3,); GMM models for (h, s, v) channels of foreground
    """

    def __init__(self,
                 input_long_side=960,
                 bg_ncomp=(3, 5, 5),
                 fg_ncomp=(10, 10, 10),
                 max_num_samples=10000,
                 color_prior_winsize=30,
                 use_opencv_gmm=False):
        assert isinstance(input_long_side, int)
        self.input_long_side = input_long_side
        assert len(bg_ncomp) == 3
        assert len(fg_ncomp) == 3
        self.bg_ncomp = bg_ncomp
        self.fg_ncomp = fg_ncomp
        assert isinstance(max_num_samples, int)
        assert max_num_samples > 2
        self.max_num_samples = max_num_samples
        assert isinstance(color_prior_winsize, int)
        assert color_prior_winsize > 0
        self.color_prior_winsize = color_prior_winsize
        self.use_opencv_gmm = use_opencv_gmm
        # initialize GMM models
        self.reset_gmms()

    def is_trained(self):
        """check if the gmms are trained.

        Returns:
            self._is_trained (bool): True if gmms have been trained, False
                otherwise
        """
        return self._is_trained

    def reset_gmms(self):
        """initialize GMM models for both foreground and background.

        Here the number of components are determined by self.bg_ncomp
        and self.fg_ncomp; covariance_type of GMM is set as `spherical`;
        and warm_start is set as True for speeding up.
        """
        self.bg_gmms = []
        self.fg_gmms = []
        for i in range(3):
            if self.use_opencv_gmm:
                bgmodel = cv2.ml.EM_create()
                bgmodel.setClustersNumber(self.bg_ncomp[i])
                bgmodel.setCovarianceMatrixType(cv2.ml.EM_COV_MAT_SPHERICAL)
                self.bg_gmms.append(bgmodel)
                fgmodel = cv2.ml.EM_create()
                fgmodel.setClustersNumber(self.fg_ncomp[i])
                fgmodel.setCovarianceMatrixType(cv2.ml.EM_COV_MAT_SPHERICAL)
                self.fg_gmms.append(fgmodel)
            else:
                self.bg_gmms.append(
                    mixture.GaussianMixture(
                        n_components=self.bg_ncomp[i],
                        covariance_type='spherical',
                        warm_start=True))
                self.fg_gmms.append(
                    mixture.GaussianMixture(
                        n_components=self.fg_ncomp[i],
                        covariance_type='spherical',
                        warm_start=True))
        self._is_trained = False

    def get_color_prior(self, img_hsv, mask, color_prior_winsize=None):
        """get the mask of background by color prior of the h channel.

        Since the background is usually pure color, e.g. green or bule screen,
        there would be a obvious peak in the histogram of h channel.
        With such prior, we can first find the peak and only keep the pixels
        near the peak within a pre-defined window size, which would highly
        reduce the influence of the noise.

        Args:
            img_hsv (np.array<uint8>): the input image, in HSV color space
            mask (np.array<bool>): the coarse mask of background
            color_prior_winsize (int, optional):
                the pixels between
                (peak-color_prior_winsize//2, peak+color_prior_winsize//2)
                in h channel would be taken as background, by the observation
                that the green/blue screen would usually have a obvious peak in
                the h channel

        Returns:
            mask_by_prior (np.array<bool>): the mask of background calculated
                by prior
        """
        if color_prior_winsize is None:
            color_prior_winsize = self.color_prior_winsize
        samples = img_hsv[:, :, 0][mask].astype(np.float)
        if len(samples) > self.max_num_samples:
            step = len(samples) // self.max_num_samples
            samples = samples[::step]
        hist, bins = np.histogram(samples, 256, [0, 256])
        peak = np.argmax(hist)
        mask_by_prior = (img_hsv[:, :, 0] > peak - color_prior_winsize//2) \
            & (img_hsv[:, :, 0] < peak + color_prior_winsize//2)
        return mask_by_prior

    def fit_bg_gmms(self, img_hsv, mask, mask_by_prior=None):
        """To fit the GMM models of background.

        The mask would first ensemble with the color prior (by interseaction),
        and then we samples from image with the ensembled mask to fit the GMMs.

        Args:
            img_hsv (np.array<uint8>): the input image, in HSV color space
            mask (np.array<bool>): the coarse mask of background
            mask_by_prior (np.array<bool>): the background mask achieved by
                color prior see "get_color_prior" for details
        """
        if mask_by_prior is None:
            mask_by_prior = self.get_color_prior(img_hsv, mask)
        mask = mask & mask_by_prior
        for i in range(3):
            samples = img_hsv[:, :, i][mask].astype(np.float)
            if len(samples) > self.max_num_samples:
                step = len(samples) // self.max_num_samples
                samples = samples[::step]
            if self.use_opencv_gmm:
                self.bg_gmms[i].trainEM(samples[..., np.newaxis])
            else:
                self.bg_gmms[i].fit(samples[..., np.newaxis])
        self._is_trained = True

    def fit_fg_gmms(self, img_hsv, mask, mask_by_prior=None):
        """To fit the GMM models of foreground.

        Sample from image with the foreground mask to fit the GMMs.

        Args:
            img_hsv (np.array<uint8>): the input image, in HSV color space
            mask (np.array<bool>): the coarse mask of foreground
            mask_by_prior (np.array<bool>): the background mask achieved by
                color prior see "get_color_prior" for details
        """
        if mask_by_prior is None:
            mask_by_prior = self.get_color_prior(img_hsv, (1 - mask),
                                                 self.color_prior_winsize // 5)
        if (mask & (1 - mask_by_prior)).sum() > max(self.fg_ncomp) * 5:
            mask = (mask & (1 - mask_by_prior).astype(np.bool))
        for i in range(3):
            samples = img_hsv[:, :, i][mask].astype(np.float)
            if len(samples) > self.max_num_samples:
                step = len(samples) // self.max_num_samples
                samples = samples[::step]
            if self.use_opencv_gmm:
                self.fg_gmms[i].trainEM(samples[..., np.newaxis])
            else:
                self.fg_gmms[i].fit(samples[..., np.newaxis])
        self._is_trained = True

    def get_prob_by_gmm(self, samples, gmm):
        """inference with a GMM model.

        Args:
            samples (np.array<uint8>): shape (1, n)
                the samples for inference
            gmm (sklearn.mixture.GaussianMixture or cv2.ml.EM): the GMM model

        Returns:
            prob (np.array<float>): shape (n,)
                probabilities of each sample
        """
        if self.use_opencv_gmm:
            means = gmm.getMeans().squeeze()
            stds = np.sqrt(np.array(gmm.getCovs()).squeeze())
            weigths = gmm.getWeights().squeeze()
        else:
            means = gmm.means_.squeeze()
            stds = np.sqrt(gmm.covariances_.squeeze())
            weigths = gmm.weights_.squeeze()
        samples_tensor = torch.from_numpy(samples).float()
        means_tesnor = torch.from_numpy(means[..., np.newaxis]).float()
        stds_tensor = torch.from_numpy(stds[..., np.newaxis]).float()
        weigths_tensor = torch.from_numpy(weigths).float().unsqueeze(dim=0)
        x = (samples_tensor - means_tesnor) / stds_tensor
        y = 1. / (stds_tensor * np.sqrt(2 * np.pi)) * \
            torch.exp(-1. / 2 * torch.pow(x, 2))
        # weighted sum of the probs from each component
        prob = torch.mm(weigths_tensor, y).squeeze()
        return prob

    def get_alpha_by_gmm(self, img_hsv):
        """calculate alpha map with foreground and background gmms.

        Args:
            img_hsv (np.array<uint8>): the input image, in HSV color space

        Returns:
            alpha (np.array<uint8>): shape (h, w)
                predicted alpha map
            confidence (float): confidence of the predicted alpha
        """
        h, w, _ = img_hsv.shape
        bg_prob = torch.ones(h * w, ).float()
        fg_prob = torch.ones(h * w, ).float()
        for i in range(3):
            samples = img_hsv[:, :, i].astype(np.float).reshape(1, -1)
            bg_prob *= self.get_prob_by_gmm(samples, self.bg_gmms[i])
            fg_prob *= self.get_prob_by_gmm(samples, self.fg_gmms[i])
        bg_prob = torch.pow(bg_prob, 1 / 3.)
        fg_prob = torch.pow(fg_prob, 1 / 3.)
        prob = fg_prob / (bg_prob + fg_prob + 1e-6)
        # confidence is estimated by the std of the probability distribution
        confidence = torch.std(prob).item
        prob = prob.view(h, w).numpy()
        alpha = np.clip(prob * 255, 0, 255).astype(np.uint8)
        return alpha, confidence

    def postprocess(self, alpha, mask, thr_ratio=0.8):
        """postprocessing the predicted alpha by GMM.

        This function would:
        1. get an adaptive threshold to set the area of background as 0
        2. run dilate->erode->erode->dilate to remove noise

        Args:
            alpha (np.array<np.uint8>): the predicted alpha map by GMMs
            mask (np.array<np.uint8>): segmentation mask
            thr_ratio (float): (mean_score * thr_ratio) would be the final
                threshold to determine where to be set as background, here
                mean_score is the mean score of the consistent area from "mask"
                and "alpha"

        Returns:
            alpha (np.array<np.uint8>): the final alpha map
        """
        pred_score = alpha.copy().astype(np.float)
        consistent_area = (alpha > 128) & (mask > 0)
        score_thr = pred_score[consistent_area].mean() * thr_ratio
        alpha[alpha < score_thr] = 0
        alpha = erode_mask(dilate_mask(alpha, 3, 2), 3, 2)
        alpha = dilate_mask(erode_mask(alpha, 3, 2), 3, 2)
        return alpha

    def forward(self, img, mask, iters=1):
        """infrence alpha, background with color filtering.

        Args:
            img (np.array<uint8>): the input image, in BGR color space
            mask (np.array<uint8>): the coarse mask of foreground,
                usually achieved by segmentation
            iters (int): iterations to run color filtering
                iters=0 means that prediction is run without fitting GMM model

        Returns:
            alpha (np.array<uint8>): shape (h, w)
                predicted alpha map
            bg_img (np.array<uint8>): shape (h, w, 3)
                predicted background, which is a pure color background
            confidence (float): confidence of the predicted alpha
        """
        # if no foreground
        if (mask > 128).sum() < max(self.fg_ncomp) * 5:
            return mask, img, 1.0
        # if no background
        if (mask < 128).sum() < max(self.bg_ncomp) * 5:
            return mask, np.zeros_like(img), 1.0

        # convert image to HSV color space
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # resize image to target size
        ori_h, ori_w = img_hsv.shape[:2]
        target_h, target_w = get_target_size(ori_h, ori_w,
                                             self.input_long_side)
        img_hsv = cv2.resize(img_hsv, (target_w, target_h))
        mask = cv2.resize(mask, (target_w, target_h))

        # calculate alpha map
        if iters == 0:
            alpha, confidence = self.get_alpha_by_gmm(img_hsv)
            alpha = self.postprocess(alpha, mask)
        else:
            for i in range(iters):
                bg_mask_by_prior = self.get_color_prior(
                    img_hsv, mask < 128, self.color_prior_winsize)
                fg_mask_by_prior = self.get_color_prior(
                    img_hsv, mask < 128, self.color_prior_winsize // 5)
                # train GMM models
                self.fit_bg_gmms(img_hsv, mask < 128, bg_mask_by_prior)
                self.fit_fg_gmms(img_hsv, mask > 128, fg_mask_by_prior)
                # predict
                alpha, confidence = self.get_alpha_by_gmm(img_hsv)
                alpha = self.postprocess(alpha, mask)
                # calculate mask for next itertation according to the predicted
                # alpha
                mask = ((alpha > 128) * 255).astype(np.uint8)
                # if no foreground or background, early stop
                if ((mask > 128).sum() < max(self.fg_ncomp) * 5
                        or (mask < 128).sum() < max(self.bg_ncomp) * 5):
                    break
        # resize alpha to original size
        alpha = cv2.resize(alpha, (ori_w, ori_h))

        # calculate background
        bgimg_hsv = np.zeros((ori_h, ori_w, 3), np.uint8)
        for i in range(3):
            if self.use_opencv_gmm:
                bg_mean = self.bg_gmms[i].getMeans().squeeze()
            else:
                bg_mean = self.bg_gmms[i].means_[0, 0]
            bgimg_hsv[:, :, i] = int(np.mean(bg_mean))
        bg_img = cv2.cvtColor(bgimg_hsv, cv2.COLOR_HSV2BGR)

        return alpha, bg_img, confidence
