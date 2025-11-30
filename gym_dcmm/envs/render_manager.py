"""
Rendering management for DcmmVecEnv.
Handles image rendering and depth processing.
"""

import numpy as np
import cv2 as cv


class RenderManager:
    """Manages rendering for the environment."""

    def __init__(self, env):
        """
        Initialize render manager.

        Args:
            env: Reference to the parent DcmmVecEnv instance
        """
        self.env = env

    def preprocess_depth_with_mask(self, rgb_img, depth_img,
                                   depth_threshold=3.0,
                                   num_white_points_range=(5, 15),
                                   point_size_range=(1, 5)):
        """
        Preprocess depth image with RGB mask and add noise.

        Args:
            rgb_img: RGB image array
            depth_img: Depth image array
            depth_threshold: Maximum depth value to consider
            num_white_points_range: Range for number of white noise points
            point_size_range: Size range for white noise points

        Returns:
            tuple: (masked_depth_img, masked_depth_mean)
        """
        # Define RGB Filter
        lower_rgb = np.array([5, 0, 0])
        upper_rgb = np.array([255, 15, 15])
        rgb_mask = cv.inRange(rgb_img, lower_rgb, upper_rgb)
        depth_mask = cv.inRange(depth_img, 0, depth_threshold)
        combined_mask = np.logical_and(rgb_mask, depth_mask)
        # Apply combined mask to depth image
        masked_depth_img = np.where(combined_mask, depth_img, 0)
        # Calculate mean depth within combined mask
        masked_depth_mean = np.nanmean(np.where(combined_mask, depth_img, np.nan))
        # Generate random number of white points
        num_white_points = np.random.randint(num_white_points_range[0], num_white_points_range[1])
        # Generate random coordinates for white points
        random_x = np.random.randint(0, depth_img.shape[1], size=num_white_points)
        random_y = np.random.randint(0, depth_img.shape[0], size=num_white_points)
        # Generate random sizes for white points in the specified range
        random_sizes = np.random.randint(point_size_range[0], point_size_range[1], size=num_white_points)
        # Create masks for all white points at once
        y, x = np.ogrid[:masked_depth_img.shape[0], :masked_depth_img.shape[1]]
        point_masks = ((x[..., None] - random_x) ** 2 + (y[..., None] - random_y) ** 2) <= random_sizes ** 2
        # Update masked depth image with the white points
        masked_depth_img[np.any(point_masks, axis=2)] = np.random.uniform(1.5, 3.0)

        return masked_depth_img, masked_depth_mean

    def render(self):
        """
        Render the current state.

        Returns:
            np.ndarray: Rendered image(s)
        """
        imgs = np.zeros((0, self.env.img_size[0], self.env.img_size[1]))
        imgs_depth = np.zeros((0, self.env.img_size[0], self.env.img_size[1]))

        for camera_name in self.env.camera_name:
            if self.env.render_mode == "human":
                self.env.mujoco_renderer.render(
                    self.env.render_mode, camera_name = camera_name
                )
                return imgs
            elif self.env.render_mode != "depth_rgb_array":
                img = self.env.mujoco_renderer.render(
                    self.env.render_mode, camera_name = camera_name
                )
                if self.env.imshow_cam and self.env.render_mode == "rgb_array":
                    cv.imshow(camera_name, cv.cvtColor(img, cv.COLOR_BGR2RGB))
                    cv.waitKey(1)
                # Converts the depth array valued from 0-1 to real meters
                elif self.env.render_mode == "depth_array":
                    img = self.env.Dcmm.depth_2_meters(img)
                    if self.env.imshow_cam:
                        depth_norm = np.zeros(img.shape, dtype=np.uint8)
                        cv.convertScaleAbs(img, depth_norm, alpha=(255.0/img.max()))
                        cv.imshow(camera_name+"_depth", depth_norm)
                        cv.waitKey(1)

                    # Resize to match self.img_size
                    img = cv.resize(img, (self.env.img_size[1], self.env.img_size[0]))

                    img = np.expand_dims(img, axis=0)
                    imgs = np.concatenate((imgs, img), axis=0)
            else:
                img_rgb = self.env.mujoco_renderer.render(
                    "rgb_array", camera_name = camera_name
                )
                img_depth = self.env.mujoco_renderer.render(
                    "depth_array", camera_name = camera_name
                )
                # Converts the depth array valued from 0-1 to real meters
                img_depth = self.env.Dcmm.depth_2_meters(img_depth)
                img_depth, _ = self.preprocess_depth_with_mask(img_rgb, img_depth)
                if self.env.imshow_cam:
                    cv.imshow(camera_name+"_rgb", cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB))
                    cv.imshow(camera_name+"_depth", img_depth)
                    cv.waitKey(1)
                img_depth = cv.resize(img_depth, (self.env.img_size[1], self.env.img_size[0]))
                img_depth = np.expand_dims(img_depth, axis=0)
                imgs_depth = np.concatenate((imgs_depth, img_depth), axis=0)
            # Sync the viewer (if exists) with the data
            if self.env.Dcmm.viewer != None:
                self.env.Dcmm.viewer.sync()

        if self.env.render_mode == "depth_rgb_array":
            # Only keep the depth image
            imgs = imgs_depth

        return imgs

