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
                    cv.waitKey(50)
                # Converts the depth array valued from 0-1 to real meters
                elif self.env.render_mode == "depth_array":
                    img = self.env.Dcmm.depth_2_meters(img)
                    if self.env.imshow_cam:
                        # Debug: Print stats
                        print(f"Depth stats: Min={img.min():.4f}, Max={img.max():.4f}")
                        
                        # Normalize to 0-255
                        depth_vis = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
                        depth_vis = depth_vis.astype(np.uint8)
                        
                        # Apply Colormap
                        depth_color = cv.applyColorMap(depth_vis, cv.COLORMAP_JET)
                        
                        cv.imshow(camera_name+"_depth", depth_color)
                        cv.waitKey(50)

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
                    cv.waitKey(50)
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

    def get_depth_obs(self, width=84, height=84, camera_name="top", add_noise=True, add_holes=True):
        """
        Get depth observation from a specific camera with realistic noise.
        
        Args:
            width (int): Target width
            height (int): Target height
            camera_name (str): Name of the camera to render
            add_noise (bool): Whether to add realistic depth noise
            add_holes (bool): Whether to add random holes (dropout/cutout)
            
        Returns:
            np.ndarray: Depth image (1, height, width) in uint8 (0-255)
        """
        # Render depth array (0-1 value, where 1 is far plane usually, but depends on renderer)
        # MujocoRenderer returns depth in meters if we use depth_2_meters, 
        # but raw depth_array is usually normalized 0-1 or similar.
        # Let's use the existing render logic which gets depth_array.
        
        depth_img = self.env.mujoco_renderer.render("depth_array", camera_name=camera_name)
        
        # Convert to meters for consistency with other parts, or just use raw 0-1?
        # The user wants "visual robustness".
        # Let's convert to meters first to have a physical meaning for noise?
        # Or just work in 0-1 space. 
        # Existing render() converts to meters: img = self.env.Dcmm.depth_2_meters(img)
        # Let's do that.
        depth_meters = self.env.Dcmm.depth_2_meters(depth_img)
        
        # Resize
        depth_resized = cv.resize(depth_meters, (width, height), interpolation=cv.INTER_AREA)
        
        # Maximum depth for normalization
        max_depth = 3.0
        
        if add_noise:
            # 1. Depth-Dependent Gaussian Noise (far objects have more noise)
            depth_resized = self._add_depth_dependent_noise(depth_resized, max_depth)
            
            # 2. Salt-and-Pepper Noise (specular reflections, sensor saturation)
            depth_resized = self._add_salt_pepper_noise(depth_resized, max_depth)
            
            # 3. Edge Blur (depth discontinuity artifacts)
            depth_resized = self._add_edge_blur(depth_resized)
            
        if add_holes:
            # 4. Cutout Noise (large block dropouts - realistic depth camera holes)
            depth_resized = self._add_cutout_noise(depth_resized)
            
            # 5. Random Pixel Dropout (legacy: 5-10% scattered dropout)
            dropout_rate = np.random.uniform(0.05, 0.10)
            mask = np.random.rand(*depth_resized.shape) > dropout_rate
            depth_resized = depth_resized * mask
            
        # Clip to be non-negative
        depth_resized = np.clip(depth_resized, 0, None)
        
        # Normalize to 0-255 uint8
        depth_norm = np.clip(depth_resized / max_depth, 0, 1)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        
        # Add channel dimension (1, H, W)
        return np.expand_dims(depth_uint8, axis=0)

    def _add_cutout_noise(self, depth_img, num_cutouts_range=(1, 3), size_ratio_range=(0.05, 0.15)):
        """
        Add rectangular cutout noise to simulate large depth camera holes.
        
        Real depth cameras often have large contiguous regions of missing data
        due to occlusion, reflective surfaces, or sensor limitations.
        
        Args:
            depth_img: Depth image in meters
            num_cutouts_range: Range for number of cutout blocks
            size_ratio_range: Range for cutout size as ratio of image dimensions
        
        Returns:
            np.ndarray: Depth image with cutout holes
        """
        h, w = depth_img.shape
        num_cutouts = np.random.randint(num_cutouts_range[0], num_cutouts_range[1] + 1)
        
        for _ in range(num_cutouts):
            # Random size for this cutout
            size_ratio = np.random.uniform(size_ratio_range[0], size_ratio_range[1])
            ch = max(1, int(h * size_ratio))
            cw = max(1, int(w * size_ratio))
            
            # Random position
            y = np.random.randint(0, max(1, h - ch))
            x = np.random.randint(0, max(1, w - cw))
            
            # Apply cutout (set to 0 = black hole)
            depth_img[y:y+ch, x:x+cw] = 0
            
        return depth_img

    def _add_salt_pepper_noise(self, depth_img, max_depth, 
                                salt_ratio_range=(0.01, 0.03), 
                                pepper_ratio_range=(0.02, 0.05)):
        """
        Add salt-and-pepper noise to simulate specular reflections and sensor saturation.
        
        - Salt (white): Far plane / max values - simulates specular reflections
        - Pepper (black): Near plane / 0 values - simulates sensor blind spots
        
        Args:
            depth_img: Depth image in meters
            max_depth: Maximum depth value for salt noise
            salt_ratio_range: Range of pixels affected by salt noise
            pepper_ratio_range: Range of pixels affected by pepper noise
        
        Returns:
            np.ndarray: Depth image with salt-pepper noise
        """
        # Salt (far plane values)
        salt_ratio = np.random.uniform(salt_ratio_range[0], salt_ratio_range[1])
        salt_mask = np.random.rand(*depth_img.shape) < salt_ratio
        depth_img[salt_mask] = max_depth
        
        # Pepper (near plane / 0)
        pepper_ratio = np.random.uniform(pepper_ratio_range[0], pepper_ratio_range[1])
        pepper_mask = np.random.rand(*depth_img.shape) < pepper_ratio
        depth_img[pepper_mask] = 0
        
        return depth_img

    def _add_edge_blur(self, depth_img, edge_noise_std=0.1):
        """
        Add noise specifically at depth edges to simulate depth camera edge artifacts.
        
        Real depth cameras often have significant errors at depth discontinuities
        (flying pixels, mixed pixels).
        
        Args:
            depth_img: Depth image in meters
            edge_noise_std: Standard deviation of noise added to edge pixels
        
        Returns:
            np.ndarray: Depth image with edge noise
        """
        # Detect edges using Sobel gradient magnitude
        sobel_x = cv.Sobel(depth_img.astype(np.float32), cv.CV_32F, 1, 0, ksize=3)
        sobel_y = cv.Sobel(depth_img.astype(np.float32), cv.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize gradient and threshold to find edges
        gradient_norm = gradient_mag / (gradient_mag.max() + 1e-6)
        edge_mask = gradient_norm > 0.1  # Threshold for edge detection
        
        # Add noise only to edge pixels
        noise = np.random.normal(0, edge_noise_std, depth_img.shape)
        depth_img = depth_img + edge_mask * noise
        
        return depth_img

    def _add_depth_dependent_noise(self, depth_img, max_depth, base_std=0.01, scale_factor=0.03):
        """
        Add Gaussian noise that increases with depth.
        
        Real depth cameras have noise that scales with distance:
        - Near objects: low noise
        - Far objects: high noise
        
        Args:
            depth_img: Depth image in meters
            max_depth: Maximum depth for scaling
            base_std: Base noise standard deviation (for near objects)
            scale_factor: Additional noise scaling with depth
        
        Returns:
            np.ndarray: Depth image with depth-dependent Gaussian noise
        """
        # Noise std increases linearly with depth
        depth_ratio = np.clip(depth_img / max_depth, 0, 1)
        noise_std = base_std + scale_factor * depth_ratio
        
        # Generate and apply noise
        noise = np.random.randn(*depth_img.shape) * noise_std
        depth_img = depth_img + noise
        
        return depth_img

