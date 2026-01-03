import comfy.samplers
import comfy.utils
import latent_preview

import folder_paths
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os

folder_paths.folder_names_and_paths["yolov8"] = ([os.path.join(folder_paths.models_dir, "yolov8")], folder_paths.supported_pt_extensions)

class Yolov8DSUKSamplerNode:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",), 
                "yolo_model_name": (folder_paths.get_filename_list("yolov8"), ),
                "padding_pixel": ("INT", {"default": 0}),
                #"class_id": ("INT", {"default": 0}),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),

                "upscale_method": (s.upscale_methods,),
                "scale_pixel_to": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),

                "model": ("MODEL",),
                "vae": ("VAE", ),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),

                "edge_blur_pixel": ("INT", {"default": 64, "min": 0})
            },
        }

    RETURN_TYPES = ("IMAGE", "MODEL", "VAE", "CONDITIONING", "IMAGE", "IMAGE")
    RETURN_NAMES = ("Image","model pass through", "vae pass through", "negative pass through", "debug detected area", "debug mask")
    FUNCTION = "sample"
    CATEGORY = "yolov8"

    def sample(self, image, yolo_model_name, padding_pixel, threshold, upscale_method, scale_pixel_to, model, vae, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, edge_blur_pixel):
        base_image = image.clone()
        image_tensor = image
        image_np = image_tensor.cpu().numpy()  # Change from CxHxW to HxWxC for Pillow
        image = Image.fromarray((image_np.squeeze(0) * 255).astype(np.uint8))  # Convert float [0,1] tensor to uint8 image
        
        yolo_model = YOLO(f'{os.path.join(folder_paths.models_dir, "yolov8")}/{yolo_model_name}')
        results = yolo_model(image)
        image_np = np.asarray(image)
        _, MAX_HEIGHT, MAX_WIDTH, _ = base_image.shape
        for r in results:
            boxes = r.boxes
            for i, box in enumerate(boxes):
                if box.conf.cpu().numpy() < threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(x1-padding_pixel, 0)
                y1 = max(y1-padding_pixel, 0)
                x2 = min(x2+padding_pixel, MAX_WIDTH)
                y2 = min(y2+padding_pixel, MAX_HEIGHT)
                cropped_image_np = image_np[y1:y2, x1:x2]
                cropped_img_tensor_out = torch.tensor(cropped_image_np.astype(np.float32) / 255.0).unsqueeze(0)
                _, H, W, _ = cropped_img_tensor_out.shape

                if "seg" in yolo_model_name or "Seg" in yolo_model_name or "SEG" in yolo_model_name:
                    # segmentation mask
                    masks = results[0].masks.data
                    boxes = results[0].boxes.data

                    # extract classes
                    clss = boxes[:, 5]
                    #class_indices = torch.where(clss == class_id)   # get indices of results where class is 0 (people in COCO)
                    class_indices = torch.where(clss == 0)   # get indices of results where class is 0 (people in COCO)
                    class_masks = masks[class_indices]  # use these indices to extract the relevant masks. shape = (1, H, W)

                    # upscale the mask to the original size since mask is shrinked
                    # 検出箇所が２以上だと class_masks[i] でループ処理する
                    class_masks_np = np.asarray(Image.fromarray((class_masks.cpu().numpy().squeeze(0) * 255).astype(np.uint8)))
                    class_masks_np = self.trim_zero_padding(class_masks_np)
                    mask_np = cv2.resize(class_masks_np, (W, H), interpolation=cv2.INTER_AREA)
                    mask_np = np.asarray(mask_np/255).astype(np.float32)
                    # mask_tensor = torch.any(torch.tensor(scalled).unsqueeze(0), dim=0) * 255 # mask_tensor.shape = torch.Size([H, W])
                    # cropped_mask_tensor = torch.any(torch.tensor(scalled[y1:y2, x1:x2]).unsqueeze(0), dim=0) * 255

                else:
                    # box mask
                    mask_np = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)
                    mask_np[y1:y2, x1:x2] = 1.0
                    # mask_tensor = torch.tensor(mask).unsqueeze(0)  # (1, H, W)
                    # cropped_mask_tensor = mask_tensor

                # scale ratio
                target_length = min(H, W)
                scale_by = scale_pixel_to / target_length
                if scale_by >= 16:
                    print(f"Cropped pixel shape is H={H}, W={W}, scale_by={scale_by}")
                    print("This object is skipped due to the region is too small.")
                    continue

                """
                Upscale
                """
                samples = cropped_img_tensor_out.movedim(-1,1)
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)
                s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
                s = s.movedim(1,-1)
            
                """
                VAE Encode
                """
                latent_image = {"samples": vae.encode(s[:,:,:,:3])}
                """
                Sample
                """
                samples = self.common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)

                """
                VAE Decode
                """
                images = vae.decode(samples[0]["samples"])
                if len(images.shape) == 5: #Combine batches
                    images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
                    
                """
                Downscale
                """
                samples = images.movedim(-1,1)
                width = round(samples.shape[3] * (1/scale_by))
                height = round(samples.shape[2] * (1/scale_by))
                s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
                s = s.movedim(1,-1)

                """
                Composite
                """
                base_image = self.composite(s, mask_np, base_image, x1, y1, edge_blur_pixel)

        return (base_image, model, vae, negative, self.result_to_debug_image(results), torch.unsqueeze(torch.tensor(mask_np), 0))


    def result_to_debug_image(self, results):
        im_array = results[0].plot()
        im = Image.fromarray(im_array[...,::-1])  # RGB PIL image

        image_tensor_out = torch.tensor(np.array(im).astype(np.float32) / 255.0)  # Convert back to CxHxW
        return torch.unsqueeze(image_tensor_out, 0)


    def common_ksampler(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
        latent_image = latent["samples"]
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                    denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                    force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
        out = latent.copy()
        out["samples"] = samples
        return (out, )

    def trim_zero_padding(self, mask: np.ndarray) -> np.ndarray:
        nz = np.argwhere(mask)
        if nz.size == 0:
            return mask[:0, :0]
        (r0, c0), (r1, c1) = nz.min(0), nz.max(0)
        return mask[r0:r1+1, c0:c1+1]

    def composite(self, overlap_image, mask_np, base_image, x, y, edge_blur_pixel):
        base = np.asarray(base_image.squeeze(0) * 255).astype(np.uint8)
        overlap = np.asarray(overlap_image.squeeze(0) * 255).astype(np.uint8)

        H, W, _ = base.shape
        oh, ow, _ = overlap.shape

        # generate alpha channel mask
        alpha = np.ones((oh, ow), dtype=np.float32)
        e = max(1, edge_blur_pixel)
        for i in range(e):
            fade = i / e
            alpha[i, :] = np.minimum(alpha[i, :], fade)              # 上
            alpha[oh - 1 - i, :] = np.minimum(alpha[oh - 1 - i, :], fade)  # 下
            alpha[:, i] = np.minimum(alpha[:, i], fade)              # 左
            alpha[:, ow - 1 - i] = np.minimum(alpha[:, ow - 1 - i], fade)  # 右

        # process mask
        overlap_rgb = overlap[..., :3].astype(np.float32)
        scaled_mask_np = cv2.resize(mask_np.astype(np.float32), (ow, oh), interpolation=cv2.INTER_AREA)
        alpha = alpha * scaled_mask_np

        # expand an image channel
        alpha3 = np.expand_dims(alpha, axis=-1)

        # coords
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + ow), min(H, y + oh)
        FW = x2 - x1
        FH = y2 - y1

        # crop base image
        region_base = base[y1:y2, x1:x2, :].astype(np.float32)
        region_overlap = overlap_rgb[0:FH, 0:FW ,:]
        region_alpha = alpha3[0:FH, 0:FW ,:]

        # --- adjust base channel count (in case of mismatch) ---
        if region_base.shape[2] != region_overlap.shape[2]:
            # drop alpha if base has 4ch
            region_base = region_base[..., :3]

        # --- blend ---
        blended = region_base * (1 - region_alpha) + region_overlap * region_alpha
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # --- write back ---
        base[y1:y2, x1:x2, :region_base.shape[2]] = blended

        # --- to tensors for ComfyUI ---
        result_tensor = torch.tensor(base.astype(np.float32) / 255.0).unsqueeze(0)

        return result_tensor

NODE_CLASS_MAPPINGS = {
    "Yolov8DSUKsampler": Yolov8DSUKSamplerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Yolov8DSUKsampler": "Yolov8DSUKsampler",
}
