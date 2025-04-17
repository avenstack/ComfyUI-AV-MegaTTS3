import json
import os
import glob
import librosa
import numpy as np
import torch

import folder_paths

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from .MegaTTS3 import MegaTTS3DiTInfer

models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "TTS")

def load_voices():
    voices_dir = os.path.join(model_path, "MegaTTS3", "Voices")
    if not os.path.exists(voices_dir):
        os.makedirs(voices_dir, exist_ok=True)
        return []

    voices = [os.path.basename(x) for x in glob.glob(os.path.join(voices_dir, "*.wav"))]
    voices += [os.path.basename(x) for x in glob.glob(os.path.join(voices_dir, "*.mp3"))]
    return voices


class AVMegaTTS3:
    infer_ins_cache = None
    @classmethod
    def INPUT_TYPES(s):
        voices = load_voices()
        default_voice = voices[0] if voices else ""
        return {
            "required": {
                "voice":(voices, {"default": default_voice}),
                "text": ("STRING",),
                "language": (["en", "zh"], {"default": "zh"}),
                "step": ("INT", {"default": 32, "min": 1, "step": 1, "tooltip": "Higher number = better quality but slower generation"}),
                "strength": ("FLOAT", {"default":1.6, "min": 0.1, "step": 0.1, "tooltip": "How closely to follow the text pronunciation"}),
                "similarity": ("FLOAT", {"default": 2.5, "min": 0.1, "step": 0.1, "tooltip": "How similar to the reference voice"}),
                "unload_model": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "Aven/AV-MegaTTS3"

    def generate_speech(self, voice, text, language, step, strength, similarity, unload_model):
        voice_path = os.path.join(model_path, "MegaTTS3", "voices", voice)
        if AVMegaTTS3.infer_ins_cache is not None:
            infer_ins = AVMegaTTS3.infer_ins_cache
        else:
            ckpt_root = os.path.join(model_path, "MegaTTS3")
            infer_ins = MegaTTS3DiTInfer(ckpt_root=ckpt_root, device="cuda")
            AVMegaTTS3.infer_ins_cache = infer_ins
        with open(voice_path, 'rb') as file:
            file_content = file.read()

        latent_file = voice_path.replace('.wav', '.npy')
        print(f"latent_file: {latent_file}")
        if os.path.exists(latent_file):
            resource_context = infer_ins.preprocess(file_content, latent_file=latent_file)
        else:
            raise Exception("latent_file not found")
        audio_data = infer_ins.forward(
            resource_context, text, 
            language_type=language, 
            time_step=step, p_w=strength, t_w=similarity)

        if unload_model:
            import gc
            if AVMegaTTS3.infer_ins_cache is not None:
                infer_ins.clean()
                AVMegaTTS3.infer_ins_cache = None
                gc.collect()
                torch.cuda.empty_cache()
                print("MegaTTS3S memory cleanup successful")

        return (audio_data,)


class AVPromptInit:
    @classmethod
    def INPUT_TYPES(cls):
               
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True, 
                    "default": ""}),
                },
        }

    CATEGORY = "Aven/AV-MegaTTS3"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "promptgen"
    
    def promptgen(self, prompt: str):
        return (prompt.strip(),)


NODE_CLASS_MAPPINGS = {
    "AVMegaTTS3": AVMegaTTS3,
    "AVPromptInit": AVPromptInit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AVMegaTTS3": "AV Mega TTS3",
    "AVPromptInit": "AV Prompt Init",
}