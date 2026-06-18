import time
import logging
from pathlib import Path
from datetime import datetime
import soundfile as sf
import torch

from omnivoice.models.omnivoice import OmniVoice
# Importáljuk a GPU/CPU leképezést végző logikát a módosított scriptből
from omnivoice.cli.infer_openvino_gpu import install_openvino_forward

def load_omnivoice_openvino(model_name: str = "k2-fsa/OmniVoice", ir_path_str: str = "openvino_ir/omnivoice-step-int8.xml", device: str = "GPU"):
    """Betölti a modellt és felülírja az OpenVINO-s forward pass-al (GPU/CPU)."""
    ir_path = Path(ir_path_str).resolve()
    if not ir_path.exists():
        raise FileNotFoundError(f"OpenVINO IR nem található: {ir_path}")

    logging.info(f"Modell betöltése: {model_name}...")
    model = OmniVoice.from_pretrained(model_name, attn_implementation="sdpa")
    model.to("cpu").eval()
    
    logging.info("A modell sikeresen betöltve!")
    logging.info("Várakozás 2 másodpercig...")
    time.sleep(2)

    # OpenVINO GPU/CPU meghajtás regisztrálása
    stats = install_openvino_forward(
        model,
        ir_path=ir_path,
        device=device,
        static_reshape=False,
    )
    
    return model, stats

def generate_text_to_audio(model, stats, text: str, output_path: str, device: str = "GPU", instruct: str = None, language: str = "Hungarian"):
    """Legenerál egyetlen szöveget a betöltött modellel, állítható nyelven."""
    logging.info(f"Audio generálása ({language}): {text[:50]}...")
    
    start_time_str = datetime.now().strftime("%H:%M:%S")
    logging.info(f"Generálás elkezdve ekkor: {start_time_str}")
    
    t0 = time.perf_counter()
    audios = model.generate(
        text=text,
        language=language,    
        instruct=instruct,
        num_step=32,          
        guidance_scale=2.0    
    )
    wall = time.perf_counter() - t0
    
    end_time_str = datetime.now().strftime("%H:%M:%S")
    logging.info(f"Generálás befejezve ekkor: {end_time_str}")

    sf.write(output_path, audios[0], model.sampling_rate)
    logging.info(f"Mentve ide: {output_path} (Idő: {wall:.2f}s)\n" + "-"*40)