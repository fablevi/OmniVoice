import logging
# Importáljuk az imént írt wrapper függvényeit
from inference_wrapper import load_omnivoice_openvino, generate_text_to_audio

def main():
    # Logging beállítása, hogy lássuk a szép időbélyegeket
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO, force=True)

    # 1. A generálandó szövegek tömbje (listája)
    szovegek = [
        "Szia! Ez az első szöveg, amit a tömbből generálok.",
        "Ez pedig a második mondat, ami már sokkal gyorsabban lefut.",
        "Végezetül a harmadik elem a listából, közvetlenül a videókártyáról."
    ]

    # 2. Modell betöltése EGYSZER (Alapértelmezetten GPU-ra)
    # Ha CPU-n akarnád: device="CPU"
    model, stats = load_omnivoice_openvino(
        #model_name="k2-fsa/OmniVoice",
        model_name="/home/fablevi/.cache/huggingface/hub/OmniVoice",
        ir_path_str="openvino_ir/omnivoice-step-int8.xml",
        device="GPU"
    )

    # 3. Végigmegyünk az array (lista) elemein
    for index, szoveg in enumerate(szovegek):
        # Dinamikusan generálunk egyedi fájlneveket: kimenet_0.wav, kimenet_1.wav, stb.
        fajlnev = f"kimenet_ref_{index}.wav"
        
        # Opcionálisan adhatsz hozzá stílust is, pl: instruct="female, calm"
        generate_text_to_audio(
            model=model,
            stats=stats,
            text=szoveg,
            output_path=fajlnev,
            ref_audio="./mucsi.mp3",
            ref_text="Mi a gecimről beszélsz te vécékefe? Milyen kis probléma? Meghazudtál?",
            device="GPU",
            language="hu",
        )

    logging.info("Minden szöveg sikeresen legenerálva!")

if __name__ == "__main__":
    main()