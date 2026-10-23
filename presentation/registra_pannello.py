"""Registra il pannello di Argo per la demo del video (D61) senza toccare lo schermo: Chrome headless (serve Chrome
vero per i video H.264 dell'archivio) pilotato da Playwright, con un profilo temporaneo.

Copione: qualche secondo sulla vista Telecamere (le telecamere collegate al pannello), poi l'Archivio: per ogni
sessione scelta la apre, la riproduce e, se c'è, clicca sull'evento ALLARME VIOLENZA o WARNING ARMA nel registro.

Uso (Playwright è nell'ambiente dello strumento scrapling):
    ~/.local/share/uv/tools/scrapling/bin/python presentation/registra_pannello.py --url http://127.0.0.1:8765 \
        --sessioni metro_rissa arma_rapina_milton --uscita runs/video/pannello_demo.mp4
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

CHROME = "/opt/google/chrome/chrome"


def registra(url: str, sessioni: list[str], uscita: Path, secondi_telecamere: float, secondi_sessione: float) -> None:
    with tempfile.TemporaryDirectory() as cartella, sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=CHROME, headless=True)
        contesto = browser.new_context(viewport={"width": 1920, "height": 1080}, record_video_dir=cartella,
                                       record_video_size={"width": 1920, "height": 1080})
        pagina = contesto.new_page()
        pagina.goto(url)
        pagina.wait_for_timeout(int(secondi_telecamere * 1000))
        pagina.click("#scheda-archivio")
        pagina.wait_for_timeout(1500)
        for nome in sessioni:
            pagina.locator("#sessioni button", has_text=nome).first.click()
            pagina.wait_for_timeout(800)
            pagina.evaluate("document.querySelector('#video-archivio').play()")
            pagina.wait_for_timeout(int(secondi_sessione * 600))
            evento = pagina.locator("#registro-archivio li.allarme, #registro-archivio li.warning").first
            if evento.count():
                evento.click()
            pagina.wait_for_timeout(int(secondi_sessione * 400))
        percorso_video = pagina.video.path()
        contesto.close()
        browser.close()
        uscita.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", percorso_video, "-r", "25", "-c:v", "libx264", "-crf", "18",
                        "-pix_fmt", "yuv420p", str(uscita)], check=True)
        shutil.copy(percorso_video, uscita.with_suffix(".webm"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--sessioni", nargs="*", default=[])
    parser.add_argument("--uscita", type=Path, default=Path("runs/video/pannello_demo.mp4"))
    parser.add_argument("--secondi-telecamere", type=float, default=8)
    parser.add_argument("--secondi-sessione", type=float, default=12)
    argomenti = parser.parse_args()
    registra(argomenti.url, argomenti.sessioni, argomenti.uscita, argomenti.secondi_telecamere, argomenti.secondi_sessione)
    print(argomenti.uscita)
