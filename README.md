# roma-safe-cctv

Riconoscimento di **violenza**, **violenza con arma** e **scippi** da telecamere CCTV, con **avviso arma**.

> README provvisorio: la versione completa per la consegna è in preparazione.

Pipeline: YOLO26-pose + tracking delle persone → timeline per finestre di 40 frame a 25 fps →
Graph-Transformer sulle relazioni tra persone e armi → classificazione della scena.

```
uv sync          # ambiente (Python 3.12)
uv run pytest    # test
```

I dataset non sono inclusi nel repository. Le annotazioni in `annotations/uca/` provengono da
[UCA – UCF-Crime Annotation](https://github.com/Xuange923/Surveillance-Video-Understanding) (Apache 2.0).
