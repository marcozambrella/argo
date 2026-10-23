#!/usr/bin/env bash
# Pubblica il sito su GitHub Pages (branch gh-pages).
# Dal 23/10 (scelta di Marco) il sito NON è nel branch main: vive in questa cartella sul PC e, pubblicato, su gh-pages.
# Il sito è statico (niente build), quindi gh-pages contiene già il sorgente completo di ciò che si vede online.
#
# Cosa fa:
#   1. copia questa cartella senza i file di lavoro (CARTELLE_E_FILE_ESCLUSI);
#   2. toglie i riquadri dei testi di Marco ancora da scrivere (D51) e si ferma se ne resta qualcuno;
#   3. controlla che ogni file citato da pagine, script e stile esista nella copia;
#   4. sostituisce il contenuto di gh-pages con la copia e fa un commit normale sopra l'ultimo (niente force push).
#
# Uso:  site/pubblica.sh --prova     prepara tutto e mostra cosa cambierebbe, senza pubblicare
#       site/pubblica.sh             pubblica (azione esterna: solo con la conferma di Marco)
set -euo pipefail

CARTELLE_E_FILE_ESCLUSI=(
  stili montaggio slide .gitignore __pycache__
  video/hero-bozza.mp4 video/argo-apertura.mp4
  immagini/hero-bozza-locandina.jpg immagini/argo-apertura-locandina.jpg
)
solo_prova=false
[ "${1:-}" = "--prova" ] && solo_prova=true

cartella_sito="$(cd "$(dirname "$0")" && pwd)"
radice_repo="$(git -C "$cartella_sito" rev-parse --show-toplevel)"
cartella_lavoro="$(mktemp -d)"
copia_sito="$cartella_lavoro/sito"
cartella_pagine="$cartella_lavoro/gh-pages"
pulisci() {
  git -C "$radice_repo" worktree remove --force "$cartella_pagine" 2>/dev/null || true
  rm -rf "$cartella_lavoro"
}
trap pulisci EXIT

# 1. copia senza i file di lavoro
opzioni_esclusione=()
for escluso in "${CARTELLE_E_FILE_ESCLUSI[@]}"; do opzioni_esclusione+=("--exclude=/$escluso"); done
rsync -a --exclude=__pycache__ "${opzioni_esclusione[@]}" "$cartella_sito"/ "$copia_sito"/

# 2. via i riquadri dei testi non ancora scritti
sed -i '/<p class="segnaposto" data-segnaposto="[A-Za-z]*"><\/p>/d' "$copia_sito/index.html"
sed -i "/<p class=\"segnaposto\">\\[MARCO'S TEXT/d" "$copia_sito/js/passi-modello.js"
if grep -q 'class="segnaposto"' "$copia_sito/index.html" "$copia_sito/js/passi-modello.js"; then
  echo "ERRORE: riquadri dei testi ancora presenti, niente pubblicato" >&2; exit 1
fi

# 3. ogni file locale citato deve esserci
mancanti=0
for percorso in $(grep -ohE '(js|dati|immagini|video)/[A-Za-z0-9_.-]+\.[a-z0-9]+' \
                    "$copia_sito"/index.html "$copia_sito"/app.js "$copia_sito"/stile.css "$copia_sito"/js/*.js | sort -u); do
  if [ ! -f "$copia_sito/$percorso" ]; then echo "ERRORE: manca $percorso" >&2; mancanti=1; fi
done
[ "$mancanti" = 0 ] || exit 1

# 4. gh-pages: stesso contenuto della copia, commit sopra l'ultimo
git -C "$radice_repo" fetch -q origin gh-pages
git -C "$radice_repo" worktree add -q --detach "$cartella_pagine" origin/gh-pages
find "$cartella_pagine" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -a "$copia_sito"/. "$cartella_pagine"/
git -C "$cartella_pagine" add -A -f .     # -f: *.mp4 è ignorato nel repo, ma il video dell'hero va online
if git -C "$cartella_pagine" diff --cached --quiet; then
  echo "niente da pubblicare: gh-pages è già uguale al sito locale"; exit 0
fi
echo "cambi rispetto a quello che è online:"
git -C "$cartella_pagine" diff --cached --stat | cat
if $solo_prova; then echo "(prova: niente pubblicato)"; exit 0; fi

git -C "$cartella_pagine" commit -q -m "Sito pubblicato il $(date '+%d/%m %H:%M')

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git -C "$cartella_pagine" push -q origin HEAD:refs/heads/gh-pages
echo "pubblicato: https://marcozambrella.github.io/argo/"
