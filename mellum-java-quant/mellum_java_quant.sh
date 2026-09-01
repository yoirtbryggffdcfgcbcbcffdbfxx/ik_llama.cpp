#!/usr/bin/env bash
# ================================================================
# Mellum2-12B-A2.5B-Thinking -> quant "Java/Fabric Edition" (~5.2 GB)
# Recette : experts en IQ3_KS/IQ3_K (quants ik), attention/embeddings
# proteges, imatrix calculee sur TON corpus Java/Fabric 1.21.4.
# A lancer sur ta machine (Lenovo i5-13420H, Linux Mint).
# ================================================================
set -euo pipefail

# ---------- A ADAPTER SI BESOIN ----------
IK_BIN="${HOME}/ik_llama.cpp_git_me/ik_llama.cpp/build/bin"   # tes binaires ik_llama
WORK="${HOME}/mellum-java-quant"                              # dossier de travail
CORPUS="${WORK}/corpus_java_fabric.txt"                       # le corpus fourni
THREADS=8                                                     # 4 P-cores + 4 E-cores
CPUMASK=0xF55                                                 # 1 thread/P-core (0,2,4,6) + E-cores (8-11)

# Source : Q8_0 statique de mradermacher (~12.8 GB). Quasi-BF16 pour
# une cible 3 bits, et 2x moins lourd a telecharger que le BF16.
SRC_REPO="mradermacher/Mellum2-12B-A2.5B-Thinking-GGUF"
SRC_FILE="Mellum2-12B-A2.5B-Thinking.Q8_0.gguf"

mkdir -p "${WORK}"
cd "${WORK}"

# ---------- ETAPE 1 : telecharger la source Q8_0 (~12.8 GB) ----------
if [ ! -f "${SRC_FILE}" ]; then
    echo "=== Telechargement ${SRC_FILE} ==="
    hf download "${SRC_REPO}" "${SRC_FILE}" --local-dir "${WORK}" \
        || huggingface-cli download "${SRC_REPO}" "${SRC_FILE}" --local-dir "${WORK}"
fi

# ---------- ETAPE 2 : imatrix sur ton corpus Java/Fabric ----------
# ~350-450k tokens ; compte 1-2 h sur ton CPU. Le fichier de sortie est
# reutilisable a vie (nouvelles recettes, nouveaux essais).
# Extra local optionnel (sources Minecraft yarn-mappées : ./gradlew genSources)
# Non redistribuable -> jamais commite, mais fusionne automatiquement si present.
if [ -f corpus_extra_local.txt ]; then
    echo "=== Fusion de corpus_extra_local.txt ($(du -h corpus_extra_local.txt | cut -f1)) ==="
    cat "${CORPUS}" corpus_extra_local.txt > corpus_merged.txt
    CORPUS="${WORK}/corpus_merged.txt"
fi

if [ ! -f mellum-java.imatrix ]; then
    echo "=== Calcul de l'imatrix Java/Fabric (long : ~1-2 h) ==="
    taskset ${CPUMASK} "${IK_BIN}/llama-imatrix" \
        -m "${SRC_FILE}" \
        -f "${CORPUS}" \
        -o mellum-java.imatrix \
        --ctx-size 512 \
        --threads ${THREADS}
fi

# ---------- ETAPE 3 : quantification custom ----------
# ~11.1B params dans les experts -> 3.19/3.44 bpw (IQ3_KS / IQ3_K)
# attention -> IQ5_K ; embeddings/output -> Q6_K ; base fallback IQ3_K.
OUT="Mellum2-Thinking-JavaFabric-IQ3K.gguf"
echo "=== Quantification -> ${OUT} ==="
taskset ${CPUMASK} "${IK_BIN}/llama-quantize" \
    --imatrix mellum-java.imatrix \
    --custom-q "token_embd\.weight=q6_K,output\.weight=q6_K,attn_.*=iq5_k,ffn_(up|gate)_exps=iq3_ks,ffn_down_exps=iq3_k" \
    "${SRC_FILE}" "${OUT}" IQ3_K ${THREADS}

ls -lh "${OUT}"

# ---------- ETAPE 4 (optionnel) : verif perplexite sur du Java ----------
# Compare l'ancien IQ4_NL et le nouveau quant sur le meme texte Java.
# Une PPL nouvelle <= PPL IQ4_NL + ~1-2% = mission accomplie (pour 1.5 GB de moins).
OLD_GGUF="${HOME}/.cache/huggingface/hub/models--mradermacher--Mellum2-12B-A2.5B-Thinking-i1-GGUF/snapshots/e0572142f9abf0e23b01296cdfa3c13b53d8fc5c/Mellum2-12B-A2.5B-Thinking.i1-IQ4_NL.gguf"
if [ -f "${OLD_GGUF}" ]; then
    echo "=== Perplexite IQ4_NL (reference) ==="
    taskset ${CPUMASK} "${IK_BIN}/llama-perplexity" -m "${OLD_GGUF}" -f "${CORPUS}" --chunks 40 --threads ${THREADS} 2>&1 | tail -3
fi
echo "=== Perplexite nouveau quant ==="
taskset ${CPUMASK} "${IK_BIN}/llama-perplexity" -m "${OUT}" -f "${CORPUS}" --chunks 40 --threads ${THREADS} 2>&1 | tail -3

echo ""
echo "Termine ! Mets a jour MELLUM_MODEL dans ton lanceur :"
echo "  MELLUM_MODEL=\"${WORK}/${OUT}\""
echo "Astuce : tu peux maintenant supprimer ${SRC_FILE} (12.8 GB) si l'espace manque."
