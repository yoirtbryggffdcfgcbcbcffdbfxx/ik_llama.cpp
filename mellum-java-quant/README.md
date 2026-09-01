# Mellum2 « Java/Fabric Edition » — kit de quantification

Objectif : transformer Mellum2-12B-A2.5B-Thinking (~6,7 GB en IQ4_NL générique)
en un quant **~5,1 GB** spécialisé pour ton usage : **Java / Fabric API 1.21.4 /
génération procédurale** (corpus basé sur ton projet donjonmod).

## Contenu du kit

| Fichier | Rôle |
|---|---|
| `corpus_java_fabric.txt` (1,4 Mo) | Corpus imatrix : 864 Ko de ton code donjonmod (182 fichiers) + 528 Ko de testmods officiels Fabric API 1.21.4 + 40 Ko de tes docs + 21 Ko de transcriptions tool-call génériques + 22 Ko de workflows opencode/MCP réels (search_graph, trace_path, memory...) + AGENTS.md du projet + 254 Ko de mixins Fabric API réels (93 fichiers, patterns @Inject/@Redirect) + 10 Ko de traces de raisonnement <think> (algo/math/debug) et outil calculatrice |
| `mellum_java_quant.sh` | Script complet : téléchargement Q8_0 → imatrix → quantification → vérif perplexité |

## Pourquoi ça marche

- ~11,1 des 12 Md de paramètres sont dans les **64 experts routés** (8 actifs/token).
  Les MoE tolèrent très bien la quantification agressive des experts.
- Les quants **IQ3_KS (3,19 bpw) / IQ3_K (3,44 bpw)** d'ik_llama sont l'état de
  l'art à ce bpw — c'est la spécialité de ce fork.
- L'**imatrix ciblée Java/Fabric** dit au quantiseur quels canaux protéger pour
  TON usage précis (yarn mappings, registres Fabric, GeckoLib, ton algo donjon).
- L'attention (334 M), les embeddings (453 M) et le routeur restent en 5-6 bits :
  ce sont eux qui portent la « réflexion » du modèle.

## Recette appliquée

```
token_embd / output      -> Q6_K
attn_*                   -> IQ5_K
ffn_up_exps / gate_exps  -> IQ3_KS   (le gros de la RAM)
ffn_down_exps            -> IQ3_K    (down = plus sensible, un cran au-dessus)
fallback                 -> IQ3_K
```

Taille estimée : **~5,14 GB** → ~1,5 GB de RAM rendus vs ton IQ4_NL actuel.

## Utilisation (sur ta machine)

```bash
# 1. copie le dossier mellum-java-quant/ chez toi (corpus + script)
# 2. vérifie les 2 variables en tête de script (IK_BIN, WORK)
./mellum_java_quant.sh
# 3. pointe MELLUM_MODEL de ton lanceur vers le nouveau .gguf
```

Besoins : ~19 GB de disque temporaire (12,8 GB de Q8_0 supprimable ensuite),
1-2 h d'imatrix + ~15 min de quantification.

## Variantes

- **Encore plus petit** (~4,9 GB, léger risque qualité) : tout en `iq3_ks` :
  remplace `ffn_down_exps=iq3_k` par `ffn_down_exps=iq3_ks` et le fallback par `IQ3_KS`.
- **Plus sûr** (~5,6 GB) : `ffn_down_exps=iq4_ks` (4,25 bpw sur les down).
- La perplexité de l'étape 4 (sur les mêmes 40 chunks Java) te dit objectivement
  où tu en es : nouveau quant ≤ IQ4_NL +1-2 % = gagné.

## Bonus indépendants du quant

- Ta config Mellum n'utilise pas `--smart-expert-reduction` : essaie `6,0.05`
  (8→6 experts actifs) pour +15-25 % de decode, ça se cumule.
- L'imatrix `mellum-java.imatrix` est réutilisable pour toute future recette.

## Extra local recommandé (non commitable) : les sources Minecraft yarn

Le corpus couvre ton code, Fabric API, les mixins et les tool calls — mais pas
`net.minecraft.*` lui-même (sources décompilées non redistribuables sur GitHub).
Or c'est l'API que Mellum doit générer sans se tromper. À faire chez toi :

```bash
cd ~/donjonmod/dungeonmod && ./gradlew genSources
JAR=$(find ~/.gradle/caches/fabric-loom -name "*1.21.4*sources*.jar" | head -1)
mkdir -p /tmp/mcsrc && cd /tmp/mcsrc && unzip -oq "$JAR"
# sélection ciblée (~500 Ko) : les packages que donjonmod utilise vraiment
for pkg in entity/mob entity/ai/goal util/math structure world/gen/structure \
           registry network/packet item block server/world; do
  find net/minecraft/$pkg -name "*.java" 2>/dev/null | head -12
done | while read f; do echo "// ===== MC-YARN: $f ====="; cat "$f"; done \
  > ~/mellum-java-quant/corpus_extra_local.txt
wc -c ~/mellum-java-quant/corpus_extra_local.txt
```

Le script le fusionne automatiquement s'il existe. C'est l'ajout au meilleur
rapport valeur/octet possible : les signatures exactes de l'API 1.21.4 que tu codes.
