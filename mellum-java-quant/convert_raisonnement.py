#!/usr/bin/env python3
import re
import sys

TS_USER = re.compile(r"(?:^|\n)(?:User|ser)\s+\d{1,2}:\d{2}(?:\s*[AP]M)?\s*", re.IGNORECASE)
TS_MODEL = re.compile(r"(?:^|\n)(?:Model|Assistant)\s+\d{1,2}:\d{2}(?:\s*[AP]M)?\s*", re.IGNORECASE)
COLLAPSE_PAT = re.compile(r"Collapse to hide model thou(?:ghts)?", re.IGNORECASE)


def clean_artifacts(text: str) -> str:
    text = text.replace("chevron_right", "")
    text = re.sub(r"(?m)^code\s*\nCode\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_reasoning_robust(raw: str):
    raw = raw.replace("\r\n", "\n")
    matches = list(re.finditer(r"ThinkingThoughts", raw))
    if not matches:
        return []

    entries = []
    # Pointeur global marquant la fin du dernier contenu traité
    last_processed_idx = 0

    for i, m in enumerate(matches):
        think_start = m.end()
        seg_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        seg = raw[think_start:seg_end]

        # 1. Déterminer où se termine le raisonnement (think)
        col = COLLAPSE_PAT.search(seg)
        if col:
            think_raw = seg[:col.start()]
            actual_think_end = think_start + col.end()
        else:
            ts_next = re.search(r"(?:Model|User|ser)\s+\d{1,2}:\d{2}", seg)
            if ts_next:
                think_raw = seg[:ts_next.start()]
                actual_think_end = think_start + ts_next.start()
            else:
                think_raw = seg
                actual_think_end = seg_end

        # 2. Le prompt se trouve UNIQUEMENT entre la fin du raisonnement précédent
        # et le début de ce bloc de pensée (m.start())
        pre_zone = raw[last_processed_idx:m.start()]

        # Retirer l'horodatage "Model HH:MM" qui précède immédiatement ThinkingThoughts
        pre_zone = TS_MODEL.sub("\n", pre_zone)

        # S'il y a un tag User/ser, on prend ce qui le suit
        u_matches = list(TS_USER.finditer(pre_zone))
        if u_matches:
            prompt_raw = pre_zone[u_matches[-1].end():]
        else:
            prompt_raw = pre_zone

        prompt = clean_artifacts(prompt_raw)
        think = clean_artifacts(think_raw)

        # Avancer le curseur de lecture pour le tour suivant
        last_processed_idx = actual_think_end

        # On n'enregistre que les paires ayant un prompt réel et distinct
        if think and prompt and prompt != "[Instruction manquante / tour précédent]":
            entries.append({"prompt": prompt, "think": think})

    return entries


def to_corpus(entries):
    out = []
    for e in entries:
        out.append("<|im_start|>user")
        out.append(e["prompt"])
        out.append("<|im_end|>")
        out.append("<|im_start|>assistant")
        out.append("<think>")
        out.append(e["think"])
        out.append("</think>")
        out.append("<|im_end|>\n")
    return "\n".join(out)


def main():
    raw_data = sys.stdin.read() if len(sys.argv) <= 1 else open(sys.argv[1], encoding="utf-8").read()
    results = split_reasoning_robust(raw_data)
    sys.stderr.write(f"# {len(results)} traces propres extraites\n")
    sys.stdout.write(to_corpus(results))


if __name__ == "__main__":
    main()
