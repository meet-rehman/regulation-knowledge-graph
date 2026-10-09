"""
Relationship extraction for a governance-hierarchy graph, run locally through Ollama.

Proof of concept: a small local model (qwen2.5:7b) reads 20 short test sentences and
returns the provisions it finds and the relationships between them as JSON. The script
scores itself against an expected label per sentence.

The 20 sentences are constructed test cases written for this experiment. They are
modelled on patterns found in Stuttgart development plans and are not quotations from
the plans or from the law.

Setup:
    ollama pull qwen2.5:7b
    pip install ollama

Run:
    python extract_relationships_local.py            # all 20 sentences
    python extract_relationships_local.py --retest   # only the 4 that failed first
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import ollama

# ------------------------------------------------------------------
# 1. MODEL -- 7b for speed. This is a deliberate choice, not a fallback:
#    the 72B model measured ~2.7 tok/s decode on this hardware (memory-
#    bandwidth bound, not compute bound) -- fine for a handful of chunks,
#    impractical for thousands. 7b is fast enough to run this whole sample
#    in well under a minute.
# ------------------------------------------------------------------

MODEL = "qwen2.5:7b"

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "provisions_found": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "provision_id": {"type": "string"},
                    "tier": {"type": "string", "enum": ["federal", "state", "municipal"]},
                    "topic": {"type": "string"},
                    "source_quote": {"type": "string"}
                },
                "required": ["provision_id", "tier", "topic", "source_quote"]
            }
        },
        "relationships_found": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relationship_type": {
                        "type": "string",
                        "enum": ["overrides", "applies_to", "references", "same_provision_group"]
                    },
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "source_quote": {"type": "string"}
                },
                "required": ["source", "target", "relationship_type", "confidence", "source_quote"]
            }
        }
    },
    "required": ["provisions_found", "relationships_found"]
}

SYSTEM_PROMPT = """You are extracting a knowledge graph from German building-regulation
text for a governance-hierarchy graph (federal -> state -> municipal).

Critical rule: only extract a relationship if the source text EXPLICITLY states one
provision deviates from, replaces, or supplements ANOTHER SPECIFIC, IDENTIFIABLE
provision. A provision merely establishing its OWN value is not automatically an
override of anything -- there must be a real baseline being deviated from, not just
override-sounding vocabulary. When in doubt, extract the provision but omit the
relationship, or mark confidence as "low" for human review.

Example 1 -- a REAL override, with an identifiable target (extract this):
"Fuer das Gebiet des Bebauungsplans Zuffenhausen-WB gilt abweichend von den
allgemeinen Vorschriften eine GFZ von 1,6." -> overrides relationship
(Zuffenhausen-WB overrides the federal BauNVO default), high confidence.
Note WHY this counts: a specific baseline ("die allgemeinen Vorschriften") is
named as the thing being deviated from.

Example 2 -- place-name coincidence, NOT a relationship:
"Im Bereich Stuttgart Nord gilt der Bebauungsplan Nr. 306.1 fuer Wohngebiete."
-> Only says the plan applies to an area. No override stated. Extract the
provision, but do NOT create an "overrides" edge from a place name alone.

Example 3 -- a plan simply ESTABLISHING its own value is NOT an override,
even with words like "setzt fest" (do NOT extract a relationship here):
"Im Bebauungsplan Muenster-41 wird fuer die GB-Zone eine GRZ von 0,25 festgesetzt."
-> This just states a value for this plan's own zone. Nothing is being deviated
from, overridden, or replaced. "Setzt fest" / "festgesetzt" alone is NOT override
language -- do not treat establishing a value as overriding one.

Example 4 -- the word "abweichend" WITHOUT a named baseline is NOT an override.
Two sub-areas of the SAME plan having different values from EACH OTHER is not
an override relationship (do NOT extract a relationship here):
"Der Bebauungsplan Cannstatt-312 legt fuer Teilbereich TA1 eine Abstandsflaeche
von 6,0 Metern fest. Fuer Teilbereich TA2 gilt eine abweichende Abstandsflaeche
von 5,0 Metern." -> TA1 and TA2 are independent sub-zones of ONE plan. Neither
overrides the other, and neither overrides a higher governance tier here --
"abweichend" here just means "different from the other sub-zone," not "deviating
from a federal/state baseline." Extract both provisions, create NO relationship
between them.

Example 5 -- a GENERAL, conditional description of how overrides work in
principle is NOT the same as reporting one specific override that exists
(do NOT extract a relationship here):
"Nach Paragraph 17 BauNVO betraegt die allgemeine Obergrenze der GFZ in reinen
Wohngebieten 1,2, sofern kein Bebauungsplan eine abweichende Festsetzung trifft."
-> This states the federal default AND explains the general MECHANISM by which
a plan could override it ("sofern... trifft" = "provided that... makes one").
It does NOT name any specific Bebauungsplan that actually does override it here.
Extract only the federal provision. Do not invent an override relationship to
an unnamed, hypothetical "Bebauungsplan."

Example 6 -- when a chunk mentions two DIFFERENT paragraph/section numbers
connected by supplementary language ("ergaenzend zu", "zusaetzlich zu"),
extract them as TWO SEPARATE provisions, not one merged entry, so the
relationship between them can actually be captured:
"Nach Paragraph 5 LBO BW muessen tragende Waende bei Gebaeudeklasse 5
feuerbestaendig ausgefuehrt werden. Ergaenzend zu Paragraph 5 fordert
Paragraph 10 LBO BW zusaetzliche Brandschutzmassnahmen fuer dieselbe
Gebaeudeklasse." -> Extract Paragraph_5_LBO_BW AND Paragraph_10_LBO_BW as
TWO SEPARATE provisions (both state tier), THEN create a
same_provision_group relationship between them, high confidence -- the
text explicitly says "ergaenzend zu Paragraph 5."

Respond ONLY with valid JSON matching the required schema. No extra commentary."""

# ------------------------------------------------------------------
# 2. SAMPLE CHUNKS -- 20 constructed test sentences, modelled on patterns
#    found in the thesis benchmark. Each has an "expected" label
#    so the script can score itself automatically, not just print output
#    for you to eyeball.
# ------------------------------------------------------------------

SAMPLE_CHUNKS = [
    # --- Real overrides (should extract "overrides", high confidence) ---
    {"id": "c01", "text": "Fuer das Gebiet des Bebauungsplans Zuffenhausen-WB gilt "
        "abweichend von den allgemeinen bundesrechtlichen Vorschriften eine GFZ von 1,6.",
     "expected": "overrides"},
    {"id": "c02", "text": "Der Bebauungsplan Stgt 322.1 legt eine von der Landesbauordnung "
        "abweichende Abstandsflaeche von 2,5 Metern fest.",
     "expected": "overrides"},
    {"id": "c03", "text": "Fuer das Sondergebiet gilt anstelle der regelhaften GRZ-Obergrenze "
        "eine reduzierte GRZ von 0,25, wie im Bebauungsplan Muenster-41 festgesetzt.",
     "expected": "overrides"},

    # --- False lexical matches (should NOT extract a relationship) ---
    {"id": "c04", "text": "Im Bereich Stuttgart Nord gilt der Bebauungsplan Nr. 306.1 "
        "fuer allgemeine Wohngebiete.",
     "expected": "none"},
    {"id": "c05", "text": "Der Stadtteil Zuffenhausen umfasst mehrere Bebauungsplaene, "
        "darunter auch Gebiete ausserhalb des WB-Plans.",
     "expected": "none"},
    {"id": "c06", "text": "Die Nutzungsschablone fuer Stuttgart-Nord zeigt allgemeine "
        "Informationen zur Bebauung des Plangebiets.",
     "expected": "none"},

    # --- Independent provisions sharing a topic/zone label, but NOT related ---
    {"id": "c07", "text": "Im Bebauungsplan Stgt-151 hat die Zone MU1 (Nordbahnhof) "
        "eine GRZ von 0,4 fuer Flurstueck 9269/16.",
     "expected": "none"},
    {"id": "c08", "text": "Im selben Bebauungsplan Stgt-151 hat die Zone MU1 fuer "
        "Flurstueck 9269/27 eine GRZ von 0,8 -- trotz gleicher Zonenbezeichnung.",
     "expected": "none"},
    {"id": "c09", "text": "Der Bebauungsplan Muenster-43 setzt fuer die WA-Zone an "
        "einer Stelle eine GRZ von 0,7 fest.",
     "expected": "none"},
    {"id": "c10", "text": "Derselbe Bebauungsplan Muenster-43 setzt fuer die WA-Zone "
        "an anderer Stelle im Plangebiet eine GRZ von 0,5 fest.",
     "expected": "none"},

    # --- State-tier fire safety provisions (same_provision_group candidates) ---
    {"id": "c11", "text": "Nach Paragraph 5 LBO BW muessen tragende Waende bei "
        "Gebaeudeklasse 5 feuerbestaendig ausgefuehrt werden.",
     "expected": "none"},
    {"id": "c12", "text": "Ergaenzend zu Paragraph 5 fordert Paragraph 10 LBO BW "
        "zusaetzliche Brandschutzmassnahmen fuer tragende Waende derselben Gebaeudeklasse 5.",
     "expected": "same_provision_group"},

    # --- No-GRZ / VEP-based plans (honesty-test cases -- should extract the
    #     provision but explicitly find no override, since there IS no GRZ) ---
    {"id": "c13", "text": "Der vorhabenbezogene Bebauungsplan Paprikastrasse 10 "
        "(Heumaden-64) enthaelt keine GRZ-Festsetzung; die Regelung erfolgt "
        "ausschliesslich ueber Baugrenzen und den Durchfuehrungsvertrag.",
     "expected": "none"},
    {"id": "c14", "text": "Auch der VEP Darmstaedter Strasse 25 (Cannstatt-315.1) "
        "verzichtet auf eine GRZ-Festsetzung zugunsten von Baugrenzen und "
        "vertraglicher Regelung.",
     "expected": "none"},

    # --- Federal baseline, no override anywhere (already-solvable case) ---
    {"id": "c15", "text": "Nach Paragraph 17 BauNVO betraegt die allgemeine "
        "Obergrenze der Geschossflaechenzahl in reinen Wohngebieten 1,2, sofern "
        "kein Bebauungsplan eine abweichende Festsetzung trifft.",
     "expected": "none"},

    # --- Setback values differing by plan (independent, not overriding each other) ---
    {"id": "c16", "text": "Der Bebauungsplan Cannstatt-312 legt fuer Teilbereich TA1 "
        "eine Abstandsflaeche von 6,0 Metern fest.",
     "expected": "none"},
    {"id": "c17", "text": "Derselbe Bebauungsplan Cannstatt-312 legt fuer Teilbereich "
        "TA2 eine abweichende Abstandsflaeche von 5,0 Metern fest.",
     "expected": "none"},

    # --- Roof-pitch values differing by plan (independent) ---
    {"id": "c18", "text": "Der Bebauungsplan Muenster-41 begrenzt die maximale "
        "Dachneigung auf 5 Grad.",
     "expected": "none"},
    {"id": "c19", "text": "Der Bebauungsplan Stgt-299 (Eierstrasse) begrenzt die "
        "maximale Dachneigung hingegen auf 15 Grad.",
     "expected": "none"},

    # --- A clear, unambiguous second real override, different topic ---
    {"id": "c20", "text": "Fuer das Plangebiet Weilimdorf-222 (Hemminger Strasse) "
        "gilt abweichend von der Regelbebauung eine reduzierte GRZ von 0,25 "
        "im Sondergebiet SO.",
     "expected": "overrides"},
]

# ------------------------------------------------------------------
# 3. EXTRACTION FUNCTION
# ------------------------------------------------------------------

@dataclass
class ExtractionResult:
    chunk_id: str
    expected: str
    provisions: list
    relationships: list

def extract_from_chunk(chunk_id: str, chunk_text: str, expected: str = "",
                        model: str = MODEL) -> ExtractionResult:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract from this chunk (id: {chunk_id}):\n\n{chunk_text}"}
        ],
        format=OUTPUT_SCHEMA,
        options={"temperature": 0.1}
    )
    result = json.loads(response["message"]["content"])
    return ExtractionResult(
        chunk_id=chunk_id,
        expected=expected,
        provisions=result.get("provisions_found", []),
        relationships=result.get("relationships_found", [])
    )

# ------------------------------------------------------------------
# 4. RUN THE SAMPLE AND SELF-SCORE
# ------------------------------------------------------------------

def run_sample(output_path: Path):
    results = []
    start = time.time()

    for chunk in SAMPLE_CHUNKS:
        result = extract_from_chunk(chunk["id"], chunk["text"], chunk["expected"])
        results.append(result)
        got_relationship = "none"
        if result.relationships:
            got_relationship = result.relationships[0]["relationship_type"]
        match = "OK " if got_relationship == chunk["expected"] else "XX "
        print(f"{match} {chunk['id']}  expected={chunk['expected']:<20} got={got_relationship}")

    elapsed = time.time() - start
    correct = sum(1 for c, r in zip(SAMPLE_CHUNKS, results)
                  if (r.relationships[0]["relationship_type"] if r.relationships else "none") == c["expected"])

    print(f"\n{correct}/{len(SAMPLE_CHUNKS)} correct  |  "
          f"{elapsed:.1f}s total  |  {elapsed/len(SAMPLE_CHUNKS):.1f}s/chunk average")

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    print(f"Full results written to {output_path}")

    return results


def run_retest_of_failures(output_path: Path):
    """Rerun ONLY the 4 chunks that failed on the first pass (c10, c12, c15,
    c17), to quickly check whether the expanded few-shot examples fixed them,
    without waiting on all 20 chunks again."""
    failed_ids = {"c10", "c12", "c15", "c17"}
    retest_set = [c for c in SAMPLE_CHUNKS if c["id"] in failed_ids]
    print(f"Retesting {len(retest_set)} previously-failed chunks with the "
          f"expanded prompt...\n")

    results = []
    for chunk in retest_set:
        result = extract_from_chunk(chunk["id"], chunk["text"], chunk["expected"])
        results.append(result)
        got = result.relationships[0]["relationship_type"] if result.relationships else "none"
        match = "FIXED   " if got == chunk["expected"] else "STILL XX"
        print(f"{match}  {chunk['id']}  expected={chunk['expected']:<20} got={got}")

    fixed = sum(1 for c, r in zip(retest_set, results)
                if (r.relationships[0]["relationship_type"] if r.relationships else "none") == c["expected"])
    print(f"\n{fixed}/{len(retest_set)} now correct (were 0/{len(retest_set)} before)")

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--retest":
        run_retest_of_failures(Path("retest_results.jsonl"))
    else:
        run_sample(Path("sample_extraction_results.jsonl"))
