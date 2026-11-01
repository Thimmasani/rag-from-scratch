"""
Fixture for Part 9 (multi-hop retrieval). Three independent two-hop fact
chains, each needing two separate documents combined to answer one
question, plus distractor documents that share vocabulary with one hop
or the other but don't complete the chain -- the thing that makes
single-shot retrieval over the whole question unreliable here.
"""

DOCUMENTS = [
    # Chain 1: Project Helios -> Maria Chen -> Thornfield
    "Maria Chen is the lead engineer on Project Helios.",                                    # 0: hop1
    "Maria Chen's team relies on the Thornfield deployment system for releases.",             # 1: hop2
    # Chain 2: Project Orion -> David Osei -> Cascade
    "David Osei is the lead engineer on Project Orion.",                                      # 2: hop1
    "David Osei's team relies on the Cascade monitoring platform for releases.",               # 3: hop2
    # Chain 3: Project Nimbus -> Priya Nair -> Anchor
    "Priya Nair is the lead engineer on Project Nimbus.",                                      # 4: hop1
    "Priya Nair's team relies on the Anchor incident response tool for releases.",              # 5: hop2
    # Distractors: share vocabulary with one hop, but don't complete any chain.
    "Maria Chen also contributes to Project Orion's design reviews.",                          # 6
    "The Thornfield deployment system was first adopted by the infrastructure team in 2024.",  # 7
    "David Osei previously led Project Nimbus before moving to Project Orion.",                # 8
    "The Cascade monitoring platform integrates with most internal dashboards.",                # 9
]

DOC_LABELS = [
    "doc0: hop1, Helios -> Maria Chen",
    "doc1: hop2, Maria Chen -> Thornfield",
    "doc2: hop1, Orion -> David Osei",
    "doc3: hop2, David Osei -> Cascade",
    "doc4: hop1, Nimbus -> Priya Nair",
    "doc5: hop2, Priya Nair -> Anchor",
    "doc6: distractor (Maria Chen + Orion, wrong chain)",
    "doc7: distractor (Thornfield, no person link)",
    "doc8: distractor (David Osei + Nimbus, wrong chain)",
    "doc9: distractor (Cascade, no person link)",
]

QUERIES = [
    "What deployment system does Project Helios's lead engineer's team use?",
    "What monitoring platform does Project Orion's lead engineer's team use?",
    "What incident response tool does Project Nimbus's lead engineer's team use?",
]

# query -> {"hop1_doc": idx, "hop2_doc": idx, "expected_answer_text": substring
#           the final generated answer should contain if the chain was
#           actually assembled correctly}
GROUND_TRUTH_MULTIHOP = {
    "What deployment system does Project Helios's lead engineer's team use?": {
        "hop1_doc": 0,
        "hop2_doc": 1,
        "expected_answer_text": "Thornfield",
    },
    "What monitoring platform does Project Orion's lead engineer's team use?": {
        "hop1_doc": 2,
        "hop2_doc": 3,
        "expected_answer_text": "Cascade",
    },
    "What incident response tool does Project Nimbus's lead engineer's team use?": {
        "hop1_doc": 4,
        "hop2_doc": 5,
        "expected_answer_text": "Anchor",
    },
}
