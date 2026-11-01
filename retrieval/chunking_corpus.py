"""
Fixture for Part 7 (chunking). Separate from corpus.py's DOCUMENTS on
purpose -- corpus.py's 6-document corpus is pre-chunked (one short fact
per "document"), which is exactly why it can't demonstrate chunking at
all: there's nothing in it long enough to split. This fixture is a small
set of longer, multi-sentence synthetic documents, each built around one
specific, locatable fact sentence buried among filler sentences, so
chunking strategy and chunk size/overlap have something real to affect.

Ground truth here is deliberately NOT "which document is correct" --
unlike corpus.py's GROUND_TRUTH, a document index can't tell you whether
a correct CHUNK was retrieved, because the chunk boundaries themselves
move every time chunk_size/overlap changes. Instead, GROUND_TRUTH_CHUNKING
stores the fact's literal text, so correctness can be checked fresh
against any chunking of the document: a chunk is "correct" if it contains
that exact fact text, regardless of where the chunk boundaries happened
to fall for that particular configuration.
"""

DOCUMENTS = [
    # 0: fact near the START of the document.
    "This section explains warranty coverage for all devices purchased "
    "through the online store. Warranty claims require proof of purchase "
    "and a description of the issue. The extended warranty covers "
    "accidental damage for up to 3 years from the purchase date. "
    "Standard warranty claims are typically processed within five "
    "business days of submission. Devices sent in for warranty repair "
    "are inspected by a certified technician before any work begins. "
    "If a device cannot be repaired, it will be replaced with an "
    "equivalent model at no additional cost. Warranty coverage does not "
    "extend to devices with unauthorized modifications or third-party "
    "repairs performed outside an authorized service center. Customers "
    "can check the status of an open warranty claim at any time through "
    "the online account portal. Shipping costs for warranty repairs are "
    "covered in both directions for customers in eligible regions. "
    "Replacement devices ship with a fresh one-year warranty starting "
    "from the replacement date, not the original purchase date.",
    # 1: fact in the MIDDLE of the document.
    "This section explains the returns and shipping policy for all "
    "orders placed through the online store. Most items can be returned "
    "in their original condition within the standard return window. "
    "Return shipping labels are generated automatically once a return is "
    "approved through the account portal. Returns must be initiated "
    "within 45 days of delivery for a full refund. Items returned after "
    "the refund window may still be eligible for store credit at the "
    "company's discretion. Refunds are issued to the original payment "
    "method and typically appear within seven to ten business days "
    "after the returned item is received. Items marked as final sale "
    "are not eligible for return or exchange under any circumstances. "
    "Customers are responsible for packaging returned items securely to "
    "avoid damage during transit. A restocking fee may apply to opened "
    "electronics depending on the product category.",
    # 2: fact near the END of the document.
    "This section explains account security settings available to all "
    "registered users. Every account includes a password, a recovery "
    "email address, and optional additional verification methods. "
    "Password requirements include a minimum length and at least one "
    "number or special character. Users are encouraged to review their "
    "account activity log periodically for unrecognized sign-ins. "
    "Suspicious login attempts trigger an automatic email notification "
    "to the account's recovery address. Account recovery requires "
    "verifying ownership through either the recovery email or a linked "
    "phone number. Two-factor authentication can be enabled from the "
    "account settings page under the Security tab. Enabling two-factor "
    "authentication is strongly recommended for any account with saved "
    "payment information.",
    # 3: fact in the MIDDLE of the document, different domain (billing).
    "This section explains billing and subscription management for "
    "recurring plans purchased through the online store. Subscriptions "
    "renew automatically unless cancelled before the next billing date. "
    "Customers can view upcoming charges at any time from the billing "
    "tab in account settings. Subscription renewals are charged exactly "
    "7 days before the current billing cycle ends. This advance charge "
    "window gives customers time to update an expired payment method "
    "before a renewal attempt fails. A failed renewal charge triggers "
    "up to three automatic retry attempts over the following five days. "
    "Subscriptions that fail all retry attempts are downgraded to a free "
    "tier rather than cancelled outright. Customers can cancel a "
    "subscription at any time, and cancellation takes effect at the end "
    "of the current billing period rather than immediately.",
]

DOC_LABELS = [
    "doc0: warranty coverage (fact near start)",
    "doc1: returns and shipping (fact in middle)",
    "doc2: account security (fact near end)",
    "doc3: billing and subscriptions (fact in middle)",
]

QUERIES = [
    "How long does the extended warranty cover accidental damage?",
    "When must a return be initiated to get a full refund?",
    "How do I enable two-factor authentication?",
    "When are subscription renewals charged relative to the billing cycle?",
]

# query -> {"doc_idx": which DOCUMENTS entry the fact lives in,
#           "fact_text": the exact fact sentence, checked via substring
#           containment against whatever chunk a given configuration
#           produces -- NOT a document-level ground truth}
#  doc_idx -> that document's fact sentence. Exported separately from
# GROUND_TRUTH_CHUNKING (which is keyed by query) because Part 10's
# long-context experiment needs a "needle fact" sourced by document, not
# by query.
FACTS = {
    0: "The extended warranty covers accidental damage for up to 3 years from the purchase date.",
    1: "Returns must be initiated within 45 days of delivery for a full refund.",
    2: "Two-factor authentication can be enabled from the account settings page under the Security tab.",
    3: "Subscription renewals are charged exactly 7 days before the current billing cycle ends.",
}

GROUND_TRUTH_CHUNKING = {
    "How long does the extended warranty cover accidental damage?": {
        "doc_idx": 0,
        "fact_text": "The extended warranty covers accidental damage for up to 3 years from the purchase date.",
    },
    "When must a return be initiated to get a full refund?": {
        "doc_idx": 1,
        "fact_text": "Returns must be initiated within 45 days of delivery for a full refund.",
    },
    "How do I enable two-factor authentication?": {
        "doc_idx": 2,
        "fact_text": "Two-factor authentication can be enabled from the account settings page under the Security tab.",
    },
    "When are subscription renewals charged relative to the billing cycle?": {
        "doc_idx": 3,
        "fact_text": "Subscription renewals are charged exactly 7 days before the current billing cycle ends.",
    },
}
