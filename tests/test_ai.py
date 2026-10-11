import json

from probe.ai import Completion, NullProvider, RecordingProvider, ReplayProvider, paraphrase_claim

CLAIM = (
    "In simulator world systemlab/R2 (seed 1), the data favour hypothesis "
    "sliding_window (posterior 1.0, n=5)."
)


class Fake:
    name = "fake"

    def __init__(self, text):
        self.text = text

    def complete(self, template_id, user):
        return Completion(self.text, self.name, "fake-model", {"output_tokens": 3})


def test_null_provider_keeps_template():
    r = paraphrase_claim(NullProvider(), CLAIM, "L0")
    assert not r["accepted"] and r["text"] == CLAIM and r["epistemic_status"] == "proposed"


def test_causal_or_number_changes_rejected():
    bad = paraphrase_claim(
        Fake("In the simulator world, sliding_window causes it (1.0, n=5, seed 1, R2)"), CLAIM, "L0"
    )
    assert not bad["accepted"] and bad["text"] == CLAIM
    drop = paraphrase_claim(
        Fake("In the simulator world the sliding window fits best."), CLAIM, "L0"
    )
    assert not drop["accepted"]


def test_good_paraphrase_accepted_and_provenance_has_no_key():
    ok = paraphrase_claim(
        Fake(
            "In simulator world systemlab/R2 (seed 1) the sliding_window "
            "explanation fits best: posterior 1.0 after n=5 readings."
        ),
        CLAIM,
        "L0",
    )
    assert ok["accepted"]
    assert "key" not in json.dumps(ok["llm_call"]).lower()


def test_record_then_replay(tmp_path):
    cas = tmp_path / "c.json"
    text = "In simulator world systemlab/R2 (seed 1): sliding_window, posterior 1.0, n=5."
    rec = RecordingProvider(Fake(text), cas)
    first = paraphrase_claim(rec, CLAIM, "L0")
    rep = ReplayProvider(cas, model="fake-model")
    again = paraphrase_claim(rep, CLAIM, "L0")
    assert again["text"] == first["text"] and again["llm_call"]["provider"] == "replay"
