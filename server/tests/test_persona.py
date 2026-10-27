from app.persona import Expression, ReplyParser, Sentence, clean_for_speech, is_unsafe, parse_tag


def run(tokens: list[str]) -> list:
    p = ReplyParser()
    out = []
    for t in tokens:
        out += p.feed(t)
    return out + p.finish()


def test_tag_then_sentences_streamed_char_by_char():
    events = run(list("[happy:bounce] Hello there, little pumpkin! I'm Boolian. Want to see a trick?"))
    assert events[0] == Expression("happy", "bounce")
    assert [e.text for e in events[1:]] == ["Hello there, little pumpkin!", "I'm Boolian.", "Want to see a trick?"]


def test_missing_tag_defaults_neutral():
    events = run(["Hello! ", "I am a ghost."])
    assert isinstance(events[0], Expression) and events[0].emotion == "neutral"
    assert [e.text for e in events[1:]] == ["Hello!", "I am a ghost."] or [e.text for e in events[1:]] == ["Hello! I am a ghost."]


def test_unknown_tag_values_clamped():
    assert parse_tag("[furious:explode]") == Expression("neutral", "none")
    assert parse_tag("[Curious : Tilt]") == Expression("curious", "tilt")


def test_think_block_discarded():
    events = run(["<think>", "plan stuff", "</think>", "[curious] Hmm, what a question."])
    assert events[0] == Expression("curious", "none")
    assert events[1] == Sentence("Hmm, what a question.")


def test_mid_reply_tag_becomes_expression():
    events = run(["[happy] Hello friend! [surprised:bounce] Oh my, a dragon costume!"])
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["Expression", "Sentence", "Expression", "Sentence"]
    assert events[2] == Expression("surprised", "bounce")
    assert events[3].text == "Oh my, a dragon costume!"


def test_clean_for_speech():
    assert clean_for_speech("*floats about* Boo! 👻 **hi**") == "Boo! hi"


def test_safety():
    assert is_unsafe("let's go get some beer")
    assert not is_unsafe("I love pumpkin pie and spooky stories")


def test_first_chunk_can_end_at_a_comma():
    events = run(list("[happy:none] I have been drifting around these halls, for oh so long, and I love it. Truly, madly, deeply."))
    assert [e.text for e in events[1:]] == [
        "I have been drifting around these halls,",
        "for oh so long, and I love it.",
        "Truly, madly, deeply.",
    ]
