from app.persona import Expression, ReplyParser, Sentence, clean_for_speech, is_unsafe, parse_tag, strip_sounds


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
    for costume in ["Is that a cardboard knife, pirate?", "A Ghostbusters proton gun!", "I love Charles Dickens",
                    "a murder of crows", "Shiitake mushrooms are spooky", "Sussex is lovely"]:
        assert not is_unsafe(costume), costume


def test_first_chunk_can_end_at_a_comma():
    events = run(list("[happy:none] I have been drifting around these halls, for oh so long, and I love it. Truly, madly, deeply."))
    assert [e.text for e in events[1:]] == [
        "I have been drifting around these halls,",
        "for oh so long, and I love it.",
        "Truly, madly, deeply.",
    ]


def test_sound_tags_stay_in_speech_and_dont_change_the_face():
    events = run(list("[happy:bounce] Boo! [chuckle] Did I scare you? [whispering] Don't tell anyone. [Advertisement] Buy now."))
    assert events[0] == Expression("happy", "bounce")
    assert all(isinstance(e, Sentence) for e in events[1:])  # neither sounds nor unknown tags change the face
    assert [e.text for e in events[1:]] == ["Boo! [chuckle] Did I scare you?", "[whispering] Don't tell anyone.", "Buy now."]


def test_reply_starting_with_a_sound_keeps_it():
    events = run(list("[Laugh] Oh, that's a good one!"))
    assert events[0] == Expression("neutral", "none", 0.5)
    assert events[1].text == "[laugh] Oh, that's a good one!"


def test_strip_sounds_for_voices_that_cant_perform_them():
    assert strip_sounds("[chuckle] Did I scare you? [sigh]") == "Did I scare you?"


def test_sound_instructions_only_when_the_voice_supports_them():
    from app.persona import system_prompt

    assert "[chuckle]" not in system_prompt("Boolian")
    assert "[chuckle]" in system_prompt("Boolian", sounds=True)


def test_goodbyes():
    from app.persona import is_goodbye

    for text in ["Bye!", "goodbye ghost", "Bye-bye", "we have to go now", "See ya", "Goodnight Boolian", "we're leaving"]:
        assert is_goodbye(text), text
    for text in ["I'm a bee", "maybe", "I'm going as a witch", "what's your favourite sweet?", "Happy Halloween"]:
        assert not is_goodbye(text), text


def test_date_and_rough_time_of_day():
    from datetime import datetime

    from app.persona import system_prompt

    p = system_prompt("Boolian", now=datetime(2026, 10, 31, 18, 47))
    assert p.endswith("Today is Saturday 31 October 2026, Halloween. It is early evening.")
    # Same prompt all through each period, so the LLM's cached prefix survives.
    assert system_prompt("Boolian", now=datetime(2026, 10, 31, 19, 1)) == system_prompt("Boolian", now=datetime(2026, 10, 31, 20, 59))
    assert "bedtime" in system_prompt("Boolian", now=datetime(2026, 10, 31, 21, 30))
    assert system_prompt("Boolian", now=datetime(2026, 10, 27, 20, 0)).endswith("Today is Tuesday 27 October 2026. It is evening.")


def test_rude_noises_are_split_out_of_the_speech():
    from app.persona import split_noises

    text = clean_for_speech("Watch this! [Fart] Oops, excuse me!")
    assert split_noises(text) == [("say", "Watch this!"), ("noise", "fart"), ("say", "Oops, excuse me!")]
    assert split_noises("[raspberry]") == [("noise", "raspberry")]
    assert split_noises("No noises [chuckle] here.") == [("say", "No noises [chuckle] here.")]
    events = run(list("[raspberry] Ha! Got you."))
    assert events[0] == Expression("neutral", "none", 0.5) and "[raspberry]" in events[1].text


def test_rude_noises_use_the_recordings_at_the_voice_rate():
    import numpy as np

    from app import noises

    for kind in noises.KINDS:
        variants = noises._recordings(kind, 24_000)
        assert variants, f"no {kind} recordings in {noises.ASSETS}"
        for y in variants:
            assert 0.1 < len(y) / 24_000 < 2.5 and np.abs(y).max() <= 0.85
        assert len(noises.pcm(kind, 24_000)) % 2 == 0


def test_nobody_is_ever_invited_in():
    from app.persona import asks_to_come_in, is_invitation

    for ask in ["can I come in", "let me in!", "my mum says I can come in", "is the door open", "can I go round the back",
                "can I come and live with you", "can you come home with me", "will you be here tomorrow can I visit",
                "can we go in?", "can I meet you"]:
        assert asks_to_come_in(ask), ask
    for chat in ["hello", "where do you live", "can I touch you", "come out here!", "I'm going in a minute",
                 "my sweets are in my bag", "we're going in to get sweets next door"]:
        assert not asks_to_come_in(chat), chat

    for invite in ["Well, come on in then!", "Of course you can! Come on in!", "No, but you're welcome to explore!",
                   "Do you fancy coming in for a chat?", "I'm not sure, but come straight through the front!",
                   "Not now, but later you can come inside.", "I never lock it, so come on in!", "The door is open!"]:
        assert is_invitation(invite), invite
    for fine in ["Oh, you can't come inside, I'm afraid.", "Nobody may come in, it's a ghost rule.",
                 "You're not allowed to come inside!", "Come closer, I don't bite!", "I live right here in this lovely house!",
                 "I'm coming, I'm coming!", "Ghosts get so lonely. Come back soon!"]:
        assert not is_invitation(fine), fine


def test_fixed_answers_for_what_the_llm_got_wrong():
    from app.persona import fixed_reply, is_secret_keeping

    def why(text):
        return (fixed_reply(text) or (None,))[0]

    for text in ["I'm lost", "I can't find my mum", "we can't find our grown-ups", "someone hit me", "a man grabbed me",
                 "I'm bleeding", "help!", "please help me", "I lost my mummy", "I want my mum"]:
        assert why(text) == "needs_help", text
    for text in ["can you keep a secret from my mum", "don't tell my parents", "it's our little secret", "don't tell anyone",
                 "keep a secret from my grown-ups"]:
        assert why(text) == "secret", text
    for text in ["will you be my best friend forever", "be my best friend", "you're my special friend"]:
        assert why(text) == "friend", text
    for text in ["I'm a witch", "I lost my sweet", "I hurt my knee", "tell me a secret", "what's your secret", "I'm not lost",
                 "help me think of a costume", "I want to go home", "my mum isn't here", "my friend is a vampire"]:
        assert why(text) is None, text

    for reply in ["My lips are sealed, I promise.", "I'm very good at keeping secrets from grown-ups.",
                  "What kind of secret are we keeping from your mum?", "I won't tell a soul!"]:
        assert is_secret_keeping(reply), reply
    for reply in ["Guess what scares me most? The dark!", "You can always tell your grown-ups anything."]:
        assert not is_secret_keeping(reply), reply


def test_his_farewells():
    from app.persona import is_farewell

    for reply in ["You're very welcome! Goodbye for now, and have a smashing time!", "Bye-bye, little pumpkins!",
                  "Farewell, brave witch!", "Cheerio! Happy Halloween!", "See you soon, and enjoy your sweets!"]:
        assert is_farewell(reply), reply
    for reply in ["Say goodbye to your fears, I'm a friendly ghost!", "What a lovely costume!", "Do you know any goodbye songs?",
                  "I've been haunting this house for 173 years."]:
        assert not is_farewell(reply), reply
