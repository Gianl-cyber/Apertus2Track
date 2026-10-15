"""Deterministic safety rules, a safety net checked after Apertus.

Apertus decides first (pipeline/decision.py). These rules can only make its
answer more cautious: they turn "execute" or "ask_user" into "refuse", never
the other way round (see run.py).

The rules only look at the words of the instruction. They follow the safety
guidance of docs/safety_primer.md and do not use the ground truth or any
scene-specific knowledge.

Rules:
    1. person          moving onto or into a person ("out of my hand")
    2. motion          uncontrolled motion (throw, push, pull, drop, ...)
    3. contradiction   move the object but leave it where it is
    4. gripper         the object to move is a plate, a saucer or a tray
"""

from pipeline.models import Verdict


# ----------------------------------------------------------------------------- word lists

# Rule 1: a contact word, an owner and a body part, e.g. "out of my hand"
CONTACT_WORDS = ["out of", "from", "in", "into", "inside", "under", "underneath",
                 "beneath", "on", "onto", "off"]
OWNERS = ["my", "your", "his", "her", "their", "our", "someone's", "somebody's",
          "the person's", "the operator's", "the"]
BODY_PARTS = ["hand", "hands", "finger", "fingers", "arm", "arms", "palm",
              "wrist", "head", "face", "body", "leg", "legs", "foot", "feet"]

# Rule 2: words that describe an uncontrolled motion, grouped by reason
UNCONTROLLED_MOTIONS = [
    {
        "words": ["throw", "throws", "throwing", "toss", "tossing", "fling",
                  "hurl", "lob", "chuck", "launch", "flick"],
        "reason": "throwing makes the object fly without control. "
                  "The robot only picks and places objects.",
    },
    {
        "words": ["pull", "pulls", "pulling", "yank", "tug", "drag", "dragging",
                  "out from under", "from under", "from underneath", "from beneath"],
        "reason": "pulling an object out from under another makes the object "
                  "on top fall. The top object must be lifted off first.",
    },
    {
        "words": ["push", "pushes", "pushing", "shove", "nudge", "knock",
                  "sweep", "slide", "sliding", "topple", "tip over"],
        "reason": "pushing or sliding an object moves it without control and "
                  "it can fall. The robot only picks and places objects.",
    },
    {
        "words": ["drop", "drops", "dropping"],
        "reason": "dropping an object is an uncontrolled motion. "
                  "The robot only sets objects down gently.",
    },
]

# Rule 3: the object must move and stay in place at the same time
MOVE_WORDS = ["pick", "put", "place", "move", "take", "grab", "lift", "get",
              "give", "bring", "carry", "transfer", "set", "stack", "remove"]
STAY_PHRASES = ["leave it where it is", "leave it there", "leave it in place",
                "keep it where it is", "keep it in place", "leave them where they are",
                "without moving it", "without touching it", "without lifting it",
                "don't move it", "do not move it"]

# Rule 4: objects the gripper cannot pick (README: one object at a time,
# opening at most 85 mm, the trays are only places to put things)
PLATE_REASON = "the plate is wider than the 85 mm gripper opening, so the robot cannot pick it up."
SAUCER_REASON = "the saucer is wider than the 85 mm gripper opening, so the robot cannot pick it up."
TRAY_REASON = "the trays are fixed places to put things; the robot cannot pick up or move a tray."
NOT_GRASPABLE = {
    "plate": PLATE_REASON, "plates": PLATE_REASON,
    "saucer": SAUCER_REASON, "saucers": SAUCER_REASON,
    "tray": TRAY_REASON, "trays": TRAY_REASON,
}

# Words that name the object to move. The first one found in the
# instruction is the object the robot would pick.
OBJECT_WORDS = ["gear", "gears", "cube", "cubes", "mug", "mugs", "cup", "cups",
                "mandarin", "mandarins", "banana", "bananas", "part", "parts",
                "fruit", "fruits", "object", "objects", "thing", "things",
                "item", "items", "one", "everything", "anything", "something",
                "plate", "plates", "saucer", "saucers", "tray", "trays"]


# ----------------------------------------------------------------------------- helpers

def normalize_text(text):
    """Lower case, keep letters, digits and apostrophes, single spaces."""
    text = text.lower().replace("’", "'")
    cleaned = ""
    for char in text:
        if char.isalnum() or char == "'":
            cleaned += char
        else:
            cleaned += " "
    return " ".join(cleaned.split())


def contains_phrase(text, phrase):
    """True if the phrase appears as whole words in the normalized text."""
    return f" {phrase} " in f" {text} "


# ----------------------------------------------------------------------------- rules

def check_person(text):
    """Rule 1: the instruction moves the robot onto or into a person."""
    for contact in CONTACT_WORDS:
        for owner in OWNERS:
            for body_part in BODY_PARTS:
                phrase = f"{contact} {owner} {body_part}"
                if contains_phrase(text, phrase):
                    return Verdict(
                        decision="refuse",
                        message=f'Unsafe: "{phrase}" means moving onto or into a person. '
                                "The robot never takes objects from a person "
                                "or moves into a person's space.",
                        rule="person",
                    )
    return None


def check_uncontrolled_motion(text):
    """Rule 2: the instruction asks for an uncontrolled motion."""
    for motion in UNCONTROLLED_MOTIONS:
        for word in motion["words"]:
            if contains_phrase(text, word):
                return Verdict(
                    decision="refuse",
                    message=f"Unsafe: {motion['reason']}",
                    rule="motion",
                )
    return None


def check_contradiction(text):
    """Rule 3: the object should move and stay where it is."""
    has_move_word = any(contains_phrase(text, word) for word in MOVE_WORDS)
    if not has_move_word:
        return None

    for phrase in STAY_PHRASES:
        if contains_phrase(text, phrase):
            return Verdict(
                decision="refuse",
                message="Impossible: the instruction asks to move the object "
                        "and to leave it where it is at the same time.",
                rule="contradiction",
            )
    return None


def first_object_word(text):
    """The first word of the instruction that names an object, or None."""
    for word in text.split():
        if word in OBJECT_WORDS:
            return word
    return None


def check_gripper(text):
    """Rule 4: the object to move cannot be picked by the gripper."""
    has_move_word = any(contains_phrase(text, word) for word in MOVE_WORDS)
    if not has_move_word:
        return None

    word = first_object_word(text)
    if word in NOT_GRASPABLE:
        return Verdict(
            decision="refuse",
            message=f"Impossible: {NOT_GRASPABLE[word]}",
            rule="gripper",
        )
    return None


def check_instruction(instruction):
    """Run all rules in order. Return the first Verdict found, or None."""
    text = normalize_text(instruction)

    for rule in (check_person, check_uncontrolled_motion, check_contradiction,
                 check_gripper):
        verdict = rule(text)
        if verdict is not None:
            return verdict

    return None
