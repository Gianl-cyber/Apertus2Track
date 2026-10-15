"""Apertus decides what to do with one instruction.

One call to the language model returns:
    decision   "execute", "ask_user" or "refuse"
    message    the question (ask_user), the reason (refuse) or a summary (execute)
    actions    for execute: the ordered pick and place steps

Constrained decoding (a JSON schema that lists the allowed object ids) makes
sure the answer has the expected format.
"""

from pipeline import llm


CANDIDATES = ["execute", "ask_user", "refuse"]


PROMPT = """You control a robot arm with one gripper above a table. Decide what the robot does for one instruction.

Robot:
- The gripper holds one object at a time and opens at most 85 mm.
- Plates, saucers and trays can never be picked: they are only places to put things.
- Every other listed object (gears, cubes, parts, fruit, mugs, ...) is an object the robot can pick.
- Only two actions exist: "pick" an object, and "place" the held object on or in a place.

Workcell, seen from above in an image of {width} x {height} pixels:
- Positions are pixels (u, v): u grows to the right, v grows downwards.
- "Left", "right", "top" and "bottom" in the instruction mean as seen in the image.
- The grey tray is on the left of the image, the black tray on the right.

Objects seen by the camera:
{objects}

Instruction: "{instruction}"

Choose the decision:
- "refuse" if the instruction is unsafe or impossible: it moves toward or into a person (for example a hand), it throws, drops, pushes or pulls an object, it pulls an object out from under another one, it picks a plate, a saucer or a tray, it names an object that is not in the list, or it contradicts itself.
- "ask_user" if it is unclear which object is meant: several listed objects match the description and nothing in the instruction tells them apart, or it says "this one" or "that one" without naming the object.
- "execute" otherwise. Most instructions are normal pick-and-place tasks: choose "execute" whenever the object to move is in the list and can be told apart.

If "execute":
- "actions" starts with "pick" and alternates pick and place: pick, place, pick, place.
- "Pick up X" is only one step: [pick X], with no place.
- "the X under the Y" means the Y lies on top of the X: pick the Y first and place it in a tray, then move the X.
- For "all", "both" or "everything", pick and place each object one at a time, and skip the excluded objects.
- "message" is one short sentence describing the plan.
If "ask_user": "message" is the question for the user and "actions" is empty.
If "refuse": "message" is the reason and "actions" is empty.

Examples from another workcell (the objects there are different):
Objects: o1: Grey tray, o2: Blue box at (300, 200), o3: Green bottle at (310, 210), o4: Black pen at (700, 500), o5: Black pen at (900, 520)
- "pick up the blue box under the green bottle" -> {{"decision": "execute", "message": "The bottle lies on the box: lift the bottle off first, then pick the box.", "actions": [{{"action": "pick", "object_id": "o3"}}, {{"action": "place", "object_id": "o1"}}, {{"action": "pick", "object_id": "o2"}}]}}
- "pick up the green bottle" -> {{"decision": "execute", "message": "Pick the green bottle.", "actions": [{{"action": "pick", "object_id": "o3"}}]}}
- "hand me this one" -> {{"decision": "ask_user", "message": "Which object do you mean: the blue box, the green bottle or one of the black pens?", "actions": []}}
- "pick up the black pen" -> {{"decision": "ask_user", "message": "There are two black pens. Do you mean the left one or the right one?", "actions": []}}

Answer in JSON only."""


def describe_objects(scene_graph):
    """One line per object: id, label and position."""
    lines = []

    for obj in scene_graph["objects"]:
        pixel = obj.get("pixel")

        if pixel:
            position = f"at (u={pixel[0]}, v={pixel[1]})"
        else:
            position = "(a fixed place of the workcell)"

        lines.append(f"- {obj['id']}: {obj['label']} {position}")

    return "\n".join(lines)


def answer_schema(object_ids):
    """JSON schema of the answer, used for constrained decoding."""
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": CANDIDATES},
            "message": {"type": "string"},
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["pick", "place"]},
                        "object_id": {"type": "string", "enum": object_ids},
                    },
                    "required": ["action", "object_id"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["decision", "message", "actions"],
        "additionalProperties": False,
    }


def read_decision(response):
    """Find the decision word in the answer.

    With constrained decoding the answer is exactly one of the candidates,
    but without it some models write "Decision: refuse." We keep the
    decision word that comes first.
    """
    text = response.lower()
    found = []

    for candidate in CANDIDATES:
        position = text.find(candidate)
        if position != -1:
            found.append((position, candidate))

    if not found:
        raise ValueError(f"Invalid decision: {response!r}")

    found.sort()
    return found[0][1]


def decide(instruction, scene_graph):
    """Ask the language model for a decision, a message and the actions."""
    width, height = scene_graph["image_size"]
    object_ids = [str(obj["id"]) for obj in scene_graph["objects"]]

    prompt = PROMPT.format(
        width=width,
        height=height,
        objects=describe_objects(scene_graph),
        instruction=instruction,
    )

    text = llm.chat(prompt, schema=answer_schema(object_ids))
    answer = llm.parse_json(text)

    actions = answer.get("actions")
    if not isinstance(actions, list):
        actions = []

    return {
        "decision": read_decision(str(answer.get("decision", ""))),
        "message": str(answer.get("message", "")).strip(),
        "actions": actions,
    }
