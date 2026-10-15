"""Check the plan written by the language model and turn it into a result.

The model proposes the actions (pipeline/decision.py). This module only
verifies them against the robot limits and converts the target to pixels.
"""

import cv2


NON_GRASPABLE = {"grey tray", "black tray", "plate", "saucer"}


def normalize_steps(actions):
    """Keep only the action and the object id of each step.

    Some models write extra fields (e.g. a comment) or call the object field
    "object" instead of "object_id". We keep what we need and ignore the rest.
    """
    steps = []

    for step in actions:
        if not isinstance(step, dict):
            raise ValueError("Invalid action")

        action = step.get("action")
        obj_id = step.get("object_id", step.get("object"))

        if action is None or obj_id is None:
            raise ValueError(f"Incomplete action: {step}")

        steps.append({"action": str(action).lower(), "object_id": str(obj_id)})

    return steps


def is_alternating(actions):
    """True if the steps go pick, place, pick, place, ..."""
    for index, step in enumerate(actions):
        expected = "pick" if index % 2 == 0 else "place"
        if step["action"] != expected:
            return False
    return True


def repair_order(actions):
    """Put grouped steps back in pick/place pairs.

    The gripper holds one object at a time, but the model sometimes writes
    "pick A, pick B, place T, place T". We pair the n-th pick with the n-th
    place. With a single place for several picks, every object goes there.
    Any other plan is returned unchanged and checked later.
    """
    if is_alternating(actions):
        return actions

    picks = [step for step in actions if step["action"] == "pick"]
    places = [step for step in actions if step["action"] == "place"]

    if len(places) == 1 and len(picks) > 1:
        places = places * len(picks)

    if len(picks) != len(places):
        return actions

    repaired = []
    for pick, place in zip(picks, places):
        repaired.append(pick)
        repaired.append(place)

    print("  Repaired the order of the steps into pick/place pairs")
    return repaired


def remove_impossible_picks(actions, objects):
    """Drop picks of objects the gripper cannot hold, with the place that follows.

    The model sometimes adds "pick plate" to an otherwise good plan. Removing
    that step keeps the rest of the plan instead of rejecting all of it.
    """
    kept = []
    skip_next_place = False

    for step in actions:
        if not isinstance(step, dict):
            kept.append(step)
            continue

        action = step.get("action")
        obj_id = step.get("object_id")

        if action == "pick":
            label = objects[obj_id]["label"].lower() if obj_id in objects else ""

            if label in NON_GRASPABLE:
                print(f"  Removed impossible step: pick {label}")
                skip_next_place = True
                continue

            skip_next_place = False

        elif action == "place" and skip_next_place:
            # This place belonged to the removed pick
            skip_next_place = False
            continue

        kept.append(step)

    return kept


def validate_plan(plan, scene_graph):
    # Verify object references and basic gripper constraints
    objects = {obj["id"]: obj for obj in scene_graph["objects"]}

    actions = plan.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("Plan must contain actions")

    actions = normalize_steps(actions)
    actions = repair_order(actions)
    actions = remove_impossible_picks(actions, objects)
    if not actions:
        raise ValueError("Plan only contained impossible picks")

    holding = None

    for step in actions:
        action = step["action"]
        obj_id = step["object_id"]

        if obj_id not in objects:
            raise ValueError(f"Unknown object: {obj_id}")

        label = objects[obj_id]["label"].lower()

        if action == "pick":
            if holding is not None:
                raise ValueError("Gripper already occupied")

            if label in NON_GRASPABLE:
                raise ValueError(f"Cannot pick {label}")

            holding = obj_id

        elif action == "place":
            if holding is None:
                raise ValueError("Nothing to place")

            holding = None

        else:
            raise ValueError(f"Unknown action: {action}")

    if actions[0]["action"] != "pick":
        raise ValueError("Plan must begin with pick")

    return actions


def build_result(instruction_id, actions, scene_graph, original_image):
    # Convert coordinates from the perception image to original pixels
    objects = {obj["id"]: obj for obj in scene_graph["objects"]}

    image = cv2.imread(str(original_image))
    if image is None:
        raise FileNotFoundError(original_image)

    original_height, original_width = image.shape[:2]
    source_width, source_height = scene_graph["image_size"]

    target_id = actions[0]["object_id"]
    x, y = objects[target_id]["pixel"]

    target_pixel = [
        round(x * original_width / source_width),
        round(y * original_height / source_height),
    ]

    if not (
        0 <= target_pixel[0] < original_width
        and 0 <= target_pixel[1] < original_height
    ):
        raise ValueError("Target pixel outside original image")

    # Construct the official action format deterministically
    return {
        "instruction_id": instruction_id,
        "target_pixel": target_pixel,
        "actions": [
            {
                "action": step["action"],
                "object": step["object_id"],
            }
            for step in actions
        ],
    }
