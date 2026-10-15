import base64
import json
import math
import os
import urllib.request
from pathlib import Path

import cv2


# Configuration, can be changed with VISION_MODEL and VISION_BASE_URL
DEFAULT_MODEL = "qwen2.5vl:7b"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

IMAGE_PATH = Path("dataset/scene_01/rgb.png")
OUTPUT_DIR = Path("outputs/scene_01")

TARGET_WIDTH = 1200


OBJECT_PROMPT = """
You are a visual grounding system for an industrial robot.

Identify ALL distinct physical objects visible in the image.

Use absolute pixel coordinates of the provided image.
Do not normalize coordinates.
The image origin [0, 0] is the top-left corner.

Known object categories:
- Red gear
- Yellow gear
- Red cube
- Mug
- Mandarin
- Banana
- Plate
- Saucer
- Grey metal part
- White part
- Small white part
- Grey tray
- Black tray

Rules:
1. Detect only objects actually visible in the image.
2. Use known labels when visually appropriate.
3. Unknown objects are allowed.
4. Include trays and containers as separate objects.
5. Exclude the table, background, shadows and reflections.
6. Do not merge distinct objects.
7. Assign a unique ID to each object.
8. Return one point in the middle INSIDE each object's visible surface.
9. Do not invent missing objects.

Return only valid JSON:
{
    "objects": [
        {
            "id": "obj_1",
            "label": "object name",
            "point": [0, 0]
        }
    ]
}

The coordinates above are placeholders, not predictions.
"""


def vision_settings(environ=None):
    """Return the vision model name and the Ollama chat URL.

        VISION_MODEL      model name in Ollama (default: qwen2.5vl:7b)
        VISION_BASE_URL   Ollama address (default: http://localhost:11434,
                          http://ollama:11434 inside Docker Compose)
    """
    if environ is None:
        environ = os.environ

    model = environ.get("VISION_MODEL", "").strip() or DEFAULT_MODEL
    base_url = environ.get("VISION_BASE_URL", "").strip() or DEFAULT_OLLAMA_URL

    return model, base_url.rstrip("/") + "/api/chat"


def query_ollama(image_path, prompt):
    """Send an image and prompt to the local Qwen model."""
    model, chat_url = vision_settings()

    # Encode the image without additional resizing
    encoded = base64.b64encode(
        Path(image_path).read_bytes()
    ).decode("ascii")

    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
        },
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [encoded],
            }
        ],
    }

    request = urllib.request.Request(
        chat_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)

    return json.loads(result["message"]["content"])


def prepare_image(image_path, output_path, target_width=TARGET_WIDTH):
    """Resize the image to dimensions aligned to multiples of 28."""

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(image_path)

    original_height, original_width = image.shape[:2]

    # Find dimensions near the target width (Qwen related VLM)
    new_width = max(28, round(target_width / 28) * 28)
    new_height = max(
        28,
        round(
            (original_height * new_width / original_width) / 28
        ) * 28,
    )

    # Resize only the copy sent to Qwen
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    if not cv2.imwrite(str(output_path), resized):
        raise RuntimeError(f"Cannot save image: {output_path}")

    return {
        "original_size": [original_width, original_height],
        "processed_size": [new_width, new_height],
    }


def query_spatial_relations(image_path, objects):
    """Identify physical relationships using Qwen."""

    prompt = f"""
You are analyzing spatial relationships in a robot workcell.

Detected objects:
{json.dumps(objects, indent=2)}

Allowed relationships:
- inside: an object is physically inside a container.
- occludes: an object visibly blocks part of another object.
- on_top_of: an object is physically resting on another object.

Rules:
1. Use only the provided object IDs.
2. Report only relationships clearly supported by the image.
3. The subject_id is the object being contained or supported.
4. The object_id is the container or supporting object.
5. A tray containing objects is NOT inside those objects.
6. Prefer "inside" for objects located within a tray.
7. Do not use "on_top_of" for objects resting on a tray's bottom.
8. Use "on_top_of" for objects stacked on other objects.
9. Do not infer physical contact from proximity alone.
10. If uncertain, omit the relationship.
11. Do not invent objects or generate robot actions.

Return only valid JSON:
{{
    "relations": [
        {{
            "subject_id": "obj_1",
            "type": "inside",
            "object_id": "obj_2"
        }}
    ]
}}

The example is illustrative, not a prediction.
"""

    return query_ollama(image_path, prompt)


def compute_geometric_relations(objects, physical_relations, margin=30):
    """Compute 2D relationships, excluding containment pairs."""

    relations = []

    # Identify object-container pairs
    inside_pairs = {
        frozenset((relation["subject_id"], relation["object_id"]))
        for relation in physical_relations
        if relation["type"] == "inside"
    }

    for subject in objects:
        for target in objects:
            if subject["id"] == target["id"]:
                continue

            # Exclude comparisons between contained objects and containers
            pair = frozenset((subject["id"], target["id"]))

            if pair in inside_pairs:
                continue

            x1, y1 = subject["pixel"]
            x2, y2 = target["pixel"]

            dx = x1 - x2
            dy = y1 - y2

            if dx < -margin:
                relations.append({
                    "subject_id": subject["id"],
                    "type": "left_of",
                    "object_id": target["id"],
                })
            elif dx > margin:
                relations.append({
                    "subject_id": subject["id"],
                    "type": "right_of",
                    "object_id": target["id"],
                })

            if dy < -margin:
                relations.append({
                    "subject_id": subject["id"],
                    "type": "above_in_image",
                    "object_id": target["id"],
                })
            elif dy > margin:
                relations.append({
                    "subject_id": subject["id"],
                    "type": "below_in_image",
                    "object_id": target["id"],
                })

    return relations


def process_scene(image_path, output_dir):
    """Generate and save a scene graph for any input image."""

    image_path = Path(image_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare image for Qwen
    processed_path = output_dir / "processed.png"

    sizes = prepare_image(image_path, processed_path)

    print(f"Image sizes: {sizes}")

    # Load processed image for visualization
    image = cv2.imread(str(processed_path))

    if image is None:
        raise FileNotFoundError(processed_path)

    height, width = image.shape[:2]

    # Detect objects
    print("Running Qwen perception...")
    prediction = query_ollama(processed_path, OBJECT_PROMPT)

    objects = []

    for obj in prediction.get("objects", []):
        point = obj.get("point")

        # Validate predicted coordinates
        if (
            not isinstance(point, list)
            or len(point) != 2
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in point
            )
        ):
            print(f"Skipping invalid point: {obj}")
            continue

        x = round(point[0])
        y = round(point[1])

        if not (0 <= x < width and 0 <= y < height):
            print(f"Skipping out-of-bounds point: ({x}, {y})")
            continue

        objects.append({
            "id": str(obj["id"]),
            "label": str(obj["label"]),
            "pixel": [x, y],
        })

        # Draw the predicted point
        cv2.circle(
            image,
            (x, y),
            6,
            (0, 0, 255),
            -1,
        )

        # Draw its label
        cv2.putText(
            image,
            str(obj["label"]),
            (x + 10, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Identify physical relationships
    print("Analyzing spatial relationships...")
    relation_result = query_spatial_relations(processed_path, objects)

    valid_ids = {obj["id"] for obj in objects}
    allowed_types = {"inside", "on_top_of", "occludes"}

    physical_relations = []

    for relation in relation_result.get("relations", []):
        if (
            isinstance(relation, dict)
            and relation.get("subject_id") in valid_ids
            and relation.get("object_id") in valid_ids
            and relation.get("subject_id") != relation.get("object_id")
            and relation.get("type") in allowed_types
        ):
            physical_relations.append(relation)

    # Compute geometric relationships
    geometric_relations = compute_geometric_relations(
        objects,
        physical_relations,
    )

    # Build the intermediate scene graph
    scene_graph = {
        "image_size": [width, height],
        "original_size": sizes["original_size"],
        "objects": objects,
        "relations": physical_relations + geometric_relations,
    }

    # Save structured perception results
    json_path = output_dir / "scene_graph.json"

    json_path.write_text(
        json.dumps(scene_graph, indent=2),
        encoding="utf-8",
    )

    # Save annotated image
    visualization_path = output_dir / "visualization.png"

    if not cv2.imwrite(str(visualization_path), image):
        raise RuntimeError(f"Cannot save image: {visualization_path}")

    print(f"Saved scene graph: {json_path}")
    print(f"Saved visualization: {visualization_path}")

    return scene_graph


def add_documented_trays(scene_graph):
    """Add the grey and the black tray when the vision model missed them.

    docs/DATA.md: "Two trays: the grey tray on the left of the image and the
    black tray on the right." The black tray covers the dark right side of
    the table, so the vision model rarely lists it as an object. A tray added
    here has no pixel: it can only be used as a place, never picked.
    """
    labels = [obj["label"].lower().replace("gray", "grey") for obj in scene_graph["objects"]]

    for tray_id, label in (("grey_tray", "Grey tray"), ("black_tray", "Black tray")):
        if label.lower() not in labels:
            scene_graph["objects"].append({"id": tray_id, "label": label, "pixel": None})

    return scene_graph


if __name__ == "__main__":
    # Standalone test on scene_01
    result = process_scene(IMAGE_PATH, OUTPUT_DIR)
    print(json.dumps(result, indent=2))
