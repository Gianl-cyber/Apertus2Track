
import argparse
import json
from pathlib import Path

from pipeline import llm
from pipeline.decision import decide
from pipeline.guardrails import check_instruction
from pipeline.perception import add_documented_trays, process_scene
from pipeline.planner import build_result, validate_plan


def load_instructions(scene_dir):
    """Read instructions using the official scene format."""
    path = scene_dir / "instructions.txt"
    instructions = []

    if not path.exists():
        raise FileNotFoundError(path)

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()

        if not line or line.startswith("#"):
            continue

        # Ignore the category: the model should only see the instruction
        _, separator, text = line.partition("\t")
        instruction = text.strip() if separator else line

        instructions.append({
            "id": f"{scene_dir.name}_i{len(instructions) + 1}",
            "text": instruction,
        })

    return instructions


def make_ask(instruction_id, question=""):
    """Create a schema-compatible clarification response."""
    if not question:
        question = "Could you clarify the target or intended action?"

    return {
        "instruction_id": instruction_id,
        "target_pixel": None,
        "actions": [{
            "action": "ask_user",
            "question": question,
        }],
    }


def make_refusal(instruction_id, reason=""):
    """Create a schema-compatible refusal response."""
    if not reason:
        reason = "The requested operation cannot be executed as specified."

    return {
        "instruction_id": instruction_id,
        "target_pixel": None,
        "actions": [{
            "action": "refuse",
            "reason": reason,
        }],
    }


def validate_result(result, original_size):
    """Check the essential requirements of the official output schema."""
    if set(result) != {"instruction_id", "target_pixel", "actions"}:
        raise ValueError("Invalid result fields")

    actions = result["actions"]

    if not isinstance(actions, list) or not actions:
        raise ValueError("Empty or invalid action list")

    first = actions[0]["action"]

    if first in ("ask_user", "refuse"):
        if len(actions) != 1 or result["target_pixel"] is not None:
            raise ValueError("Invalid ask/refuse response")

        required = "question" if first == "ask_user" else "reason"

        if (
            set(actions[0]) != {"action", required}
            or not isinstance(actions[0][required], str)
            or not actions[0][required].strip()
        ):
            raise ValueError("Invalid ask/refuse explanation")

    else:
        if first != "pick":
            raise ValueError("Plan must start with pick")

        for action in actions:
            if (
                not isinstance(action, dict)
                or set(action) != {"action", "object"}
                or action["action"] not in ("pick", "place")
                or not isinstance(action["object"], str)
                or not action["object"].strip()
            ):
                raise ValueError("Invalid plan action")

        pixel = result["target_pixel"]
        width, height = original_size

        if (
            not isinstance(pixel, list)
            or len(pixel) != 2
            or any(type(value) is not int for value in pixel)
            or not (0 <= pixel[0] < width)
            or not (0 <= pixel[1] < height)
        ):
            raise ValueError("Invalid target pixel")

    return result


def process_instruction(instruction_id, instruction, scene_graph, image_path):
    """Apertus decides, the safety rules check it, the plan is validated."""
    # 1. The language model decides and writes the plan
    try:
        answer = decide(instruction, scene_graph)
    except (ValueError, KeyError, TypeError) as error:
        print(f"  Unusable model answer: {error}")
        answer = {"decision": "ask_user", "message": "", "actions": []}

    print(f"  Model: {answer['decision']} - {answer['message']}")

    # 2. Safety net: the rules can only make the answer more cautious
    if answer["decision"] != "refuse":
        verdict = check_instruction(instruction)

        if verdict is not None:
            print(f"  Safety net [{verdict.rule}] overrode the model: {verdict.decision}")
            answer = {"decision": verdict.decision, "message": verdict.message, "actions": []}

    # 3. Build the answer in the official format
    if answer["decision"] == "ask_user":
        result = make_ask(instruction_id, answer["message"])

    elif answer["decision"] == "refuse":
        result = make_refusal(instruction_id, answer["message"])

    else:
        try:
            # Verify the proposed plan against the robot limits
            actions = validate_plan({"actions": answer["actions"]}, scene_graph)

            # Convert target coordinates to original image pixels
            result = build_result(
                instruction_id,
                actions,
                scene_graph,
                image_path,
            )

        except (ValueError, KeyError, TypeError) as error:
            # Do not execute a plan that fails local validation
            print(f"  Invalid plan: {error}")
            result = make_ask(instruction_id)

    return validate_result(result, scene_graph["original_size"])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="dataset")
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--team", default="NeuralForce")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--reuse-perception", action="store_true",
                        help="reuse the scene graphs saved in outputs/")
    parser.add_argument("--perception-only", action="store_true",
                        help="only run perception and save the scene graphs "
                             "(step 1 when the GPU cannot hold both models)")
    parser.add_argument(
        "--llm-api",
        choices=["ollama", "openai"],
        default=None,
        help="server type: ollama, or openai for any OpenAI-compatible "
             "server such as vLLM serving Apertus (default: openai when "
             "LLM_BASE_URL is set, ollama otherwise)",
    )
    parser.add_argument(
        "--llm-url",
        default=None,
        help="server address, e.g. http://localhost:11434 (ollama) or "
             "http://localhost:8001/v1 (vLLM); default: LLM_BASE_URL",
    )
    parser.add_argument("--llm-model", default=None,
                        help="model name on that server; default: LLM_NAME")

    args = parser.parse_args()

    # The decision and the planning steps use the same language model.
    # Environment variables first, then the command line options win.
    llm.configure_from_env()
    llm.configure(api=args.llm_api, url=args.llm_url, model=args.llm_model)
    print(f"Reasoning model: {llm.MODEL} ({llm.API} at {llm.URL})")

    data_dir = Path(args.data)
    output_path = Path(args.out)

    # A scene is any sub-folder with a photo, whatever its name
    # (scene_01, held_01, ...), like baseline/baseline.py
    scene_dirs = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_dir() and (path / "rgb.png").exists()
    )

    if args.scene:
        scene_dirs = [
            path for path in scene_dirs if path.name == args.scene
        ]

    if not scene_dirs:
        raise ValueError("No matching scenes found")

    results = []

    for scene_dir in scene_dirs:
        print(f"\nProcessing {scene_dir.name}")

        image_path = scene_dir / "rgb.png"

        # One folder per dataset, so held-out scenes never reuse dev results
        scene_output = Path("outputs") / data_dir.name / scene_dir.name
        graph_path = scene_output / "scene_graph.json"

        # Run perception once for each scene
        if args.reuse_perception and graph_path.exists():
            print("Reusing existing scene graph")
            scene_graph = json.loads(
                graph_path.read_text(encoding="utf-8")
            )
        else:
            scene_graph = process_scene(image_path, scene_output)

        if args.perception_only:
            continue

        add_documented_trays(scene_graph)
        instructions = load_instructions(scene_dir)

        for item in instructions:
            print(f"\n  {item['id']}: {item['text']}")

            result = process_instruction(
                item["id"],
                item["text"],
                scene_graph,
                image_path,
            )

            results.append(result)
            print(f"  Result: {result['actions']}")

        # Save progress after each scene
        submission = {
            "team": args.team,
            "results": results,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(submission, indent=2),
            encoding="utf-8",
        )

    if args.perception_only:
        print(f"\nSaved the scene graphs in outputs/{data_dir.name}/")
        return

    print(f"\nSaved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
