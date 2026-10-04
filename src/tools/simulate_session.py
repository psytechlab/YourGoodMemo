import os
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

from src.llm_client import LLMuser
from src.conversation_manager import ConversationManager
from src.reasoner import RandomReasoner, LLMReasoner, DummyReasoner

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Simulate psychotherapy session")
    parser.add_argument("--reasoner", type=str, default="random", choices=["random", "llm", "dummy"], help="Type of reasoner to use")
    parser.add_argument("--anchors_path", type=str, default="data/anchors.yaml", help="Path to anchors file")
    parser.add_argument("--filler_topics_path", type=str, default="data/filler_topics.yaml", help="Path to filler topics file")
    parser.add_argument("--filler_instruction_path", type=str, default="prompts/filler_instruction.txt", help="Path to filler instruction file")
    parser.add_argument("--user_base_model", type=str, default="mistral-nemo:12b", help="Model name for user LLM")
    parser.add_argument("--turns", type=int, default=8, help="Number of turns (pairs of friend and user utterances)")
    parser.add_argument("--user_prompt_path", type=str, default=Path.cwd()/"prompts/user_prompt.txt", help="Path to user persona prompt")
    parser.add_argument("--character_path", type=str, default=Path.cwd()/"data/characters/александр_лебедев.txt", help="Path to user persona description")
    parser.add_argument("--output_dir", type=str, default=Path.cwd()/"output/dialogues", help="Output directory for dialogues")
    parser.add_argument("--dialogue_id", type=str, default="id_1", help="ID of the dialogue")

    args = parser.parse_args()

    # LLM Config
    base_url = os.getenv("OPENAI_API_BASE_URL")
    if not base_url:
        base_url = "http://localhost:11434/v1"
        print("OPENAI_API_BASE_URL не найден, Ollama по умолчанию")

    auth_token = os.getenv("OPENAI_API_KEY")
    if not auth_token:
        auth_token = "dummy"
        print("OPENAI_API_KEY не найден, заглушка dummy")
    friend_model = os.getenv("OPENAI_MODEL_NAME")

    # 1. Initialize friend
    with open("prompts/friend_prompt.txt", "r", encoding="utf-8") as f:
        friend_system_prompt = f.read()
    friend_user = LLMuser(model_name=friend_model, base_url=base_url, auth_token=auth_token, system_prompt=friend_system_prompt)

    # 2. Initialize user
    with open(args.user_prompt_path, 'r', encoding='utf-8') as f:
        user_persona = f.read()

    with open(args.character_path, 'r', encoding='utf-8') as f:
        character_plist = f.read()

    user_persona = user_persona.format(character_plist)

    user_llm = LLMuser(model_name=args.user_base_model, base_url=base_url, auth_token=auth_token)

    # 3. Initialize Reasoner
    if args.reasoner == "random":
        reasoner = RandomReasoner(args.anchors_path, args.filler_topics_path, args.filler_instruction_path)
    elif args.reasoner == "llm":
        reasoner = LLMReasoner(user_llm, args.anchors_path)
    else:
        reasoner = DummyReasoner()

    with open(args.filler_instruction_path, 'r', encoding='utf-8') as f:
        filler_instruction = f.read()

    # 4. Initialize Managers
    user_manager = ConversationManager(user_llm, user_persona, reasoner, filler_instruction=filler_instruction)
    friend_manager = ConversationManager(friend_user, friend_system_prompt, reasoner, filler_instruction=filler_instruction)

    # 5. Container
    container = {
        "id": args.dialogue_id,
        "meta": {
            "mode": "dialog",
            "author": "александр_лебедев"
        },
        "utterances": []
    }

    # Simulation state
    history = []
    transcript = []
    previous_anchor_id = None

    print(f"Starting simulation for {args.turns} turns (pairs)...")

    for turn in range(1, args.turns + 1):
        block_type, user_directive, friend_directive, anchor_id, topic, length, block_id = reasoner.think(
            previous_anchor_id=previous_anchor_id
        )

        # 2. Генерация ответа друга
        friend_centric_history = []
        for msg in history[-10:]:
            role = "user" if msg["role"] == "user" else "assistant"
            friend_centric_history.append({"role": role, "content": msg["content"]})

        if turn == 1:
            friend_prompt = "The session is starting. Greet your friend and begin the first topic of the chat."
            friend_response = friend_manager.get_response(
                history=[],
                user_message=friend_prompt,
                directive=friend_directive,
                is_filler=(block_type == "filler"),
            )
        else:
            friend_response = friend_manager.get_response(
                history=friend_centric_history,
                user_message=history[-1]["content"],
                directive=friend_directive,
                is_filler=(block_type == "filler"),
            )

        if not friend_response:
            print(f"Turn {turn}: friend failed to generate response.")
            break

        history.append({"role": "assistant", "content": friend_response})
        log_entry = f"Turn {turn}/{args.turns} - friend: {friend_response}\n"
        transcript.append(log_entry)
        print(f"Turn {turn}/{args.turns} - friend: {friend_response}")

        container["utterances"].append({
            "role": "friend",
            "text": friend_response,
            "annotation": {
                "anchor_id": anchor_id,
                "directive": friend_directive
            },
            "meta": {
                "step": turn,
                "block_id": block_id
            }
        })

        # Генерация ответа пользователя
        user_message = history[-1]["content"]

        user_centric_history = []
        for msg in history[-10:]:
            role = "user" if msg["role"] == "assistant" else "assistant"
            user_centric_history.append({"role": role, "content": msg["content"]})

        user_response = user_manager.get_response(
            history=user_centric_history,
            user_message=user_message,
            directive=user_directive,
            is_filler=(block_type == "filler")
        )

        if not user_response:
            print(f"Turn {turn}: user failed to generate response.")
            break

        history.append({"role": "user", "content": user_response})
        log_entry = f"Turn {turn}/{args.turns} - user: {user_response}\n"
        transcript.append(log_entry)
        print(f"Turn {turn}/{args.turns} - user: {user_response}")

        container["utterances"].append({
            "role": "user",
            "text": user_response,
            "annotation": {
                "anchor_id": anchor_id,
                "directive": user_directive
            },
            "meta": {
                "step": turn,
                "block_id": block_id
            }
        })

        previous_anchor_id = anchor_id

    # Container в JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"dialogue_{args.dialogue_id}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(container, f, ensure_ascii=False, indent=2)

    print(f"\nDialogue saved to {output_file}")


    transcript_file = output_dir / f"dialogue_{args.dialogue_id}_transcript.txt"
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write("\n".join(transcript))

    print(f"Transcript saved to {transcript_file}")


if __name__ == "__main__":
    main()