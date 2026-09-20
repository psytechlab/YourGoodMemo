import os
import argparse
from datetime import datetime
from dotenv import load_dotenv, parser
from pathlib import Path

from src.llm_client import LLMuser
from src.conversation_manager import ConversationManager
from src.reasoner import RandomReasoner, LLMReasoner, DummyReasoner

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Simulate psychotherapy session")
    parser.add_argument("--reasoner", type=str, default="random", choices=["random", "llm", "dummy"], help="Type of reasoner to use")
    parser.add_argument("--anchors_path", type=str, default="data/anchors.yaml", help="Path to anchors file")       
    parser.add_argument("--user_base_model", type=str, default="mistral-nemo:12b", help="Model name for user LLM")
    parser.add_argument("--turns", type=int, default=8, help="Number of turns (pairs of friend and user utterances) in the session")
    parser.add_argument("--user_prompt_path", type=str, default=Path.cwd()/"prompts/user_prompt.txt", help="Path to user persona prompt")
    parser.add_argument("--character_path", type=str, default=Path.cwd()/"data/characters/александр_лебедев.txt", help="Path to user persona description")
    parser.add_argument("--output_file", type=str, default=Path.cwd()/"session_log.txt", help="Output file for the session log")
    
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

    if args.reasoner == "random":
        reasoner = RandomReasoner(args.anchors_path)
    elif args.reasoner == "llm":
        reasoner = LLMReasoner(user_llm, args.anchors_path)
    else:
        reasoner = DummyReasoner()

    user_manager = ConversationManager(user_llm, user_persona, reasoner)

    # Simulation
    history = []
    transcript = []
    anchor_log = []

    # Header for the log
    header = [
        f"Simulation Parameters",
        f"--------------------",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {friend_model}",
        f"user Persona: {args.user_prompt_path}",
        f"Reasoner: {args.reasoner}",
        f"Planned Turns (Pairs): {args.turns}",
        f"--------------------\n"
    ]
    # transcript.extend(header)

    print(f"Starting simulation for {args.turns} turns (pairs)...")

    for turn in range(1, args.turns + 1):
        # директивы дляя обоих
        history_for_reasoner = []
        for msg in history:
            history_for_reasoner.append({"role": msg["role"], "content": msg["content"]})
        
        user_directive, friend_directive, anchor_id = reasoner.think(history=history_for_reasoner)
        
        # генерация ответа друга
        if turn == 1:
            friend_prompt = "The session is starting. Greet the user and begin the first session."
            friend_messages = [
                {"role": "system", "content": friend_system_prompt},
                {"role": "user", "content": friend_prompt + f"| {friend_directive}"}
            ]
            friend_response = friend_user.generate(friend_messages)
        else:
            friend_history = []
            for msg in history:
                role = "assistant" if msg["role"] == "assistant" else "user"
                friend_history.append({"role": role, "content": msg["content"]})
            
            friend_messages = [
                {"role": "system", "content": friend_system_prompt}
            ] + friend_history + [
                {"role": "user", "content": f"Продолжи разговор.| {friend_directive}"}
            ]
            friend_response = friend_user.generate(friend_messages)

        if not friend_response:
            print(f"Turn {turn}: friend failed to generate response.")
            break

        history.append({"role": "assistant", "content": friend_response})
        log_entry = f"Turn {turn}/{args.turns} - friend: {friend_response}\n"
        transcript.append(log_entry)
        print(f"Turn {turn}/{args.turns} - friend: {friend_response}")

        # ответ пользователя
        user_message = history[-1]["content"]
        
        user_centric_history = []
        for msg in history:
            role = "user" if msg["role"] == "assistant" else "assistant"
            user_centric_history.append({"role": role, "content": msg["content"]})
        
        # Передаём директиву пользователя в get_response
        user_response = user_manager.get_response(
            history=user_centric_history,
            user_message=user_message,
            directive=user_directive 
        )
        
        if not user_response:
            print(f"Turn {turn}: user failed to generate response.")
            break

        history.append({"role": "user", "content": user_response})
        log_entry = f"Turn {turn}/{args.turns} - user: {user_response}\n"
        transcript.append(log_entry)
        print(f"Turn {turn}/{args.turns} - user: {user_response}")
        
        # сохранение разметки
        anchor_log.append({
            "turn": turn,
            "anchor_id": anchor_id,
            "user_directive": user_directive,
            "friend_directive": friend_directive,
            "user_response": user_response,
            "friend_response": friend_response
        })

        

    # Write to file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(transcript))

    print(f"\nSimulation complete. Log written to {args.output_file}")

if __name__ == "__main__":
    main()
