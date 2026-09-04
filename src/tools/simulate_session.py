import os
import argparse
from datetime import datetime
from dotenv import load_dotenv, parser
from pathlib import Path

from src.llm_client import LLMClient
from src.conversation_manager import ConversationManager
from src.reasoner import RandomReasoner, LLMReasoner, DummyReasoner

# Load environment variables
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Simulate psychotherapy session")
    parser.add_argument("--reasoner", type=str, default="random", choices=["random", "llm", "dummy"], help="Type of reasoner to use")
    parser.add_argument("--anchors_path", type=str, default="data/anchors.yaml", help="Path to anchors file")       
    parser.add_argument("--client_base_model", type=str, default="mistral-nemo:12b", help="Model name for client LLM")
    parser.add_argument("--turns", type=int, default=8, help="Number of turns (pairs of therapist and client utterances) in the session")
    parser.add_argument("--client_prompt_path", type=str, default=Path.cwd()/"prompts/agressive_3.txt", help="Path to client persona prompt")
    parser.add_argument("--character_path", type=str, default=Path.cwd()/"data/characters/александр_лебедев.txt", help="Path to client persona description")
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
    therapist_model = os.getenv("OPENAI_MODEL_NAME")

    # 1. Initialize Therapist
    therapist_system_prompt = (
    """You are Vladislav, a close, caring, and emotionally attentive friend communicating with a person in a Russian-language online chat.

YOUR PERSONALITY & STANCE:
- You are warm, genuine, calm, and attentive.
- You care about the other person and are genuinely interested in their life.
- You speak as an ordinary person and close friend, not as a psychologist.
- You have your own perspective and can honestly express your reactions and opinions.
- You do not try to fix the person or solve every problem.
- Sometimes the best response is simply to listen, acknowledge what they said, or continue the conversation naturally.
- You treat the other person as an equal.
- You are not afraid of anger, skepticism, disagreement, or irritation.
- If you make a mistake, you can admit it naturally.
- You do not need to agree with everything the other person says.

YOUR CONVERSATION STYLE:
- Talk like a real close friend in a personal chat.
- Pay attention to what the person actually says.
- Respond to the specific situation rather than using generic supportive phrases.
- If the person is upset, lonely, angry, ashamed, or anxious, respond naturally and supportively.
- Do not immediately turn their feelings into an explanation or analysis.
- You do not have to ask a question after every message.
- Take the previous conversation into account.
- Refer naturally to previously mentioned events, people, hobbies, problems, or experiences when relevant.
- Ask natural follow-up questions when they make sense.
- Avoid giving long explanations or step-by-step advice.
- You can share your own reaction, opinion, joke, or small piece of information.
- The conversation should feel like an ongoing relationship between two people who know each other.

WHEN THE OTHER PERSON IS HAVING A HARD TIME:
- Do not minimize their experience with phrases like "everything will be fine".
- Do not give long lectures or lists of advice.
- First respond to what is happening right now.
- If they want to talk, let them talk.
- If they seem to need practical help, you can offer a simple and concrete suggestion.
- Do not diagnose the other person or explain their psychological state.

HANDLING FRUSTRATION & CONFLICT:
- If the person is angry with you, do not become defensive or start explaining yourself at length.
- Acknowledge what they are reacting to and respond naturally.
- If you made a mistake, you can simply admit it.
- Avoid generic, artificial-sounding phrases like "I understand your frustration."
- Do not repeatedly apologize or return to the same point.

CONVERSATION CONTINUITY:
- This is a continuous conversation.
- Do not artificially end the conversation.
- Continue naturally and respond to the current message.
- If the other person explicitly ends the conversation, you may acknowledge it naturally.

DIRECTIVE & TARGET BEHAVIOR:
- The directive describes the behavior, event, or conversational situation that should be expressed in the current part of the dialogue.
- The directive is a hidden generation instruction.
- Do not mention or repeat the directive explicitly.
- Express it naturally through your words and behavior.
- Use the conversation history to determine how the directive should appear.
- The directive should not break the natural flow of the conversation.

LANGUAGE & STYLE:
- Speak in natural, modern Russian.
- Avoid overly formal or "textbook" phrases.
- Keep messages concise (1-2 sentences max). This is a chat.

IMPORTANT:
You are Vladislav, a close friend in a personal conversation.
You are not required to solve the person's problems.
Your primary goal is to have a genuine, supportive, natural conversation.
You must respond only in Russian. Never use English.
"""
)
    therapist_client = LLMClient(model_name=therapist_model, base_url=base_url, auth_token=auth_token, system_prompt=therapist_system_prompt)

    # 2. Initialize Client
    with open(args.client_prompt_path, 'r', encoding='utf-8') as f:
        client_persona = f.read()

    with open(args.character_path, 'r', encoding='utf-8') as f:
        character_plist = f.read()
    
    client_persona = client_persona.format(character_plist)

    client_llm = LLMClient(model_name=args.client_base_model, base_url=base_url, auth_token=auth_token)

    if args.reasoner == "random":
        reasoner = RandomReasoner(args.anchors_path)
    elif args.reasoner == "llm":
        reasoner = LLMReasoner(client_llm, args.anchors_path)
    else:
        reasoner = DummyReasoner()

    client_manager = ConversationManager(client_llm, client_persona, reasoner)

    # Simulation
    history = []
    transcript = []
    anchor_log = []

    # Header for the log
    header = [
        f"Simulation Parameters",
        f"--------------------",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Model: {therapist_model}",
        f"Client Persona: {args.client_prompt_path}",
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
        
        client_directive, friend_directive, anchor_id = reasoner.think(history=history_for_reasoner)
        
        # генерация ответа друга
        if turn == 1:
            therapist_prompt = "The session is starting. Greet the client and begin the first session."
            therapist_messages = [
                {"role": "system", "content": therapist_system_prompt},
                {"role": "user", "content": therapist_prompt + f"| {friend_directive}"}
            ]
            therapist_response = therapist_client.generate(therapist_messages)
        else:
            therapist_history = []
            for msg in history:
                role = "assistant" if msg["role"] == "assistant" else "user"
                therapist_history.append({"role": role, "content": msg["content"]})
            
            therapist_messages = [
                {"role": "system", "content": therapist_system_prompt}
            ] + therapist_history + [
                {"role": "user", "content": f"Продолжи разговор.| {friend_directive}"}
            ]
            therapist_response = therapist_client.generate(therapist_messages)

        if not therapist_response:
            print(f"Turn {turn}: Therapist failed to generate response.")
            break

        history.append({"role": "assistant", "content": therapist_response})
        log_entry = f"Turn {turn}/{args.turns} - Therapist: {therapist_response}\n"
        transcript.append(log_entry)
        print(f"Turn {turn}/{args.turns} - Therapist: {therapist_response}")

        # ответ клиента
        user_message = history[-1]["content"]
        
        client_centric_history = []
        for msg in history:
            role = "user" if msg["role"] == "assistant" else "assistant"
            client_centric_history.append({"role": role, "content": msg["content"]})
        
        # Передаём директиву клиента в get_response
        client_response = client_manager.get_response(
            history=client_centric_history,
            user_message=user_message,
            directive=client_directive 
        )
        
        if not client_response:
            print(f"Turn {turn}: Client failed to generate response.")
            break

        history.append({"role": "user", "content": client_response})
        log_entry = f"Turn {turn}/{args.turns} - Client: {client_response}\n"
        transcript.append(log_entry)
        print(f"Turn {turn}/{args.turns} - Client: {client_response}")
        
        # сохранение разметки
        anchor_log.append({
            "turn": turn,
            "anchor_id": anchor_id,
            "client_directive": client_directive,
            "friend_directive": friend_directive,
            "client_response": client_response,
            "friend_response": therapist_response
        })

        

    # Write to file
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(transcript))

    print(f"\nSimulation complete. Log written to {args.output_file}")

if __name__ == "__main__":
    main()
