import random
import yaml
from src.llm_client import LLMClient


class IReasoner:
    def think(self, **kwargs) -> str:
        pass



# Narrator отвечает за структуру текста
class Narrator:

    def __init__(self, anchors_path: str, filler_topics_path: str, filler_instruction_path: str):
        with open(anchors_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            self.anchors = data.get('anchors', [])

        with open(filler_topics_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            self.filler_topics = data.get('filler_topics', [])

        with open(filler_instruction_path, 'r', encoding='utf-8') as f:
            self.filler_instruction = f.read()

        self.current_anchor = None
        self.current_block_type = None
        self.current_topic = None
        self.persistence_steps = 0
        self.current_block_length = 0
        self.recent_filler_topics = []

        self._block_counter = 1
        self.current_block_id = None

    def think(self, previous_anchor_id=None, **kwargs):
        if self.persistence_steps > 0:
            self.persistence_steps -= 1
            return self._current_block()
        return self._start_block(previous_anchor_id)

    def _start_block(self, previous_anchor_id=None):
        self._switch_block_type()

        if self.current_block_type == "anchor":
            self.current_anchor = self._select_anchor(previous_anchor_id)
            self.current_topic = None
            self.current_block_length = random.randint(2, 5)
            self.current_block_id = f"anchor_{self._block_counter:03d}"

        elif self.current_block_type == "filler":
            self.current_anchor = None
            self.current_topic = self._select_filler_topic()
            self._update_recent_topics(self.current_topic)
            self.current_block_length = random.randint(3, 6)
            self.current_block_id = f"filler_{self._block_counter:03d}"

        self._block_counter += 1
        self.persistence_steps = self.current_block_length - 1

        return self._current_block()

    def _current_block(self):
        if self.current_block_type == "anchor":
            return (
                "anchor",
                self.current_anchor.get("directive", ""),
                self.current_anchor["id"],
                "none",
                self.current_block_length,
                self.current_block_id,
            )

        if self.current_block_type == "filler":
            return (
                "filler",
                self.current_topic,
                "none",
                self.current_topic,
                self.current_block_length,
                self.current_block_id,
            )

        return ("none", "none", "none", "none", 0, "none")

    def _switch_block_type(self):
        if self.current_block_type is None:
            self.current_block_type = random.choice(["anchor", "filler"])
        elif self.current_block_type == "anchor":
            self.current_block_type = "filler"
        else:
            self.current_block_type = "anchor"

    def _select_anchor(self, previous_anchor_id):
        available = [a for a in self.anchors if a["id"] != previous_anchor_id]
        if not available:
            available = self.anchors
        return random.choice(available)

    def _select_filler_topic(self):
        available = [t for t in self.filler_topics if t not in self.recent_filler_topics]
        if not available:
            available = self.filler_topics
        return random.choice(available)

    def _update_recent_topics(self, topic):
        self.recent_filler_topics.append(topic)
        self.recent_filler_topics = self.recent_filler_topics[-2:]



class LLMNarrator(Narrator):
    def __init__(self, llm_client, anchors_path, filler_topics_path, filler_instruction_path):
        super().__init__(anchors_path, filler_topics_path, filler_instruction_path)
        self.llm_client = llm_client

        self.prompt_template = """
        Analyze the following dialogue history. Identify which anchor from the list above best matches the current state of the conversation, especially the last message.

        Available anchors:
        {}

        Dialogue History:
        {}

        Return ONLY the ID of the anchor.
        """

    def _get_anchors_context(self) -> str:
        if not self.anchors:
            return "No anchors available."
        context = "Available anchors:\n"
        for a in self.anchors:
            context += f"- {a['id']}: {a.get('name', '')}\n"
        return context

    def _start_block(self, previous_anchor_id=None, history=None):
        self._switch_block_type()

        if self.current_block_type == "anchor":
            # Используем LLM для выбора anchor'а
            prompt = self.prompt_template.format(
                self._get_anchors_context(),
                history or ""
            )
            anchor_id = self.llm_client.respond(prompt).strip()

            selected_anchor = None
            for a in self.anchors:
                if a["id"] == anchor_id:
                    selected_anchor = a
                    break

            if not selected_anchor:
                selected_anchor = random.choice(self.anchors)

            # Запрет повтора
            if previous_anchor_id and selected_anchor["id"] == previous_anchor_id:
                available = [a for a in self.anchors if a["id"] != previous_anchor_id]
                selected_anchor = random.choice(available)

            self.current_anchor = selected_anchor
            self.current_topic = None
            self.current_block_length = random.randint(2, 5)
            self.current_block_id = f"anchor_{self._block_counter:03d}"

        elif self.current_block_type == "filler":
            self.current_anchor = None
            self.current_topic = self._select_filler_topic()
            self._update_recent_topics(self.current_topic)
            self.current_block_length = random.randint(3, 6)
            self.current_block_id = f"filler_{self._block_counter:03d}"

        self._block_counter += 1
        self.persistence_steps = self.current_block_length - 1

        return self._current_block()


class DummyNarrator:
    def think(self, previous_anchor_id=None, **kwargs):
        return ("anchor", "none", "none", "none", 0, "none")



# Reasoner (локальный) отвечает только за директиву для реплики внутри блока, think возвращает только одну директиву
class Reasoner(IReasoner):

    def __init__(self, anchors_path: str, filler_instruction_path: str):
        with open(anchors_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            self.anchors = data.get('anchors', [])

        with open(filler_instruction_path, 'r', encoding='utf-8') as f:
            self.filler_instruction = f.read()

    def think(self, block_type, anchor_id, topic, **kwargs) -> str:
        if block_type == "anchor":
            for a in self.anchors:
                if a["id"] == anchor_id:
                    return a.get("directive", "")
            return ""
        if block_type == "filler":
            return topic
        return ""


class DummyReasoner(IReasoner):
    def think(self, **kwargs):
        return "none"