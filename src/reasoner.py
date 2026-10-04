import random
import yaml
from src.llm_client import LLMuser

class IReasoner:
    """The part of a brain that decide how to behaive.

    In the context of chat bot it decide the strategy to use
    that is represented in directive - a short command descrived
    the behaviour.

    Currently, the reasiner is any entity that produce the directive given any
    nessesery input.
    """

    def think(self, **kwargs) -> str:
        pass


class RandomReasoner(IReasoner):

    def __init__(self, anchors_path: str, filler_topics_path: str, filler_instruction_path: str,):

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

#    def dice_directives(self):
#        """Возвращает две директивы: для пользователя и для друга"""
#        random_num = random.randint(0, len(self.anchors) - 1)
#        current_anchor = self.anchors[random_num]
        
        # Выбираем случайную директиву для пользователя из списка
#        user_dir = random.choice(current_anchor.get("user_directives", [""]))
        
#        # Выбираем случайную директиву для друга из списка
#        friend_dir = random.choice(current_anchor.get("friend_directives", [""]))
        
#        return user_dir, friend_dir, current_anchor.get("id")


#def think(self, **kwargs):
#    # Если нет текущего anchor или время вышло
#    if self.anchor_persistence_steps <= 0 or not self.current_anchor:
#        random_num = random.randint(0, len(self.anchors) - 1)
#        self.current_anchor = self.anchors[random_num]
#        self.anchor_persistence_steps = random.randint(2, 5)
    
#    self.anchor_persistence_steps -= 1
    
#    user_dir = random.choice(self.current_anchor.get("user_directives", [""]))
#    friend_dir = random.choice(self.current_anchor.get("friend_directives", [""]))
    
#    return user_dir, friend_dir, self.current_anchor.get("id")

    def think(self, previous_anchor_id=None, **kwargs):

        # 1. Продолжаем текущий блок
        if self.persistence_steps > 0:
            self.persistence_steps -= 1
            return self._current_block()

        # 2. Новый блок
        return self._start_block(previous_anchor_id)

    def _start_block(self, previous_anchor_id=None):
        # Новый блок
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
            user_dir = random.choice(self.current_anchor.get("user_directives", [""]))
            friend_dir = random.choice(self.current_anchor.get("friend_directives", [""]))
            return (
                "anchor",
                user_dir,
                friend_dir,
                self.current_anchor["id"],
                "none",
                self.current_block_length,
                self.current_block_id,
            )

        if self.current_block_type == "filler":
            directive = self.current_topic
            return (
                "filler",
                directive,
                directive,
                "none",
                self.current_topic,
                self.current_block_length,
                self.current_block_id,
            )

        return ("none", "none", "none", "none", "none", 0, "none")



    def _switch_block_type(self):
        # Меняет тип блока: anchor на filler
        if self.current_block_type is None:
            self.current_block_type = random.choice(["anchor", "filler"])
        elif self.current_block_type == "anchor":
            self.current_block_type = "filler"
        else:
            self.current_block_type = "anchor"

    def _select_anchor(self, previous_anchor_id):
        # Выбирает anchor, не равный предыдущему
        available = [a for a in self.anchors if a["id"] != previous_anchor_id]
        if not available:
            available = self.anchors
        return random.choice(available)

    def _select_filler_topic(self):
        # Выбирает тему, не из recent_filler_topics
        available = [t for t in self.filler_topics if t not in self.recent_filler_topics]
        if not available:
            available = self.filler_topics
        return random.choice(available)

    def _update_recent_topics(self, topic):
        # Хранит 2 последние темы
        self.recent_filler_topics.append(topic)
        self.recent_filler_topics = self.recent_filler_topics[-2:]




class LLMReasoner(RandomReasoner):
    def __init__(self, llm_client: LLMuser, anchors_path: str):
        self.llm_client = llm_client
        self.anchor_persistence_steps = 0
        self.current_anchor = None  
        self.directive_repeat_counter = 0
        self.cooldown_steps = 0
        self.delay = random.choice([2,4,6,8,10])
        
        with open(anchors_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            self.anchors = data.get('anchors', []) 

        # Trigger new directive
        self.prompt_template = """
        Analyze the following dialogue history. By using provided description and typical utterances, identify which anchor from the list above best matches the current state of the conversation, especially the last message.

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

    def think(self, **kwargs) -> tuple:
        history = kwargs.get('history', [])
        
        if self.anchor_persistence_steps > 0 and self.current_anchor:
            self.anchor_persistence_steps -= 1
            user_dir = random.choice(self.current_anchor.get("user_directives", [""]))
            friend_dir = random.choice(self.current_anchor.get("friend_directives", [""]))
            return user_dir, friend_dir, self.current_anchor.get("id")
        
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
        
        # Проверка на повторение
        if self.current_anchor and self.current_anchor.get("id") == selected_anchor.get("id"):
            self.directive_repeat_counter += 1
            if self.directive_repeat_counter >= 2:
                selected_anchor = random.choice(self.anchors)
                self.directive_repeat_counter = 0
        else:
            self.directive_repeat_counter = 0
        
        self.current_anchor = selected_anchor
        self.anchor_persistence_steps = random.randint(2, 5)
        
        user_dir = random.choice(self.current_anchor.get("user_directives", [""]))
        friend_dir = random.choice(self.current_anchor.get("friend_directives", [""]))
        
        return user_dir, friend_dir, self.current_anchor.get("id")



class DummyReasoner(IReasoner):
    def think(self, **kwargs):
        return ("anchor", "none", "none", "none", "none", 0, "none")
