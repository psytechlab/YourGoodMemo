import random
import yaml
from src.llm_client import LLMClient

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

    def __init__(self,  anchors_path: str):
        with open(anchors_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            self.anchors = data.get('anchors', [])

        self.anchor_persistence_steps = 0
        self.current_anchor = None

    def dice_directives(self):
        """Возвращает две директивы: для клиента и для друга"""
        random_num = random.randint(0, len(self.anchors) - 1)
        current_anchor = self.anchors[random_num]
        
        # Выбираем случайную директиву для клиента из списка
        client_dir = random.choice(current_anchor.get("client_directives", [""]))
        
        # Выбираем случайную директиву для друга из списка
        friend_dir = random.choice(current_anchor.get("friend_directives", [""]))
        
        return client_dir, friend_dir, current_anchor.get("id")

#    def dice_directive(self):
#        random_num = random.randint(0, len(self.situations) - 1)
#        return self.situations[random_num]["directive"]

            
#    def think(self, **kwargs):
#        if self.directive_persistence_steps == 0:
#            self.directive_persistence_steps = random.randint(1, 3)
#            self.current_direcitve = self.dice_directive()
#            return self.current_direcitve
#        self.directive_persistence_steps -= 1
#        return "Продолжай общаться в контексте предыдущей директивы" 


def think(self, **kwargs):
    # Если нет текущего anchor или время вышло
    if self.anchor_persistence_steps <= 0 or not self.current_anchor:
        random_num = random.randint(0, len(self.anchors) - 1)
        self.current_anchor = self.anchors[random_num]
        self.anchor_persistence_steps = random.randint(2, 5)
    
    self.anchor_persistence_steps -= 1
    
    client_dir = random.choice(self.current_anchor.get("client_directives", [""]))
    friend_dir = random.choice(self.current_anchor.get("friend_directives", [""]))
    
    return client_dir, friend_dir, self.current_anchor.get("id")

class LLMReasoner(RandomReasoner):
    def __init__(self, llm_client: LLMClient, anchors_path: str):
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
            client_dir = random.choice(self.current_anchor.get("client_directives", [""]))
            friend_dir = random.choice(self.current_anchor.get("friend_directives", [""]))
            return client_dir, friend_dir, self.current_anchor.get("id")
        
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
        
        client_dir = random.choice(self.current_anchor.get("client_directives", [""]))
        friend_dir = random.choice(self.current_anchor.get("friend_directives", [""]))
        
        return client_dir, friend_dir, self.current_anchor.get("id")



class DummyReasoner(IReasoner):
    def think(self, **kwargs) -> str:
        return ""
