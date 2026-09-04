import requests

class LLMClient:

    def __init__(self, model_name: str, base_url:str, auth_token:str, system_prompt=None, temperature=0.7):
        self.base_url = base_url
        self.model_name = model_name
        headers = {
        'Content-Type': 'application/json'
        }
        if auth_token is not None:
            headers['Authorization'] = f'Bearer {auth_token}'
        self.sess = requests.Session()
        self.sess.headers.update(headers)
        self.system_prompt = system_prompt
        self.temperature = temperature

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt

    def make_request(self, data):
        try:
            # Формируем правильный URL
            url = self.base_url.rstrip("/") + "/chat/completions"
            
            response = self.sess.post(url, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            print(f"Произошла ошибка: {str(e)}")
            try:
                print(response.text)
            except:
                pass
            return None
        
    def generate(self, messages: list[dict[str, str]]):
        # ПРИНУДИТЕЛЬНАЯ ПОДСТАНОВКА, ЕСЛИ model_name = None
        if self.model_name is None:
            self.model_name = "mistral-nemo:12b"
            print("🔧 ПРИНУДИТЕЛЬНО установлена модель в generate")
        
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature
        }
        return self.make_request(data)
        
    

    def respond(self, user_prompt: str):
        messages = []
        if self.system_prompt is not None:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        return self.generate(messages)