import requests


class Message:
    def __init__(self, message_dict):
        self.content = message_dict.get("content", None)
        self.role = message_dict.get("role", None)


class Choice:
    def __init__(self, choice_dict):
        self.finish_reason = choice_dict.get("finish_reason", None)
        self.index = choice_dict.get("index", None)
        self.message = Message(choice_dict.get("message", {}))


class ResponseObject:
    def __init__(self, data_dict):
        self.choices = [Choice(choice) for choice in data_dict.get("choices", [])]
        self.created = data_dict.get("created", None)
        self.id = data_dict.get("id", None)
        self.model = data_dict.get("model", None)
        self.request_id = data_dict.get("request_id", None)
        self.usage = data_dict.get("usage", {})


class OpenAI3P:
    """
    通过 HTTP 请求与大型语言模型交互，模拟 OpenAI 的调用方式。
    """

    def __init__(self, model: str, url: str, api_key: str = ""):
        """
        初始化实例。

        :param url: LLM 服务的 URL。
        :param api_key: API 认证密钥。
        """
        self.model = model
        self.base_url = url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, prompt: str, system_message=None, max_tokens: int = 10000):
        """
        模拟 OpenAI 的 chat.completions.create 方法。

        :param model: 模型名称。
        :param messages: 消息列表。
        :param max_tokens: 最大生成标记数。
        :return: LLM 的响应或错误信息。
        """
        if system_message is None:
            system_message = "You are a helpful assistant that answer questions and provide guidance."
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        url = f"{self.base_url}/v4/chat/completions"
        data = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            return ResponseObject(response.json())
        except requests.RequestException as e:
            return f"发生错误：{e}"

    def get_response(self, prompt: str, system_message=None, max_tokens: int = 10000):
        try:
            return (
                self.create(prompt, system_message, max_tokens)
                .choices[0]
                .message.content
            )
        except:
            raise ValueError("Received None.")
