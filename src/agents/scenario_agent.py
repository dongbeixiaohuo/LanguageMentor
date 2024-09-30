import json
import random
import os

from langchain_ollama.chat_models import ChatOllama  # 导入 ChatOllama 模型
from langchain_openai import ChatOpenAI  # 导入 ChatOpenAI 模型
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 导入提示模板相关类
from langchain_core.messages import HumanMessage, AIMessage  # 导入人类消息类和AI消息类
from langchain_core.runnables.history import RunnableWithMessageHistory  # 导入带有消息历史的可运行类

from .session_history import get_session_history  # 导入会话历史相关方法
from utils.logger import LOG

class ScenarioAgent:
    def __init__(self, scenario):
        self.scenario = scenario
        self.prompt_file = os.path.join("prompts", f"{scenario}_prompt.txt")
        self.intro_file = os.path.join("content", "intro", f"{scenario}.json")
        self.prompt = self.load_prompt()
        self.intro_messages = self.load_intro()
        self.create_chatbot()

    def load_prompt(self):
        try:
            with open(self.prompt_file, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file {self.prompt_file} not found!")

    def load_intro(self):
        try:
            with open(self.intro_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Intro file {self.intro_file} not found!")
        except json.JSONDecodeError:
            raise ValueError(f"Intro file {self.intro_file} contains invalid JSON!")

    def create_chatbot(self):
        # 定义可用的模型
        self.models = {
            "llama": "llama3.1:8b-instruct-q8_0",
            "gpt4o": "gpt-4o",
            "gpt4o_mini": "gpt-4o-mini"
        }

        # 创建聊天提示模板
        system_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])

        # 初始化多个聊天机器人
        self.chatbots = {}
        for model_name, model_id in self.models.items():
            if model_name == "llama":
                self.chatbots[model_name] = system_prompt | ChatOllama(
                    model=model_id,
                    base_url="http://localhost:11435",
                    max_tokens=8192,
                    temperature=0.8,
                )
            else:  # gpt-4o 和 gpt-4o-mini
                self.chatbots[model_name] = system_prompt | ChatOpenAI(
                    model=model_id,
                    base_url="https://api.javis3000.com/v1",  # 替换为实际的自定义 API 地址
                    api_key=os.getenv("OPENAI_API_KEY"),  # 确保设置了环境变量
                   
                )

        # 默认使用 llama 模型
        self.current_model = "llama"
        self.chatbot = self.chatbots[self.current_model]
        self.chatbot_with_history = RunnableWithMessageHistory(self.chatbot, get_session_history)

    def switch_model(self, model_name):
        if model_name in self.models:
            self.current_model = model_name
            self.chatbot = self.chatbots[model_name]
            self.chatbot_with_history = RunnableWithMessageHistory(self.chatbot, get_session_history)
            return f"已切换到 {model_name} 模型"
        else:
            return f"无效的模型名称: {model_name}"

    def start_new_session(self, session_id: str = None):
        """
        开始一个新的聊天会话，并发送初始AI消息。
        
        参数:
            session_id (str): 会话的唯一标识符
        """
        if session_id is None:
            session_id = self.scenario  # 使用场景名称作为默认的会话ID

        history = get_session_history(session_id)
        LOG.debug(f"[history]:{history}")

        if not history.messages:  # 检查历史记录是否为空
            initial_ai_message = random.choice(self.intro_messages)  # 随机选择初始AI消息
            history.add_message(AIMessage(content=initial_ai_message))  # 添加初始AI消息到历史记录
            return initial_ai_message
        else:
            return history.messages[-1].content  # 返回历史记录中的最后一条消息

    def chat_with_history(self, user_input, session_id: str = None):
        """
        处理用户输入并生成包含聊天历史的回复。
        
        参数:
            user_input (str): 用户输入的消息
            session_id (str): 会话的唯一标识符
        
        返回:
            str: 代理生成的回复内容
        """
        if session_id is None:
            session_id = self.scenario  # 使用场景名称作为默认的会话ID

        response = self.chatbot_with_history.invoke(
            [HumanMessage(content=user_input)],
            {"configurable": {"session_id": session_id}},
        )
        
        return response.content
