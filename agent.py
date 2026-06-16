# agent.py — LangGraph-агент з інструментами (Gemini)

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

import numexpr as ne
from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.utilities import WikipediaAPIWrapper

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

# ============================================================
# КОНСТАНТИ
# ============================================================
MODEL_NAME = "gemini-3.1-flash-lite"

# Системна інструкція, щоб модель ДІЙСНО користувалась інструментами
SYSTEM_PROMPT = SystemMessage(
    content=(
        "Ти — AI-агент із інструментами. Відповідай українською, будь чітким і корисним.\n"
        "1) Якщо потрібно дізнатися дату, час або день тижня — викликай current_datetime.\n"
        "2) Для управління задачами використовуй create_task(title, priority), set_task_done(task_id) або list_open_tasks().\n"
        "3) Якщо інструмент повернув помилку, поясни її користувачу та попроси уточнення.\n"
        "4) Якщо ти не маєш точної відповіді, чесно скажи, що не знаєш.\n"
    )
)

# ============================================================
# СХЕМА СТАНУ
# ============================================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tasks: list[dict] # Кожне завдання: {'id': int, 'title': str, 'status': str}
    next_task_id: int

# ============================================================
# ІНСТРУМЕНТИ
# ============================================================
from datetime import timedelta
import json # Для серіалізації/десеріалізації структурованих даних

# ============================================================
# ІНСТРУМЕНТ 1: Поточна дата та час
# ============================================================
@tool
def current_datetime() -> str:
    """
    Повертає поточну дату, час та день тижня (UTC+3).
    """
    now_utc = datetime.now()
    now_utc3 = now_utc + timedelta(hours=3)

    days_ua = {
        "Monday": "понеділок", "Tuesday": "вівторок", "Wednesday": "середа",
        "Thursday": "четвер", "Friday": "п'ятниця", "Saturday": "субота", "Sunday": "неділя"
    }

    day_en = now_utc3.strftime("%A")
    return f"Зараз {days_ua.get(day_en, day_en)}, {now_utc3.strftime('%d.%m.%Y %H:%M:%S')}"

# ============================================================
# ІНСТРУМЕНТИ ДЛЯ ЗАВДАНЬ (To-Do List)
# Ці інструменти повертають структуровані дані (JSON-рядок),
# які потім будуть оброблені у вузлі task_manager_node.
# ============================================================
@tool
def create_task(title: str, priority: str = "Medium") -> str:
    """
    Генерує запит на створення нового завдання.
    Використовуй цей інструмент, коли користувач просить записати або створити завдання.

    Args:
        title: Опис завдання.
    """
    return json.dumps(
        {
            'type': 'task_action',
            'action': 'create',
            'title': title,
            'priority': priority,
        },
        ensure_ascii=False,
    )

@tool
def set_task_done(task_id: int) -> str:
    """Генерує запит на позначення завдання як виконаного за його ID.

    Args:
        task_id: Унікальний ідентифікатор завдання.
    """
    return json.dumps({'type': 'task_action', 'action': 'set_done', 'task_id': task_id}, ensure_ascii=False)

@tool
def list_open_tasks() -> str:
    """
    Генерує запит на отримання списку всіх відкритих завдань.
    Використовуй цей інструмент, коли користувач просить показати завдання.
    """
    return json.dumps(
        {'type': 'task_action', 'action': 'list'},
        ensure_ascii=False,
    )

# ============================================================
# Збираємо всі інструменти в список
# ============================================================
tools = [current_datetime, create_task, set_task_done, list_open_tasks]


# ============================================================
# ПОБУДОВА ГРАФА
# ============================================================
def create_agent(api_key: str, model_name: str = MODEL_NAME):
    """
    Створює скомпільований LangGraph-агент.

    Args:
        api_key: Google API key
        model_name: Назва моделі Gemini

    Returns:
        Скомпільований граф агента
    """
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.0,  # нижче = краще для tool-calling
        api_key=api_key,
    )
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        """
        Вузол агента: викликає мовну модель з поточною історією діалогу.

        Модель аналізує контекст і вирішує одне з двох:
        1. Викликати інструмент — генерує AIMessage з tool_calls
        2. Відповісти напряму — генерує AIMessage зі звичайним текстом

        Args:
            state: Поточний стан графа з історією повідомлень

        Returns:
            Часткове оновлення стану: словник з новим повідомленням моделі.
            Завдяки редуктору add_messages це повідомлення буде ДОДАНО
            до існуючої історії, а не замінить її.
        """
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)

        # Нормалізуємо вміст AIMessage, щоб він завжди був у форматі [{'text': '...'}]
        # Це допомагає уникнути попереджень 'Unrecognized message part format'.
        normalized_content = []
        if isinstance(response.content, str):
            if response.content:
                normalized_content.append({'text': response.content})
        elif isinstance(response.content, list):
            # Якщо вміст вже є списком, припускаємо, що він у правильному форматі
            # (наприклад, від task_manager_node) або може бути далі оброблений.
            # Проте, для уніфікації, спробуємо об'єднати текстові частини в один словник.
            combined_text = []
            for item in response.content:
                if isinstance(item, dict) and 'text' in item: # Для вже структурованих частин
                    combined_text.append(item['text'])
                elif isinstance(item, str): # Для простих текстових частин
                    combined_text.append(item)
            if combined_text:
                normalized_content.append({'text': ' '.join(combined_text)})

        # Створюємо нове AIMessage з нормалізованим вмістом та зберігаємо tool_calls
        # Якщо `tool_calls` присутні, а `normalized_content` порожній, це є коректним.
        standardized_ai_message = AIMessage(
            content=normalized_content,
            tool_calls=response.tool_calls
        )
        return {"messages": [standardized_ai_message]}

    tool_node = ToolNode(tools)
    graph = StateGraph(AgentState, None, input_schema=AgentState, output_schema=AgentState)
    graph.add_sequence([agent_node, tool_node, task_manager_node])
    graph.set_entry_point("agent_node")
    graph.set_finish_point("task_manager_node")
    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)

# ============================================================
# ВУЗОЛ 2: Інструменти (виконання tool calls)
# ============================================================

# ============================================================
# ВУЗОЛ 3: Менеджер завдань (опрацювання та оновлення стану завдань)
# ============================================================
def task_manager_node(state: AgentState) -> dict:
    """
    Вузол, який обробляє результати tool_calls, пов'язаних із завданнями,
    та оновлює стан `tasks`.
    """
    current_tasks = state.get("tasks", [])
    next_id = state.get("next_task_id", 1)
    messages = state["messages"]

    # Додаємо перевірку, щоб уникнути помилки list index out of range, якщо messages порожній
    if not messages:
        return {} # Якщо немає повідомлень, немає tool_output для обробки

    last_message = messages[-1]

    updated_tasks = list(current_tasks) # Створюємо змінну копію списку завдань
    response_for_agent = ""

    if isinstance(last_message, ToolMessage):
        tool_output_str = last_message.content
        try:
            tool_output = json.loads(tool_output_str)
        except json.JSONDecodeError:
            # Якщо результат інструменту не JSON, це, ймовірно, інший інструмент
            return {}

        if tool_output.get('type') == 'task_action':
            action = tool_output['action']

            if action == 'create':
                title = tool_output['title']
                priority = tool_output.get('priority', 'Medium')
                new_task = {
                    'id': next_id,
                    'title': title,
                    'status': 'open',
                    'priority': priority,
                }
                updated_tasks.append(new_task)
                next_id += 1
                response_for_agent = f"Завдання '{title}' (ID: {new_task['id']}) успішно створено."

            elif action == 'set_done':
                task_id_to_complete = tool_output['task_id']
                found = False
                for task in updated_tasks:
                    if task['id'] == task_id_to_complete:
                        task['status'] = 'done'
                        found = True
                        response_for_agent = f"Завдання '{task['title']}' (ID: {task_id_to_complete}) позначено як виконане."
                        break
                if not found:
                    response_for_agent = f"Завдання з ID {task_id_to_complete} не знайдено."

            elif action == 'list':
                open_tasks = [task for task in updated_tasks if task['status'] == 'open']
                if open_tasks:
                    task_list_str = "\n".join([f"- {t['id']}: {t['title']}" for t in open_tasks])
                    response_for_agent = f"Ось ваші відкриті завдання:\n{task_list_str}"
                else:
                    response_for_agent = "Відкритих завдань немає."

            # Додаємо повідомлення для агента, щоб він міг відповісти користувачеві
            # Форматуємо як ToolMessage, щоб агент міг обробити цей результат
            return {
                "tasks": updated_tasks,
                "next_task_id": next_id,
                "messages": [ToolMessage(content=response_for_agent, tool_call_id="task_manager_feedback")]
            }

    return {} # Якщо це не дія з завданням, нічого не змінюємо

def extract_response_text(message) -> str:
    """Витягує текст з повідомлення LangChain."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)

def extract_tools_debug(messages: list[Any]) -> list[dict]:
    """Повертає короткий список викликів інструментів/результатів (для debug UI)."""
    debug = []
    for m in messages:
        if isinstance(m, ToolMessage):
            debug.append({"type": "tool_result", "content": m.content, "tool_call_id": m.tool_call_id})
        else:
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    debug.append(
                        {
                            "type": "tool_call",
                            "name": tc.get("name"),
                            "args": tc.get("args"),
                            "id": tc.get("id"),
                        }
                    )
    return debug
