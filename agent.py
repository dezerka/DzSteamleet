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
from langgraph.prebuilt import ToolNode, tools_condition
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
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.0,
        api_key=api_key,
    )
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        
        # 1. Додаємо системний промпт на початок, якщо його там ще немає
        if not messages or not isinstance(messages[0], SystemMessage):
            invoke_messages = [SYSTEM_PROMPT] + messages
        else:
            invoke_messages = messages

        # 2. Викликаємо LLM з повним контекстом
        response = llm_with_tools.invoke(invoke_messages)
        return {"messages": [response]}

    tool_node = ToolNode(tools)
    graph = StateGraph(AgentState)
    
    # Додаємо вузли
    graph.add_node("agent_node", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("task_manager_node", task_manager_node)

    graph.set_entry_point("agent_node")

    # УМОВНИЙ ПЕРЕХІД: Якщо є виклик інструменту -> йдемо в tools, інакше -> кінець
    graph.add_conditional_edges(
        "agent_node",
        tools_condition,
        {"tools": "tools", "__end__": "__end__"}
    )
    
    # Після tools йдемо в task_manager
    graph.add_edge("tools", "task_manager_node")
    
    # ГОЛОВНЕ ВИПРАВЛЕННЯ: Після оновлення задач ЗАВЖДИ повертаємось до агента!
    graph.add_edge("task_manager_node", "agent_node")

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
    Вузол оновлює стан завдань та підставляє реальні дані у відповіді інструментів.
    """
    current_tasks = state.get("tasks", [])
    next_id = state.get("next_task_id", 1)
    messages = state.get("messages", [])

    if not messages:
        return {}

    last_message = messages[-1]
    updated_tasks = list(current_tasks)
    
    # Словник для повернення змін у граф
    return_changes = {}

    if isinstance(last_message, ToolMessage):
        try:
            tool_output = json.loads(last_message.content)
            if tool_output.get('type') == 'task_action':
                action = tool_output['action']

                # 1. Створення завдання
                if action == 'create':
                    updated_tasks.append({
                        'id': next_id,
                        'title': tool_output['title'],
                        'status': 'open',
                        'priority': tool_output.get('priority', 'Medium'),
                    })
                    next_id += 1

                # 2. Закриття завдання
                elif action == 'set_done':
                    for task in updated_tasks:
                        if task['id'] == tool_output['task_id']:
                            task['status'] = 'done'
                            break

                # 3. ГОЛОВНЕ ВИПРАВЛЕННЯ: Запит списку завдань
                elif action == 'list':
                    # Фільтруємо завдання з поточного стану графа
                    open_tasks = [t for t in updated_tasks if t.get('status') != 'done']
                    
                    # Модифікуємо вміст ОСТАННЬОГО повідомлення інструменту.
                    # Ми зберігаємо оригінальний id та tool_call_id, які очікує Gemini,
                    # але підміняємо пусту команду на реальний список задач!
                    updated_message = ToolMessage(
                        id=last_message.id,
                        tool_call_id=last_message.tool_call_id,
                        content=json.dumps({"status": "success", "tasks": open_tasks}, ensure_ascii=False),
                        name=last_message.name
                    )
                    # Додаємо оновлене повідомлення (LangGraph замінить старе за його ID)
                    return_changes["messages"] = [updated_message]

        except json.JSONDecodeError as e:
            print(f"Помилка при розборі JSON: {str(e)}")
            pass 

    # Оновлюємо масив завдань та лічильник ID у стані
    return_changes.update({
        "tasks": updated_tasks,
        "next_task_id": next_id,
    })
    
    return return_changes
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
