# app.py — Крок 6: Інтеграція з LangGraph-агентом (Gemini)

import json
import uuid

import streamlit as st
from google import genai
from google.genai.types import GenerateContentConfig

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# Імпортуємо агента
from agent import create_agent, extract_response_text, extract_tools_debug, MODEL_NAME

# ============================================================
# НАЛАШТУВАННЯ СТОРІНКИ
# ============================================================
st.set_page_config(
    page_title="AI Чатбот з Gemini",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# ІНІЦІАЛІЗАЦІЯ
# ============================================================
@st.cache_resource
def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)

@st.cache_resource
def get_langgraph_agent(api_key: str, model_name: str):
    return create_agent(api_key, model_name)

api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ Не знайдено GOOGLE_API_KEY у secrets.toml")
    st.stop()

# ============================================================
# ЗАГОЛОВОК
# ============================================================
st.title("🤖 AI Чатбот з Gemini")

# ============================================================
# ІНІЦІАЛІЗАЦІЯ СТАНУ
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

if "tasks" not in st.session_state:
    st.session_state.tasks = []

if "task_next_id" not in st.session_state:
    st.session_state.task_next_id = 1

if "filter_status" not in st.session_state:
    st.session_state.filter_status = "All"

if "filter_priority" not in st.session_state:
    st.session_state.filter_priority = "All"


def get_filtered_tasks() -> list[dict]:
    tasks = st.session_state.get("tasks", [])
    if st.session_state.get("filter_status", "All") != "All":
        tasks = [task for task in tasks if task.get("status") == st.session_state.filter_status]
    if st.session_state.get("filter_priority", "All") != "All":
        tasks = [task for task in tasks if task.get("priority") == st.session_state.filter_priority]
    return tasks


def render_task_card(task: dict, key_suffix: str = ""):
    st.markdown(f"**{task['title']}**")
    st.caption(f"ID: {task['id']} • Пріоритет: {task['priority']} • Статус: {task['status']}")

    status_options = ["open", "in_progress", "done"]
    priority_options = ["High", "Medium", "Low"]
    suffix = f"_{key_suffix}" if key_suffix else ""

    current_status = task.get("status", "open")
    if current_status not in status_options:
        current_status = "open"

    status_key = f"status_{task['id']}{suffix}"
    if st.session_state.get(status_key) != current_status:
        st.session_state[status_key] = current_status

    new_status = st.selectbox(
        "Статус",
        status_options,
        index=status_options.index(current_status),
        key=status_key,
    )
    if new_status != task["status"]:
        task["status"] = new_status

    current_priority = task.get("priority", "Medium")
    if current_priority not in priority_options:
        current_priority = "Medium"

    priority_key = f"priority_{task['id']}{suffix}"
    if st.session_state.get(priority_key) != current_priority:
        st.session_state[priority_key] = current_priority

    new_priority = st.selectbox(
        "Пріоритет",
        priority_options,
        index=priority_options.index(current_priority),
        key=priority_key,
    )
    if new_priority != task["priority"]:
        task["priority"] = new_priority

    done_key = f"done_{task['id']}{suffix}"
    done_default = task["status"] == "done"
    if st.session_state.get(done_key) != done_default:
        st.session_state[done_key] = done_default

    done_value = st.checkbox(
        "Виконано",
        value=st.session_state[done_key],
        key=done_key,
    )
    if done_value and task["status"] != "done":
        task["status"] = "done"
    elif not done_value and task["status"] == "done":
        task["status"] = "open"

# ============================================================
# БІЧНА ПАНЕЛЬ
# ============================================================
status_filter_options = ["All", "open", "in_progress", "done"]
priority_filter_options = ["All", "High", "Medium", "Low"]

with st.sidebar:
    st.header("⚙️ Налаштування")

    mode = st.radio(
        "Режим",
        ["💬 Звичайний чат", "🛠️ Агент з інструментами"],
        index=0,
        key="mode_radio",
        help="Агент може використовувати інструменти, проте звичайний режим стабільніший",
    )

    st.divider()
    st.info(f"Модель: **{MODEL_NAME}**")

    temperature = st.slider(
        "Температура (для звичайного чату)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        key="temperature_slider",
    )

    show_agent_debug = False
    if "Агент" in mode:
        show_agent_debug = st.checkbox("Показувати debug агента (tool calls)", value=False)

    st.divider()
    st.subheader("Фільтри задач")
    st.selectbox("Статус", status_filter_options, key="filter_status")
    st.selectbox("Пріоритет", priority_filter_options, key="filter_priority")

    st.divider()
    st.subheader("Статистика")
    total_tasks = len(st.session_state.get("tasks", []))
    done_tasks = len([task for task in st.session_state.get("tasks", []) if task.get("status") == "done"])
    open_tasks = len([task for task in st.session_state.get("tasks", []) if task.get("status") == "open"])
    in_progress_tasks = len([task for task in st.session_state.get("tasks", []) if task.get("status") == "in_progress"])
    st.metric("Виконано", f"{done_tasks}/{total_tasks}")
    st.metric("Відкриті", open_tasks)
    st.metric("В процесі", in_progress_tasks)

    st.divider()
    st.subheader("Швидке додавання задачі")
    with st.form("quick_add_task", clear_on_submit=True):
        quick_title = st.text_input("Назва задачі")
        quick_priority = st.selectbox("Пріоритет", ["Medium", "High", "Low"], index=0)
        quick_add = st.form_submit_button("Додати задачу")
        if quick_add and quick_title:
            task_id = st.session_state.get("task_next_id", 1)
            st.session_state.tasks.append(
                {
                    "id": task_id,
                    "title": quick_title,
                    "status": "open",
                    "priority": quick_priority,
                }
            )
            st.session_state.task_next_id = task_id + 1
            st.success("Задачу додано")

    st.divider()
    if "Агент" in mode:
        with st.expander("🛠️ Доступні інструменти"):
            st.markdown(
                """
- **create_task(title, priority)** — створення задачі
- **set_task_done(task_id)** — позначити задачу як виконану
- **list_open_tasks()** — показати відкриті задачі
- **current_datetime** — поточна дата/час
"""
            )
    st.divider()
# ============================================================
# ФУНКЦІЇ ДЛЯ ЗВИЧАЙНОГО РЕЖИМУ (стрімінг)
# ============================================================
def convert_to_gemini_history(messages: list) -> list[dict]:
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            {
                "role": role,
                "parts": [{"text": msg["content"]}],
            }
        )
    return contents

def stream_gemini_response(prompt: str, history: list):
    client = get_gemini_client(api_key)
    contents = convert_to_gemini_history(history)
    contents.append(
        {
            "role": "user",
            "parts": [{"text": prompt}],
        }
    )

    try:
        stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
            config=GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=2048,
            ),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"\n\n❌ **Помилка:** {str(e)}"


def sync_tasks_from_agent(result: dict):
    if not isinstance(result, dict):
        return

    if "tasks" in result and isinstance(result["tasks"], list):
        st.session_state.tasks = result["tasks"]

    if "next_task_id" in result and isinstance(result["next_task_id"], int):
        st.session_state.task_next_id = result["next_task_id"]

    # Fallback: якщо tasks не прийшли, оновлюємо статус за tool_call set_task_done
    if "tasks" not in result:
        messages = result.get("messages", [])
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            content = message.content
            if not isinstance(content, str):
                continue
            try:
                tool_output = json.loads(content)
            except json.JSONDecodeError:
                continue

            if tool_output.get("type") != "task_action":
                continue

            action = tool_output.get("action")
            if action == "set_done":
                task_id = tool_output.get("task_id")
                for task in st.session_state.tasks:
                    if task.get("id") == task_id:
                        task["status"] = "done"
                        break


def get_agent_state(prompt: str) -> dict:
    history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))

    history.append(HumanMessage(content=prompt))

    return {
        "messages": history,
        "tasks": st.session_state.get("tasks", []),
        "next_task_id": st.session_state.get("task_next_id", 1),
    }


# ============================================================
# ФУНКЦІЯ ДЛЯ АГЕНТА
# ============================================================
def get_agent_response(prompt: str):
    agent = get_langgraph_agent(api_key, MODEL_NAME)

    # ВАЖЛИВО: thread_id потрібен для збереження/відновлення state між викликами
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    state = get_agent_state(prompt)

    try:
        result = agent.invoke(state, config)
        final_message = result["messages"][-1]
        text = extract_response_text(final_message)
        debug = extract_tools_debug(result["messages"])
        sync_tasks_from_agent(result)
        return text, debug, result
    except Exception as e:
        return f"❌ **Помилка агента:** {str(e)}", [], None

# ============================================================
# ГОЛОВНА ОБЛАСТЬ: ЧАТ
# ============================================================
chat_col, task_col = st.columns([3, 1], gap="large")

with chat_col:
    st.subheader("Чат")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Введіть ваше повідомлення..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        if "Агент" in mode:
            with st.chat_message("assistant"):
                with st.spinner("🤔 Агент думає..."):
                    response_text, debug, result = get_agent_response(prompt)
                    sync_tasks_from_agent(result)
                    st.markdown(response_text)

                    if show_agent_debug and debug:
                        with st.expander("Debug: tool calls / results", expanded=False):
                            st.json(debug)

            response = response_text
        else:
            response_text = ""
            with st.chat_message("assistant"):
                for chunk in stream_gemini_response(prompt, st.session_state.messages[:-1]):
                    response_text += chunk
                    st.markdown(chunk)
            response = response_text

        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})

with task_col:
    st.subheader("Завдання")
    filtered_tasks = get_filtered_tasks()
    if filtered_tasks:
        for task in filtered_tasks:
            with st.container():
                render_task_card(task, key_suffix=f"sidebar_{task['id']}")
                st.divider()
    else:
        st.info("Немає задач за поточними фільтрами.")

st.divider()
st.subheader("Kanban")
kanban_columns = st.columns(3)
kanban_statuses = [
    ("open", "🟡 Відкриті"),
    ("in_progress", "🔵 В процесі"),
    ("done", "✅ Виконані"),
]

for (status_key, status_label), column in zip(kanban_statuses, kanban_columns):
    with column:
        st.markdown(f"### {status_label}")
        board_tasks = [task for task in st.session_state.tasks if task.get("status") == status_key]
        if not board_tasks:
            st.info("Порожньо")
        for task in board_tasks:
            with st.expander(task["title"], expanded=False):
                render_task_card(task, key_suffix=f"kanban_{task['id']}")
