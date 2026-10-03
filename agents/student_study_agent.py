"""
Benchmark Agent: Autonomous Student AI Study & Exam Prep Assistant
A highly relatable multi-step agent with 4 specialized student tools:
1. web_search: Searches web/notes for topic summaries
2. flashcard_generator: Generates active recall Q&A flashcards
3. quiz_evaluator: Generates multiple-choice practice exam questions
4. study_schedule_planner: Calculates pomodoro schedules based on exam dates

Uses Google Gemini via standard OpenAI client.
Notice: Only a single standard `import blackbox.sdk` is included at the top.
No wrapper functions, no modified calls, pure standard LLM agent code!
"""

import os
import json
import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Simply importing blackbox automatically routes all calls through the proxy!
import blackbox.sdk

load_dotenv()

# Standard OpenAI client pointing to Gemini
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy_key"
client = OpenAI(
    api_key=api_key,
    base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
)


# ── 4 RELATABLE STUDENT TOOLS ─────────────────────────────────────────────────

STUDENT_KNOWLEDGE = {
    "recursion": "Recursion in programming is a technique where a function calls itself to solve smaller instances of the same problem. Base case prevents infinite loop.",
    "operating systems": "An operating system manages hardware, memory allocation, process scheduling, and file systems. Examples include Linux, Windows, macOS.",
    "machine learning": "Machine learning focuses on algorithms that learn patterns from training data to make predictions, covering supervised, unsupervised, and reinforcement learning.",
    "dbms": "A Database Management System (DBMS) stores, manages, and queries structured data with ACID properties and relational integrity.",
}


def search_study_notes(topic: str) -> str:
    """Tool 1: Searches subject notes and textbook summaries."""
    t = topic.lower()
    for key, text in STUDENT_KNOWLEDGE.items():
        if key in t or t in key:
            return json.dumps({"topic": key, "summary": text})
    return json.dumps({"topic": topic, "summary": f"{topic} is a core academic topic requiring focused concept revision and problem practice."})


def generate_flashcards(topic: str, difficulty: str = "medium") -> str:
    """Tool 2: Creates active-recall flashcards for quick revision."""
    flashcards = [
        {"q": f"What is the fundamental principle behind {topic}?", "a": f"Core concept definition and mechanism of {topic}."},
        {"q": f"What is a common mistake students make in {topic} exams?", "a": "Forgetting boundary conditions and practical edge cases."},
        {"q": f"How do you apply {topic} in practical real-world projects?", "a": "By structuring modular components and verifying outputs step-by-step."}
    ]
    return json.dumps({"topic": topic, "difficulty": difficulty, "cards": flashcards})


def generate_practice_quiz(topic: str, num_questions: int = 2) -> str:
    """Tool 3: Generates multiple-choice exam questions with answer keys."""
    quiz = [
        {
            "question": f"Which statement best describes the primary advantage of {topic}?",
            "options": ["A) Reduces execution overhead", "B) Improves structural elegance and problem decomposition", "C) Bypasses hardware constraints", "D) None of the above"],
            "correct_answer": "B"
        },
        {
            "question": f"When troubleshooting an issue in {topic}, what should be inspected first?",
            "options": ["A) The base condition or input validation", "B) Network bandwidth", "C) Screen resolution", "D) Operating system version"],
            "correct_answer": "A"
        }
    ]
    return json.dumps({"topic": topic, "questions": quiz[:num_questions]})


def create_study_schedule(days_until_exam: int, daily_hours: float) -> str:
    """Tool 4: Generates a prioritized Pomodoro study timetable."""
    total_study_time = days_until_exam * daily_hours
    pomodoro_sessions = int((total_study_time * 60) // 30)
    schedule = {
        "days_left": days_until_exam,
        "daily_hours": daily_hours,
        "total_study_hours": total_study_time,
        "recommended_pomodoros": pomodoro_sessions,
        "plan": [
            "Phase 1: Concept Mastery & Flashcards Review (40% of time)",
            "Phase 2: Practice Exam Questions & Debugging (40% of time)",
            "Phase 3: Final Mock Test & Formula Cheat Sheet Review (20% of time)"
        ]
    }
    return json.dumps(schedule)


STUDENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_study_notes",
            "description": "Searches textbook notes and subject concept summaries.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string", "description": "Subject or topic name"}},
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_flashcards",
            "description": "Generates active-recall flashcards with questions and answers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]}
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_practice_quiz",
            "description": "Creates multiple-choice practice quiz questions with answer keys.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "num_questions": {"type": "integer"}
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_study_schedule",
            "description": "Generates an organized Pomodoro revision schedule given days remaining and daily hours.",
            "parameters": {
                "type": "object",
                "properties": {
                    "days_until_exam": {"type": "integer"},
                    "daily_hours": {"type": "number"}
                },
                "required": ["days_until_exam", "daily_hours"]
            }
        }
    }
]


# ── INTERACTIVE AGENT EXECUTION ───────────────────────────────────────────────

def run_study_agent():
    print("\n" + "=" * 70)
    print("  📚 Autonomous Student AI Study & Exam Prep Assistant")
    print("  Tools: Search Notes | Flashcard Creator | Practice Quiz | Study Timetable")
    print("=" * 70)

    prompt = input("\nEnter your study request (or press Enter for default): ").strip()
    if not prompt:
        prompt = (
            "I have an exam on 'Recursion' in 4 days. I can study 3 hours each day.\n"
            "1. Search notes for a quick concept summary of Recursion.\n"
            "2. Generate flashcards for quick revision.\n"
            "3. Create a 2-question multiple-choice practice quiz.\n"
            "4. Build me a structured study timetable."
        )

    print(f"\n🚀 Agent starting study workflow for query:\n{prompt}\n")

    messages = [
        {"role": "system", "content": "You are a friendly and structured personal AI tutor. Use all available tools step-by-step to assist the student with revision, testing, and scheduling."},
        {"role": "user", "content": prompt}
    ]

    max_steps = 8
    for step in range(max_steps):
        print(f"--- Step {step + 1} ---")
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=messages,
            tools=STUDENT_TOOLS,
        )

        choice = response.choices[0]
        message = choice.message
        messages.append(message)

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                fn_name = tc.function.name
                args = json.loads(tc.function.arguments)
                print(f"  ⚡ Tool Call: `{fn_name}`")
                print(f"  📥 Arguments: {args}")

                if fn_name == "search_study_notes":
                    res = search_study_notes(args.get("topic", ""))
                elif fn_name == "generate_flashcards":
                    res = generate_flashcards(args.get("topic", ""), args.get("difficulty", "medium"))
                elif fn_name == "generate_practice_quiz":
                    res = generate_practice_quiz(args.get("topic", ""), args.get("num_questions", 2))
                elif fn_name == "create_study_schedule":
                    res = create_study_schedule(int(args.get("days_until_exam", 3)), float(args.get("daily_hours", 2.0)))
                else:
                    res = json.dumps({"error": f"Unknown tool {fn_name}"})

                print(f"  📤 Tool Result: {res}\n")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": res
                })
        else:
            print("\n" + "=" * 70)
            print("🏁 Final Exam Prep Plan:")
            print("=" * 70)
            print(f"{message.content}\n")
            break


if __name__ == "__main__":
    run_study_agent()
