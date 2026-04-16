from day13_capstone import app

def ask(question: str, thread_id: str = "user") -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"question": question}, config=config)
    return result