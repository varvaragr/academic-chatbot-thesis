import os
import chainlit as cl
import httpx


# Config

API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:5000/ask")

# Prevent the Chainlit server from freezing if the backend is slow.
HTTP_TIMEOUT_S = float(os.getenv("RAG_API_TIMEOUT", "60"))


@cl.on_chat_start
async def start():
    await cl.Message(
        content=(
            "👋 Γεια σου! Είμαι ο ακαδημαϊκός βοηθός του Τμήματος. "
            "Ρώτησέ με οτιδήποτε για το τμήμα/σπουδές/γραμματεία κτλ."
        )
    ).send()


@cl.on_message
async def main(message: cl.Message):
    user_text = (message.content or "").strip()
    greetings = ["hi", "hello", "hey", "γεια", "καλησπέρα", "καλημέρα", "καληνύχτα"]

    if not user_text:
        await cl.Message(content="⚠️ Γράψε μια ερώτηση για να μπορέσω να απαντήσω.").send()
        return

    if any(g in user_text.lower() for g in greetings):
        await cl.Message(content="👋 Γεια! Πώς μπορώ να βοηθήσω;").send()
        return

    # Show progress in the UI
    msg = cl.Message(content="⏳ Ψάχνω στις αποθηκευμένες πληροφορίες...")
    await msg.send()

    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_S) as client:
            r = await client.post(API_URL, json={"question": user_text})

        if r.status_code == 200:
            answer = (r.json() or {}).get("answer") or "⚠️ Δεν επέστρεψε απάντηση ο server."
        else:
            body = (r.text or "").strip()
            answer = f"⚠️ Backend error: {r.status_code}. {body}"

    except httpx.ConnectError as e:
        answer = (
            "⚠️ Δεν μπόρεσα να συνδεθώ με το backend RAG API.\n\n"
            f"• API_URL: {API_URL}\n"
            f"• Λεπτομέρειες: {e}"
        )
    except httpx.ReadTimeout:
        answer = (
            "⚠️ Το backend άργησε να απαντήσει (timeout).\n\n"
            "Δοκίμασε ξανά ή δες τα logs του rag_api.py μήπως κολλάει στο retrieval/LLM."
        )
    except Exception as e:
        answer = f"⚠️ Σφάλμα στην εφαρμογή Chainlit: {type(e).__name__}: {e}"

    msg.content = answer
    await msg.update()
