"""LLM integrations for the QuranChatBot.

This package exposes the public interfaces for generating Qur'an commentary
and pronunciation guides. By default it uses the OpenAI client defined in
``openai_client.py``. If you wish to swap LLM providers or implement
additional functionality, you can do so here and update the imports in
``app/gradio_ui.py`` and ``app/fastapi_app.py`` accordingly.
"""

from .openai_client import generate_quran_commentary, generate_pronunciation_guide  # noqa: F401
