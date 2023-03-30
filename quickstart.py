import asyncio
import signal

import vocode
from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.helpers import create_microphone_input_and_speaker_output
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)
from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import AzureSynthesizerConfig
from vocode.streaming.synthesizer.azure_synthesizer import AzureSynthesizer
from vocode.streaming.transcriber.deepgram_transcriber import DeepgramTranscriber



from dotenv import load_dotenv


load_dotenv()
import os

# load the api keys from the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")


vocode.setenv(
    OPENAI_API_KEY=OPENAI_API_KEY,
    DEEPGRAM_API_KEY=DEEPGRAM_API_KEY,
    AZURE_SPEECH_KEY=AZURE_SPEECH_KEY,
    AZURE_SPEECH_REGION=AZURE_SPEECH_REGION,
)


base_message_sv = "Hej, välkommen hit. Hur mår du idag?"
base_message_english = "Hello, welcome here. How are you feeling today?"
prompt_preamble_sv = "Du är en världsklass psykolog som är fokuserad på att hjälpa människor att lösa sina psykiska hälsoproblem och förbättra sitt välbefinnande."

prompt_preamble_english = "You are a world class psychologist who is focused on helping people resolve their mental health issues and improve their wellbeing."

async def main():
    microphone_input, speaker_output = create_microphone_input_and_speaker_output(
        streaming=True, use_default_devices=False
    )

    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input, endpointing_config=PunctuationEndpointingConfig()
            )
        ),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                initial_message=BaseMessage(text=base_message_english),
                prompt_preamble=prompt_preamble_english,
            ),
        ),
        synthesizer=AzureSynthesizer(
            AzureSynthesizerConfig.from_output_device(speaker_output)
        ),
    )
    await conversation.start()
    print("Conversation started, press Ctrl+C to end")
    signal.signal(signal.SIGINT, lambda _0, _1: conversation.terminate())
    while conversation.is_active():
        chunk = microphone_input.get_audio()
        if chunk:
            conversation.receive_audio(chunk)
        await asyncio.sleep(0)


if __name__ == "__main__":
    asyncio.run(main())
