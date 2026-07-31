from pathlib import Path
from llama_cpp import Llama
from portablellm.utils import application_path

class LLMService:

    def __init__(self, config):
        self.config = config
        self.models = {}

    def load_model(self, model_number: int):
        if model_number in self.models:
            return

        model_name = self.config.get_model(model_number)
        model_path = ( application_path() / "portablellm" / "models" / model_name)

        print(f"Loading model {model_number}:")
        print(model_path)

        self.models[model_number] = Llama(
            model_path=str(model_path),
            n_ctx=self.config.context_size,
            verbose=False,
        )

        print(f"Model {model_number} loaded.")

    def load_prompt(self, filename: str) -> str:
        prompt_path = ( application_path() / "portablellm" / "prompts" / filename )

        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read()

    def load_prompt_template(self, filename: str, values: dict[str, str]) -> str:
        prompt = self.load_prompt(filename)

        for key, value in values.items():
            prompt = prompt.replace(f"{{{key}}}", value)

        return prompt

    def generate(self, prompt: str, prompt_file: str, model: int, prompt_information: str, ) -> str:
        self.load_model(model)
        llm = self.models[model]

        if not prompt_information or not prompt_information.strip():
            system_prompt = self.load_prompt_template(
                            prompt_file,
                            {
                                "question": prompt
                            }
                        )
        else:
            system_prompt = prompt_information

        print(
            f"System Prompt: {system_prompt[:20] + ("..." if len(system_prompt) > 20 else "")}\n"
            f"Content: {prompt[:20] + ("..." if len(prompt) > 20 else "")}"
        )

        result = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        returnResult = result["choices"][0]["message"]["content"]

        if "</think>" in returnResult:
            returnResult = returnResult.split("</think>", 1)[1].strip()

        print(
            f"Result: {returnResult[:20] + ("..." if len(returnResult) > 20 else "")}"
        )

        return returnResult