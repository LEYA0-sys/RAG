from openai import OpenAI


class CHAT_MODEL:
    def __init__(self, api_key, base_url,model_name):
        self.llm = OpenAI(
            api_key=api_key, base_url=base_url
        )
        self.model_name = model_name

    def chat(self, user_prompt):
        completion = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[
                # {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        response = completion.choices[0].message.content
        return response

def main():
    api_key = ""  # 你的 api key， 模型供应商都会提供
    base_url = "" # 请求URL, 如 https://open.bigmodel.cn/api/paas/v4 , 模型供应商都会提供
    model_name = "" # 需要调用的模型, 如 glm-4-0520 ， 模型供应商都会提供。
    chat_model = CHAT_MODEL(api_key=api_key, base_url=base_url, model_name=model_name)
    prompt = "What is your name?"
    response = chat_model.chat(user_prompt=prompt)
    print(response)

if __name__ == "__main__":
    main()