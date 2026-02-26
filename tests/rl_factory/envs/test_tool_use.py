from transformers import AutoTokenizer
from envs.base import Env


def test():
    config = type('Config', (), {
        'config_path': 'envs/configs/calculator.json',
        'step_token': '\n'
    })
    env = Env(config)
    tokenizer = AutoTokenizer.from_pretrained('/your/path/to/Qwen/Qwen2.5-7B-Instruct')

    response_action = """
Hello!
<actions>
    <action>
        <args>
            {"expression": "1+1"}
        </args>
    </action>
    <action>
        <name>calculator</name>
        <args>
            {"expressions": "1+2"}
        </args>
    </action>
</actions>
"""
    response_answer = f"""
Hello!
<answer>
2
</answer>
"""
    env.step([response_action, response_answer], tokenizer)

if __name__ == '__main__':
    test()
