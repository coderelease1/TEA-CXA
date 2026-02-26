import re
import json
import string
import torch
import sys

from typing import List
from PIL import Image
from .mmbase import MMEnv
import random

class VisionEnv(MMEnv):
    def __init__(self, config, centralized_actor=None):
        """Initialize vision environment with configuration"""
        super().__init__(config, centralized_actor)
        self.use_verify_tool = False
        


    def get_step_reward(self, responses, format_score=0.1):
        """Calculate step reward based on tool usage validity"""
        step_reward = []
    
        for response in responses:
            temp_action, temp_tool_list = self.tool_manager.parse_response(response_content=response)
            if temp_action == 'answer':
                step_reward.append(torch.nan)  # No reward for direct answer
            else:
                if temp_tool_list[0]['name'] == '<empty>':
                    step_reward.append(-0.5 * format_score)  # Penalty for empty tool
                else:
                    fail_number = 0
                    for i in range(len(temp_tool_list)):
                        if temp_tool_list[i]['name'] == '<error>':
                            fail_number += 1
                    # Calculate reward based on success/failure ratio
                    step_rew = ((len(temp_tool_list) - 2 * fail_number) / len(temp_tool_list)) * format_score
                    step_reward.append(step_rew)
        return step_reward

    @staticmethod
    def _normalize_answer(s):
        """Normalize answer string by removing articles/punctuation/extra spaces"""
        s = re.sub(r"\b(an|the)\b", " ", s.lower())#a|
        s = "".join(ch for ch in s if ch not in string.punctuation)
        return " ".join(s.split())

    @staticmethod
    def _extract_answer(solution_str):
        """Extract answer content from solution string"""
        #solution_str = re.sub(r'<think>.*?</think>', '', solution_str, flags=re.DOTALL)
        # 找到最后一次出现 'assistant' 的位置（不区分大小写可改用 lower() 或 re.search with flags）
        """ idx = solution_str.rfind("assistant\n")
        if idx != -1:
            search_region = solution_str[idx + len("assistant\n"):]  # 只在 'assistant' 之后搜索
        else:
            return None """
        search_region = solution_str
        matches = re.findall(r'<answer>(.*?)</answer>', search_region, re.DOTALL)
        return matches[-1].strip() if matches else None  # Get last answer

    @staticmethod
    def _check_tags_valid(text, tag_pattern):
        """Validate proper tag nesting and formatting"""
        tags = re.findall(tag_pattern, text)
        stack = []
        for tag in tags:
            if tag.startswith('</'):
                if not stack or stack[-1] != tag[2:-1]:
                    return False
                stack.pop()
            else:
                stack.append(tag[1:-1])
        return len(stack) == 0  # Check balanced tags

    @staticmethod
    def _is_valid_json(text):
        """Validate JSON format correctness"""
        try:
            json.loads(text)
            return True
        except json.JSONDecodeError:
            return False

    def _compute_format_score(self, solution_str, format_score):
        """Calculate formatting score for answer/tool_call tags"""
        # Check answer tags
        answer_score = format_score if self._check_tags_valid(solution_str, r"</?answer>") else -format_score
        
        # Check tool_call tags
        if not self._check_tags_valid(solution_str, r"</?tool_call>"):
            return answer_score - 0.5 * format_score
            
        tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", solution_str, re.DOTALL)
        #if not tool_calls:
        #    return answer_score + format_score #- format_score  # Penalty for no tool calls
            
        # Count valid JSON tool calls
        valid_calls = sum(1 for call in tool_calls if self._is_valid_json(call.strip()))
        total_calls = len(tool_calls)
        
        # Calculate tool score with penalty for excessive calls
        tool_score = 2 * format_score * valid_calls / total_calls - format_score
        #if total_calls > 2:
        #    tool_score -= 0.5 * format_score
        tool_count_penalty = -format_score if total_calls < 2 else 0.0    
        return answer_score + (tool_score if total_calls > 0 else 0) + tool_count_penalty
    
    # 定位最后一个 "assistant\n" 的位置（用于判断是否至少两轮对话以及标签是否出现在第二轮之后）
    @staticmethod
    def _last_user_sep_index(text: str) -> int:
        """Return index of the last occurrence of 'assistant\\n'; -1 if not found"""
        return text.rfind("assistant\n")

    # 用于检测成对标签的平衡性（不区分层级，仅做成对数量与顺序校验）
    @staticmethod
    def _has_balanced_pair(text: str, tag_name: str) -> bool:
        """Check if <tag_name>...</tag_name> pairs are balanced in order"""
        opens = [m.start() for m in re.finditer(fr"<{tag_name}>", text)]
        closes = [m.start() for m in re.finditer(fr"</{tag_name}>", text)]
        if len(opens) != len(closes):
            return False
        # each open must appear before its corresponding close in order
        for i in range(len(opens)):
            if opens[i] > closes[i]:
                return False
        return True

    # 新：根据新规则分别计算部分分数的函数
    def _compute_format_subscores(self, solution_str: str, format_score: float):
        """
        Return a dict with four components:
         - ans_fmt_score: 0 or format_score
         - tool_fmt_score: 0 or valid_calls/total_calls*format_score
         - recap_fmt_score: 0 or format_score
        """
        subscores = {
            "ans_fmt_score": 0.0,
            "tool_fmt_score": 0.0,
            "recap_fmt_score": 0.0,
        }

        # 2) 答案格式分数：至少两轮对话，且最后一轮（最后一个 "assistant\n" 之后）包含 <answer>...</answer>
        last_assistant_idx = self._last_user_sep_index(solution_str)
        if last_assistant_idx != -1:
            tail = solution_str[last_assistant_idx + len("assistant\n") :]
            if re.search(r"<answer>.*?</answer>", tail, re.DOTALL):
                subscores["ans_fmt_score"] = format_score

        # 3) 调用工具格式分数：
        #check tool call contents
        tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", solution_str, re.DOTALL)

        penalty = 0  # 发现特定错误时的次数
        for call in tool_calls:
            call_str = call.strip()

            # 判定是否为特定错误：vision-vqa_* 使用了 img_ids 列表而非 img_id
            specific_error = False
            try:
                data = json.loads(call_str)
                if isinstance(data, dict):
                    name = data.get("name", "")
                    args = data.get("arguments", {})
                    if isinstance(args, dict):
                        # 针对 vision-vqa_medgemma 与 vision-vqa_lingshu
                        if name in ("vision-vqa_medgemma", "vision-vqa_lingshu"):
                            # 错误情形：存在 img_ids 且为列表；并且缺少正确的 img_id 或 img_id 不是标量
                            if "img_ids" in args:
                                specific_error = True
                            # 也可视为错误：img_id 存在但为列表（不是标量）
                            if "img_id" in args and isinstance(args.get("img_id"), list):
                                specific_error = True
            except Exception:
                # 若不是合法 JSON，交由 is_valid_json 处理，这里不增加 specific_error
                pass
        
            if specific_error:
                penalty += 1

        # 条件：至少两轮对话；<tool_call> 成对平衡；总数 >= 2；然后 tool_fmt = (valid_calls / total_calls) * format_score
        if last_assistant_idx != -1:
            # 成对性检查（不允许不配对）
            if self._has_balanced_pair(solution_str, "tool_call"):
                tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", solution_str, re.DOTALL)
                total_calls = len(tool_calls)

                if total_calls >= 2:#2
                                       
                    valid_calls = sum(1 for call in tool_calls if self._is_valid_json(call.strip()))
                    subscores["tool_fmt_score"] = (valid_calls / total_calls) * format_score
                    
        if penalty > 0:
            subscores["tool_fmt_score"] = subscores["tool_fmt_score"] - 0.5

        # 4) recap 格式分数：
        #    (a) 在第一个 "user\n" 之前出现 <think>...</think>
        #    (b) 在第二轮对话或之后（即 "assistant\n" 之后）出现 <recap>...</recap>
        first_user_idx = solution_str.find("user\n")
        has_think_before_first_user = False
        if first_user_idx != -1:
            head = solution_str[:first_user_idx]
            if re.search(r"<think>.*?</think>", head, re.DOTALL):
                has_think_before_first_user = True

        if last_assistant_idx != -1 and has_think_before_first_user:
            subscores["recap_fmt_score"] = format_score
        """ if last_assistant_idx != -1 and has_think_before_first_user:
            tail = solution_str[last_assistant_idx + len("assistant\n") :]

            recap_match = re.search(r"<recap>(.*?)</recap>", tail, re.DOTALL)
            if recap_match:
                recap_content = recap_match.group(1)
                if ("tool" not in recap_content) and ("report" not in recap_content):
                    subscores["recap_fmt_score"] = format_score
            #if re.search(r"<recap>.*?</recap>", tail, re.DOTALL):
            #    subscores["recap_fmt_score"] = format_score """

        return subscores
    
    def _compute_score_with_rules(self, data, tokenizer, if_val=False, verbose=False):
        """Compute final scores using exact match and format rules"""
        format_score = 0.0 if if_val else 0.1  # Disable format score in validation mode
        scores = []
        
        if verbose:
            print(f"\n=== start evaluating {len(data)} samples ===", file=sys.stderr)
        
        for i, data_item in enumerate(data):
            processed_data = self._process_data(data_item=data_item, tokenizer=tokenizer)
            ground_truth = processed_data['ground_truth']
            ground_truth = [ground_truth]
            response_str = processed_data['response_str']
            
            # Extract answer and calculate format score
            answer = self._extract_answer(response_str)
            #format_score_val = self._compute_format_score(response_str, format_score)
            subscores = self._compute_format_subscores(response_str, format_score)
            
            # Determine final score
            if answer is None:
                #score = -format_score + 0.5 * format_score_val
                ans_score = 0.0
                result = "No Ans"
            else:
                normalized_answer = self._normalize_answer(answer)
                is_correct = any(self._normalize_answer(truth) == normalized_answer 
                               for truth in ground_truth)
                if is_correct:
                    ans_score = 1.0 #score = 1.0 + 0.5 * format_score_val  # Base score for correct answer
                    result = "Correct"
                else:
                    ans_score = 0.0 #score = format_score_val  # Only format score
                    result = "Wrong"
            score = ans_score + subscores["ans_fmt_score"] + subscores["tool_fmt_score"] + subscores["recap_fmt_score"]
            if verbose:
                print(f"\n#{i+1}: {result} | score: {score:.3f}", file=sys.stderr)
                print(f"Q: {processed_data.get('prompt_str', '')[:200]}...", file=sys.stderr)
                print(f"A: {answer or 'extraction failure'}", file=sys.stderr)
                print(f"std: {ground_truth}", file=sys.stderr)
            
            scores.append([score])
            
            do_print = random.randint(1, 16) == 1           
            if do_print:
                print(f"--------------------------------")
                #print(f"prompt_str: {processed_data['prompt_str']}")
                print(f"Golden answers: {ground_truth}")
                print(f"Solution string: {response_str}")
                print(f"Score: {score}")
        
        if verbose:
            avg_score = sum(s[0] for s in scores) / len(scores)
            print(f"\n=== Avg Score: {avg_score:.3f} ===", file=sys.stderr)

        return scores
