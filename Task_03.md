### 第四章 智能体经典范式构建

#### 4.1 环境准备和基础工具定义

```python
# LLM_client.py
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables from a .env file if present
load_dotenv()

class HeloAgentsLLM:
    """
    用于兼容调用api
    """
    def __init__(self, model: str = None, apikey: str = None, baseUrl: str = None, timeout: int = None):
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.apikey = apikey or os.getenv("LLM_API_KEY")
        self.base_url = baseUrl or os.getenv("LLM_BASE_URL")
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))

        if not all([self.model, self.apikey, self.base_url]):
            raise ValueError("Model, API key, and Base URL must be provided either as arguments or environment variables.")
        
        self.client = OpenAI(api_key=self.apikey, base_url=self.base_url, timeout=self.timeout)
        
    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用接口进行思考
        """
        try:
            response = self.client.chat.completions.create(
                model = self.model,
                messages = messages,
                temperature = temperature,
                stream = True,
            )
            # 处理流式响应
            print("大模型响应成功，开始接收数据流...")
            collected_contend = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_contend.append(content)
            print()
            return "".join(collected_contend)
        except Exception as e:
            print(f"调用大模型接口出错: {e}")
            return None
        
if __name__ == '__main__':
    try:
        llmclient = HeloAgentsLLM()

        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]

        print("开始调用大模型接口...")
        responseText = llmclient.think(exampleMessages)
        if responseText:
            print("大模型响应成功:")
            print(responseText)
    except ValueError as e:
        print(f"初始化大模型客户端出错: {e}")

```

#### 4.2 ReAct

ReAct（智能体范式）= Reason + Act （推理 + 行动）= 思考 - 行动 - 观察

- 纯思考型：引导模型进行思考但是无法和环境交互，容易产生幻觉（思维链）
- 纯行动型：模型直接输出要执行的动作，但是缺乏规划和纠错能力

 流程：

- 思考：分析当前情况，分解任务，制定下一步计划，反思上一步结果
- 行动：执行的具体动作，通常是调用一个外部工具
- 观察：执行Action之后从外部工具返回的结果（搜索结果的摘要或api得返回值）

智能体将不断重复循环（*Thought -> Action -> Observation*），根据不断返回的上下文直到**Thought认为返回了最终答案**。

***推理使得行动更具目的性，而行动则为推理提供了事实依据***

![image-20251114110512414](/Users/yyyy/Library/Application Support/typora-user-images/image-20251114110512414.png)

适用于：需要外部知识的任务、需要精确计算的任务、需要与api交互的任务

**实现一个搜索工具的核心逻辑：**

- 名称：一个简洁、唯一的标识符
- 描述：一段清晰的自然语言描述用来描述工具的用途（key），大模型会根据这段描述来判断什么时候使用什么工具
- 执行逻辑

```
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下：
{tools}

请严格按照以下格式进行回应：

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一：
- '{tool_name}[{tool_input}]'，调用一个可用工具。
- 'Finish[最终答案]': 当你任务已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 finish（answer=“...“）来输出最终答案。

现在请开始解决以下问题：
Question：{question}
History:{history}
"""
```

这一个提示词模版定义了智能体和LLM之间的交互的规范：

- 角色定义
- 工具清单
- 格式规约
- 动态上下文

**ReAct的特点、局限性、调试技巧：**

（1）ReAct 的主要特点

1. **高可解释性**：ReAct 最大的优点之一就是透明。通过 `Thought` 链，我们可以清晰地看到智能体每一步的“心路历程”——它为什么会选择这个工具，下一步又打算做什么。这对于理解、信任和调试智能体的行为至关重要。
2. **动态规划与纠错能力**：与一次性生成完整计划的范式不同，ReAct 是“走一步，看一步”。它根据每一步从外部世界获得的 `Observation` 来动态调整后续的 `Thought` 和 `Action`。如果上一步的搜索结果不理想，它可以在下一步中修正搜索词，重新尝试。
3. **工具协同能力**：ReAct 范式天然地将大语言模型的推理能力与外部工具的执行能力结合起来。LLM 负责运筹帷幄（规划和推理），工具负责解决具体问题（搜索、计算），二者协同工作，突破了单一 LLM 在知识时效性、计算准确性等方面的固有局限。

（2）ReAct 的固有局限性

1. **对LLM自身能力的强依赖**：ReAct 流程的成功与否，高度依赖于底层 LLM 的综合能力。如果 LLM 的逻辑推理能力、指令遵循能力或格式化输出能力不足，就很容易在 `Thought` 环节产生错误的规划，或者在 `Action` 环节生成不符合格式的指令，导致整个流程中断。
2. **执行效率问题**：由于其循序渐进的特性，完成一个任务通常需要多次调用 LLM。每一次调用都伴随着网络延迟和计算成本。对于需要很多步骤的复杂任务，这种串行的“思考-行动”循环可能会导致较高的总耗时和费用。
3. **提示词的脆弱性**：整个机制的稳定运行建立在一个精心设计的提示词模板之上。模板中的任何微小变动，甚至是用词的差异，都可能影响 LLM 的行为。此外，并非所有模型都能持续稳定地遵循预设的格式，这增加了在实际应用中的不确定性。
4. **可能陷入局部最优**：步进式的决策模式意味着智能体缺乏一个全局的、长远的规划。它可能会因为眼前的 `Observation` 而选择一个看似正确但长远来看并非最优的路径，甚至在某些情况下陷入“原地打转”的循环中。

（3）调试技巧

当你构建的 ReAct 智能体行为不符合预期时，可以从以下几个方面入手进行调试：

- **检查完整的提示词**：在每次调用 LLM 之前，将最终格式化好的、包含所有历史记录的完整提示词打印出来。这是追溯 LLM 决策源头的最直接方式。
- **分析原始输出**：当输出解析失败时（例如，正则表达式没有匹配到 `Action`），务必将 LLM 返回的原始、未经处理的文本打印出来。这能帮助你判断是 LLM 没有遵循格式，还是你的解析逻辑有误。
- **验证工具的输入与输出**：检查智能体生成的 `tool_input` 是否是工具函数所期望的格式，同时也要确保工具返回的 `observation` 格式是智能体可以理解和处理的。
- **调整提示词中的示例 (Few-shot Prompting)**：如果模型频繁出错，可以在提示词中加入一两个完整的“Thought-Action-Observation”成功案例，通过示例来引导模型更好地遵循你的指令。
- **尝试不同的模型或参数**：更换一个能力更强的模型，或者调整 `temperature` 参数（通常设为0以保证输出的确定性），有时能直接解决问题。

#### 4.3 Plan-and-Solve

- 规划阶段：将问题分解，并制定出一个清晰、分步骤的行动计划
- 执行阶段：严格按照计划中的步骤，逐一执行

适用于：多步数学应用题、需要整个多个信息源的报告撰写、代码生成任务

#### 4.4 Reflection

**核心思想：**为Agent引入一种事后的自我校正循环，对自己的工作进行迭代优化

流程：执行 -> 反思 -> 优化

- 执行：使用ReAct或者Plan-and-Solve尝试完成任务
- 反思：调用一个独立的、或者带有特殊提示词的大语言模型实例来评审，从多个维度进行评估
  - 事实性错误
  - 逻辑漏洞
  - 效率问题
  - 遗漏信息
- 优化：将初稿和反馈作为新的上下文，根据反馈内容进行修正

![image-20251114195802240](/Users/yyyy/Library/Application Support/typora-user-images/image-20251114195802240.png)

**总结**

![image-20251114200400585](/Users/yyyy/Library/Application Support/typora-user-images/image-20251114200400585.png)

---

#### 习题

> **提示**:部分习题没有标准答案，重点在于培养学习者对智能体范式设计的综合理解和实践能力。

1. 本章介绍了三种经典的智能体范式:`ReAct`、`Plan-and-Solve` 和 `Reflection`。请分析:

   - 这三种范式在"思考"与"行动"的组织方式上有什么本质区别？
   - 如果要设计一个"智能家居控制助手"（需要控制灯光、空调、窗帘等多个设备，并根据用户习惯自动调节），你会选择哪种范式作为基础架构？为什么？
   - 是否可以将这三种范式进行组合使用？若可以，请尝试设计一个混合范式的智能体架构，并说明其适用场景。

   ```
   - ReAct:Thought-Action-Observation
   - Plan-and-Solve:Plan-Solve
   - Reflection:执行后会对结果进行反思，然后进行优化
   
   - ReAct（决策）+Reflection（学习习惯）
   ```

2. 在4.2节的 `ReAct` 实现中，我们使用了正则表达式来解析大语言模型的输出（如 `Thought` 和 `Action`）。请思考:

   - 当前的解析方法存在哪些潜在的脆弱性？在什么情况下可能会失败？
   - 除了正则表达式，还有哪些更鲁棒的输出解析方案？
   - 尝试修改本章的代码，使用一种更可靠的输出格式，并对比两种方案的优缺点

   ```
   - 对格式极度敏感、跨行内容解析失败、内容里包含类似标记
   - Json格式、block格式
   ```

3. 工具调用是现代智能体的核心能力之一。基于4.2.2节的 `ToolExecutor` 设计，请完成以下扩展实践:

   > **提示**:这是一道动手实践题，建议实际编写代码

   - 为 `ReAct` 智能体添加一个"计算器"工具，使其能够处理复杂的数学计算问题（如"计算 `(123 + 456) × 789/ 12 = ?` 的结果"）
   - 设计并实现一个"工具选择失败"的处理机制:当智能体多次调用错误的工具或提供错误的参数时，系统应该如何引导它纠正？
   - 思考:如果可调用工具的数量增加到$50$个甚至$100$个，当前的工具描述方式是否还能有效工作？在可调用工具数量随业务需求显著增加时，从工程角度如何优化工具的组织和检索机制？

   ```python
   import math
   
   def calculator_tool(expr: str):
       # 简易安全计算环境，只暴露 math 里的一些函数 & 基本运算
       allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
       allowed_names.update({
           "__builtins__": {},
       })
       try:
           result = eval(expr, allowed_names, {})
           return f"计算结果: {result}"
       except Exception as e:
           return f"计算出错: {e}"
   
   # 提示词      
   可用工具如下：
   {tools}
   
   例如，当你需要计算复杂数学表达式时，请使用 Calculator 工具
   
   - 工具分组 + 层次化描述
   - 基于 embedding 的工具检索
   - 结构化工具描述
   ```

4. `Plan-and-Solve` 范式将任务分解为"规划"和"执行"两个阶段。请深入分析:

   - 在4.3节的实现中，规划阶段生成的计划是"静态"的（一次性生成，不可修改）。如果在执行过程中发现某个步骤无法完成或结果不符合预期，应该如何设计一个"动态重规划"机制？
   - 对比 `Plan-and-Solve` 与 `ReAct`:在处理"预订一次从北京到上海的商务旅行（包括机票、酒店、租车）"这样的任务时，哪种范式更合适？为什么？
   - 尝试设计一个"分层规划"系统:先生成高层次的抽象计划，然后针对每个高层步骤再生成详细的子计划。这种设计有什么优势？

   ```python
   """
   在执行时，对每个步骤维护一个状态：
   	PENDING / RUNNING / DONE / FAILED
   一旦某步失败，比如：
   	机票预订接口报错：该日期无票。
   触发一个“重规划调用”：
   	把当前信息喂给规划模型：
   	重新得到一个“更新计划”（可以是剩余步骤，也可以是全局新计划）。
   用新的计划替换后续步骤，继续执行。
   这就是典型的 Plan–Execute–Replan 回路，在自动规划领域很常见。
   """
   
   - plan-and-solve;整体规划效果会更好
   ```

5. `Reflection` 机制通过"执行-反思-优化"循环来提升输出质量。请思考:

   - 在4.4节的代码生成案例中，不同阶段使用的是同一个模型。如果使用两个不同的模型（例如，用一个更强大的模型来做反思，用一个更快的模型来做执行），会带来什么影响？
   - `Reflection` 机制的终止条件是"反馈中包含**无需改进**"或"达到最大迭代次数"。这种设计是否合理？能否设计一个更智能的终止条件？
   - 假设你要搭建一个"学术论文写作助手"，它能够生成初稿并不断优化论文内容。请设计一个多维度的Reflection机制，从段落逻辑性、方法创新性、语言表达、引用规范等多个角度进行反思和改进。

   ```
   - 效果提升，但是工程复杂度增加
   - 不合理，多评审者共识
             用多个“虚拟评审角色”：
             逻辑性评审
             语言表达评审
             规范性评审
             
   步骤 1：生成初稿（例如引言+相关工作+方法+实验+结论）。
   步骤 2：在多个维度上分别反思：
       逻辑性 / 结构性
           检查段落是否有清晰的论点→论据→小结。
           检查章节之间是否连贯（比如方法是否真的回答了前面提出的问题）。
       方法创新性
           评估：方法相对已有工作是否有实质差异还是只换皮。
           指出需要增加的对比 / 消融实验。
       语言表达
           学术风格：是否过于口语，是否有模糊词。
           句式是否冗长，是否有重复表达。
       引用规范
           引文格式：是否符合目标期刊/会议的规范。
           引用是否与正文主张匹配（不乱用引用）。
       实现上可以有四个“反思子 Agent”，每个有独立提示词，例如：
       		逻辑性评审提示：
       				你是一位严谨的审稿人，专注于论文的结构与逻辑连贯性……
       		语言表达评审提示：
       				你是一位擅长学术英语润色的编辑……
       		每个维度输出两部分：
       				问题列表。
       				建议修改的具体句子 / 段落。
       		最后由一个“整合 Agent”把这些建议应用到初稿上，生成新版本。           
   ```

6. 提示词工程是影响智能体最终效果的关键技术。本章展示了多个精心设计的提示词模板。请分析:

   - 对比4.2.3节的 `ReAct` 提示词和4.3.2节的 `Plan-and-Solve` 提示词，它们显然存在结构设计上的明显不同，这些差异是如何服务于各自范式的核心逻辑的？
   - 在4.4.3节的 `Reflection` 提示词中，我们使用了"你是一位极其严格的代码评审专家"这样的角色设定。尝试修改这个角色设定（如改为"你是一位注重代码可读性的开源项目维护者"），观察输出结果的变化，并总结角色设定对智能体行为的影响。
   - 在提示词中加入 `few-shot` 示例往往能显著提升模型对特定格式的遵循能力。请为本章的某个智能体尝试添加 `few-shot` 示例，并对比其效果。

7. 某电商初创公司现在希望使用"客服智能体"来代替真人客服实现降本增效，它需要具备以下功能:

   a. 理解用户的退款申请理由

   b. 查询用户的订单信息和物流状态

   c. 根据公司政策智能地判断是否应该批准退款

   d. 生成一封得体的回复邮件并发送至用户邮箱

   e. 如果判断决策存在一定争议（自我置信度低于阈值），能够进行自我反思并给出更审慎的建议

   此时作为该产品的负责人:

   - 你会选择本章的哪种范式（或哪些范式的组合）作为系统的核心架构？
   - 这个系统需要哪些工具？请列出至少3个工具及其功能描述。
   - 如何设计提示词来确保智能体的决策既符合公司利益，又能保持对用户的友好态度？
   - 这个产品上线后可能面临哪些风险和挑战？如何通过技术手段来降低这些风险？

   ```
   - Plan-and-Solve + ReAct + Reflection
   
   - OrderService.get_order(order_id) # 功能：查询用户订单详情：商品信息、价格、支付方式、下单时间等。
   - LogisticsService.get_status(tracking_id) # 功能：查询物流状态：未发货 / 运输中 / 已签收 / 丢件异常 / 拒收等。
   - PolicyService.evaluate_refund(order_info, logistics_status, reason)
   # 功能：根据公司退款政策做初步规则判断：
   # 输出：{ "recommendation": "approve/deny/manual_review", "reason": "..." }
   
   “”“
   决策原则优先级
       明确写出：
           必须严格遵守公司退款政策文档。
           在政策允许范围内，适当倾向用户满意度。
           涉及法律合规时，严格遵守法律法规优先于公司与用户的任何一方利益。
   行为要求
       “不得编造不存在的订单信息、物流状态或政策条款。”
       “所有与订单、物流相关的事实信息必须来自工具调用结果，不能凭空推断。”
   对话风格
       “与用户沟通时，语气保持礼貌、共情，避免指责用户。”
       “在拒绝或部分拒绝退款时，要给出清晰易懂的理由，并尽量提供替代方案（比如优惠券、重新发货）。”
   反思触发
       “当你对是否应批准退款的判断信心不足时（例如，政策条款存在模糊空间，或工具返回的信息不完整），请：
           标明自己的不确定性；
           进行一次内部反思（Reflection），重新检查政策与事实；
           如仍然不确定，建议转交人工客服处理。”
   ”“”
   ```

---

