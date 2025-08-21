#使用多套提示词，添加MRR和Recall指标，生成输出文件.json
from typing import List
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
import random
import concurrent.futures

# 针对每种类型的问题设计专属提示词模板
PROMPT_TEMPLATES = {
    "对比": """你是一名资深保险产品分析师，请基于用户原始问题，生成5个精准的对比性检索问题。要求：
1. 每个问题需明确对比维度（如价格、保障范围、免责条款等）。
2. 覆盖不同保险类型（重疾险、医疗险、寿险等）。
3. 使用结构化表达，例如“A产品与B产品在XX方面的差异是什么？”

原始问题: {question}
示例输出:
1. 重疾险和医疗险在癌症保障范围上有哪些具体差异？
2. 公司A和公司B的意外险保费和赔付比例如何对比？
3. 消费型与返还型重疾险的长期成本孰优孰劣？
4. 高端医疗险与普通医疗险的医院覆盖范围有何不同？
5. 不同保险公司的免责条款中，对“既往症”的定义是否一致？""",

    "流程": """你是一名保险流程专家，请生成5个关于保险办理、理赔或售后流程的检索问题。要求：
1. 问题需分阶段（投保前、中、后）或分场景（线上/线下）。
2. 包含具体操作细节（如所需材料、时效、常见问题）。

原始问题: {question}
示例输出:
1. 线上购买重疾险时需要准备哪些身份和健康证明文件？
2. 保险理赔申请提交后，通常需要多少个工作日完成审核？
3. 如果投保人中途更换工作，如何更新职业类别信息？
4. 退保流程中，如何计算现金价值并减少损失？
5. 通过代理人购买和官网直购的理赔流程有何差异？""",

    "条款": """你是一名保险条款法律顾问，请生成5个聚焦条款细节的检索问题。要求：
1. 突出免责条款、隐性限制、例外情况。
2. 使用法律术语（如“不可抗辩条款”“等待期”）。

原始问题: {question}
示例输出:
1. 重疾险条款中“确诊即赔”与“达到特定状态才赔”的具体区别？
2. 意外险的免责条款是否包含高风险运动导致的伤害？
3. 医疗险中“合理且必要”的医疗费用如何界定？
4. 寿险的不可抗辩条款在哪些情况下可能失效？
5. 保险合同中“等待期”和“观察期”是同一概念吗？""",

    "案例": """你是一名保险案例库负责人，请生成5个基于真实场景的检索问题。要求：
1. 结合典型用户画像（如年龄、职业、健康状况）。
2. 包含争议性案例或法院判决先例。

原始问题: {question}
示例输出:
1. 糖尿病患者购买医疗险被拒赔的常见原因有哪些？
2. 保险公司因“未如实告知”拒赔的案例中，法院如何判决？
3. 自由职业者如何通过保险组合解决养老和医疗问题？
4. 儿童重疾险理赔中，哪些疾病最容易引发纠纷？
5. 车祸后对方全责，自己的意外险还能重复理赔吗？""",

    "定义": """你是一名保险术语标准化专家，请生成5个专业术语解析问题。要求：
1. 覆盖基础术语和行业黑话（如“趸交”“共付比例”）。
2. 提供通俗化解释和英文缩写（如“P2P保险”）。

原始问题: {question}
示例输出:
1. “现金价值”和“保单账户价值”在理财险中有何区别？
2. 保险中的“免赔额”（Deductible）具体如何计算？
3. 什么是“保证续保条款”？哪些产品会提供？
4. “相互保险社”与传统保险公司在运营模式上有何不同？
5. 分红险中的“三差收益”指哪三部分利差？""",

    "其他": """你是一名保险智能助手，请生成5个多角度检索问题。要求：
1. 包含用户常见误解的澄清（如“保险都是骗人的吗？”）。
2. 结合最新行业动态（如互联网保险新规）。

原始问题: {question}
示例输出:
1. 为什么年轻人也需要配置寿险？有哪些高性价比方案？
2. 2023年互联网保险新规对短期健康险有哪些影响？
3. “保险姓保”政策下，理财型保险还值得购买吗？
4. 保险公司破产后，消费者的保单如何得到保障？
5. 如何辨别保险销售话术中的夸大宣传？"""
}

k = 20

class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))

output_parser = LineListOutputParser()

# 加载QA对
qa_file_path = '../QA对/QA4.json'
with open(qa_file_path, 'r', encoding='utf-8') as f:
    qa_data = json.load(f)
queries = [item['question'] for item in qa_data]
answers = [item['answer'] for item in qa_data]
import re
pages = [item['page_num'] for item in qa_data]

output_dir = '../outputs/outputs_4.0/多提示词检索'
os.makedirs(output_dir, exist_ok=True)
output_file_path = os.path.join(output_dir, f'multi_prompt_results_with_mrr_recall.json')

def hybrid_retrieve(query, k=k):
    # 使用向量数据库进行检索
    docs = vector_db.similarity_search(query, k=k)
    return docs

def classify_question(question: str) -> str:
    """
    简单规则判断问题类型
    返回类型如：'对比', '流程', '条款', '案例', '定义', '其他'
    """
    if "区别" in question or "对比" in question or "不同" in question or "差别" in question or "差异" in question:
        return "对比"
    elif "流程" in question or "步骤" in question or "怎么做" in question or "过程" in question:
        return "流程"
    elif "条款" in question or "免责" in question or "条件" in question or "要求" in question:
        return "条款"
    elif "案例" in question or "举例" in question or "例子" in question or "实例" in question:
        return "案例"
    elif "是什么" in question or "定义" in question or "解释" in question or "含义" in question:
        return "定义"
    else:
        return "其他"

def select_prompt_by_type(q_type: str) -> str:
    """
    根据问题类型返回专属prompt模板。
    """
    return PROMPT_TEMPLATES.get(q_type, PROMPT_TEMPLATES["其他"])

def select_prompt_template(query):
    q_type = classify_question(query)
    prompt_str = select_prompt_by_type(q_type)
    return prompt_str, q_type

def multi_prompt_retrieve(query, k=k):
    # 根据问题类型选择prompt模板
    prompt_str, question_category = select_prompt_template(query)
    prompt = PromptTemplate(input_variables=["question"], template=prompt_str)
    llm_chain = prompt | llm | output_parser
    expanded_queries = llm_chain.invoke(query)
    # 对每个扩展问题分别检索，合并去重
    doc_set = {}
    for q in expanded_queries:
        docs = hybrid_retrieve(q, k=k)
        for doc in docs:
            doc_id = doc.metadata.get("uuid", "") + str(doc.metadata.get("page_num", ""))
            if doc_id not in doc_set:
                doc_set[doc_id] = doc
    return list(doc_set.values()), prompt_str, question_category  # 返回检索结果、所用prompt、分类

# 使用多线程处理每个问题
def process_query(args):
    idx, query, answer, correct_page = args
    # 多提示词检索
    raw_docs, used_prompt_template, question_category = multi_prompt_retrieve(query, k=k)
    # 不进行压缩，直接输出检索到的文档
    raw_documents = [
        {
            "uuid": doc.metadata.get("uuid", ""),
            "source_file": doc.metadata.get("source_file", ""),
            "content": doc.page_content,
            "page": str(doc.metadata.get("page_num", "未指定"))
        }
        for doc in raw_docs
    ]
    # 统计检索到的所有页码
    retrieved_pages = [doc.get("page", "") for doc in raw_documents]
    # recall指标
    recall = 1 if correct_page in retrieved_pages else 0
    # MRR指标
    rank = 0
    for i, p in enumerate(retrieved_pages):
        if p == correct_page:
            rank = i + 1
            break
    mrr = 1.0 / rank if rank > 0 else 0.0

    # 生成输出结果，包括检索到的文档、问题类型、使用的提示模板、检索指标等
    return {
        "query": query,
        "answer": answer,
        "correct_page": correct_page,
        "recall": recall,
        "mrr": mrr,
        "retrieved_documents": raw_documents,
        "question_category": question_category,
        "used_prompt_template": used_prompt_template
    }, recall, mrr

args_list = [(idx, query, answer, pages[idx] if idx < len(pages) else "") for idx, (query, answer) in enumerate(zip(queries, answers))]

output_data = []
recall_list = []
mrr_list = []

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_query, args) for args in args_list]
    for future in concurrent.futures.as_completed(futures):
        result, recall, mrr = future.result()
        output_data.append(result)
        recall_list.append(recall)
        mrr_list.append(mrr)

# 保证输出顺序与输入一致（可选）
output_data.sort(key=lambda x: queries.index(x["query"]))

avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0.0
avg_mrr = sum(mrr_list) / len(mrr_list) if mrr_list else 0.0

# 保存整体指标到输出结果
summary = {
    "avg_recall": avg_recall,
    "avg_mrr": avg_mrr
}
output = {
    "summary": summary,
    "results": output_data
}
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

print(f"检索结果已保存到 {output_file_path}")