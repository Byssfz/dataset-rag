"""
RAGAS RAG 评估主脚本

功能：
1. 加载测试数据集（问答对 + 标准答案）
2. 对每条问题运行完整的 RAG 查询流程（query_app）
3. 收集检索上下文（reranked_docs）和生成答案（answer）
4. 用 RAGAS 框架评估 4 项核心指标
5. 输出评估报告到控制台 + JSON 文件

运行方式：
    python -m app.eval.ragas_evaluator

前置条件：
    - Milvus / MongoDB / MinIO 等服务已启动
    - 已有文档导入向量数据库
"""

import argparse
import json
import os
import sys
import types
import uuid
import time

# ============================================================================
# 兼容垫片：RAGAS 0.4.x 从 langchain_community.chat_models.vertexai 导入 ChatVertexAI，
# 但 langchain-community 0.4.x 已 sunset 并移除该模块。
# 在导入 ragas 之前，注入一个空壳模块避免 ImportError。
# ============================================================================
_vertexai_shim = types.ModuleType("langchain_community.chat_models.vertexai")
_vertexai_shim.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_shim

from app.core.logger import logger
from app.conf.ragas_config import ragas_config
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_query_default_state
from app.lm.lm_utils import get_llm_client

# RAGAS 核心导入
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextRecall, FactualCorrectness

# ResponseRelevancy 在不同 RAGAS 版本中名称可能不同，做兼容处理
try:
    from ragas.metrics import ResponseRelevancy
except ImportError:
    try:
        from ragas.metrics import AnswerRelevancy as ResponseRelevancy
    except ImportError:
        ResponseRelevancy = None
        logger.warning("RAGAS 未找到 ResponseRelevancy / AnswerRelevancy 指标，将跳过该指标")


def load_test_dataset(dataset_path: str) -> list[dict]:
    """
    加载测试数据集 JSON 文件

    :param dataset_path: 数据集文件路径
    :return: 问答对列表，每条包含 question / ground_truth / item_name
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"测试数据集不存在: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"已加载测试数据集: {dataset_path}，共 {len(data)} 条问答对")
    return data


def run_single_query(question: str, item_name_hint: str = "") -> dict:
    """
    运行单条 RAG 查询，收集检索上下文和生成答案

    :param question: 用户问题
    :param item_name_hint: 预期商品名（仅用于日志，不影响查询）
    :return: 包含 user_input / response / retrieved_contexts 的字典
    """
    session_id = f"eval_{uuid.uuid4().hex[:8]}"

    state = create_query_default_state(
        session_id=session_id,
        original_query=question,
        is_stream=False
    )

    logger.info(f"  [查询] session={session_id}, question={question[:60]}...")

    try:
        final_state = query_app.invoke(state)
    except Exception as e:
        logger.error(f"  [查询失败] session={session_id}, error={e}")
        return {
            "user_input": question,
            "response": f"[查询异常] {str(e)}",
            "retrieved_contexts": [],
            "session_id": session_id,
            "error": str(e)
        }

    # 从最终状态提取结果
    answer = final_state.get("answer", "")
    reranked_docs = final_state.get("reranked_docs", [])

    # 从 reranked_docs 提取文本内容作为检索上下文
    contexts = []
    for doc in reranked_docs:
        text = doc.get("text", "")
        if text:
            contexts.append(text)

    answer_len = len(answer) if answer else 0
    logger.info(f"  [完成] 检索到 {len(contexts)} 条上下文，生成答案 {answer_len} 字")

    if not contexts:
        logger.warning(f"  [警告] 未检索到上下文，可能是商品名未匹配或向量库为空")

    return {
        "user_input": question,
        "response": answer,
        "retrieved_contexts": contexts,
        "session_id": session_id,
        "error": None
    }


def build_evaluation_samples(test_data: list[dict]) -> list[dict]:
    """
    对测试数据集中的每条问题运行 RAG 查询，组装 RAGAS 评估样本

    :param test_data: 测试数据集
    :return: RAGAS 评估样本列表
    """
    total = len(test_data)
    samples = []

    for i, item in enumerate(test_data, start=1):
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        item_name_hint = item.get("item_name", "")

        print(f"\n[{i}/{total}] 运行查询: {question}")
        if item_name_hint:
            print(f"        预期商品: {item_name_hint}")

        # 运行 RAG 查询
        result = run_single_query(question, item_name_hint)

        # 组装 RAGAS 样本
        sample = {
            "user_input": question,
            "response": result["response"],
            "retrieved_contexts": result["retrieved_contexts"],
        }

        # 如果有标准答案，添加 reference 字段（LLMContextRecall / FactualCorrectness 需要）
        if ground_truth:
            sample["reference"] = ground_truth

        samples.append(sample)

    return samples


def build_metrics() -> list:
    """
    根据 ragas_config 构建评估指标列表

    :return: RAGAS 指标实例列表
    """
    metrics = [Faithfulness()]

    if ragas_config.enable_context_recall:
        metrics.append(LLMContextRecall())

    if ragas_config.enable_factual_correctness:
        metrics.append(FactualCorrectness())

    if ragas_config.enable_response_relevancy and ResponseRelevancy is not None:
        metrics.append(ResponseRelevancy())

    metric_names = [type(m).__name__ for m in metrics]
    logger.info(f"已配置评估指标: {metric_names}")
    return metrics


def run_evaluation(samples: list[dict], metrics: list) -> dict:
    """
    运行 RAGAS 评估

    :param samples: 评估样本列表
    :param metrics: 评估指标列表
    :return: 评估结果（含汇总指标 + 逐条详情）
    """
    # 构建 EvaluationDataset
    evaluation_dataset = EvaluationDataset.from_list(samples)
    logger.info(f"已构建评估数据集，共 {len(samples)} 条样本")

    # 配置评估 LLM（复用项目的 ChatOpenAI 实例）
    llm = get_llm_client()
    evaluator_llm = LangchainLLMWrapper(llm)
    logger.info("评估 LLM 已就绪")

    # 按需配置评估 Embeddings（ResponseRelevancy 需要）
    evaluator_embeddings = None
    if ragas_config.enable_response_relevancy and ResponseRelevancy is not None:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from app.eval.bge_m3_lc_embeddings import BGEM3LangchainEmbeddings
        evaluator_embeddings = LangchainEmbeddingsWrapper(BGEM3LangchainEmbeddings())
        logger.info("评估 Embeddings (BGE-M3) 已就绪")

    # 运行评估
    logger.info("开始运行 RAGAS 评估...")
    start_time = time.time()

    kwargs = {
        "dataset": evaluation_dataset,
        "metrics": metrics,
        "llm": evaluator_llm,
    }
    if evaluator_embeddings is not None:
        kwargs["embeddings"] = evaluator_embeddings

    result = evaluate(**kwargs)

    elapsed = time.time() - start_time
    logger.info(f"RAGAS 评估完成，耗时 {elapsed:.1f} 秒")

    return result


def save_report(result, samples: list[dict], output_path: str):
    """
    保存评估报告到 JSON 文件

    :param result: RAGAS 评估结果
    :param samples: 评估样本
    :param output_path: 输出文件路径
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 转为 DataFrame 再转 JSON
    try:
        df = result.to_pandas()
        records = json.loads(df.to_json(orient="records", force_ascii=False))
    except Exception as e:
        logger.warning(f"转 DataFrame 失败，使用原始结果: {e}")
        records = []

    # 构建完整报告
    repr_dict = getattr(result, "_repr_dict", {})
    report = {
        "summary": repr_dict,
        "details": records,
        "total_samples": len(samples),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(f"评估报告已保存至: {output_path}")


def print_summary(result, total: int):
    """
    在控制台打印评估汇总

    :param result: RAGAS 评估结果
    :param total: 样本总数
    """
    print("\n" + "=" * 60)
    print("                    RAGAS RAG 评估报告")
    print("=" * 60)
    print(f"  评估样本数: {total}")
    print("-" * 60)
    print("  核心指标汇总:")
    print("-" * 60)

    # EvaluationResult._repr_dict 是 {指标名: 平均分} 的字典
    repr_dict = getattr(result, "_repr_dict", {})
    for key, value in repr_dict.items():
        if isinstance(value, (int, float)):
            print(f"  {key:<30s} {value:>8.4f}")
        else:
            print(f"  {key:<30s} {value}")

    print("=" * 60)

    # 尝试打印逐条详情
    try:
        df = result.to_pandas()
        print("\n  逐条评分详情:")
        print("-" * 60)
        print(df.to_string(index=False))
    except Exception:
        pass

    print("\n" + "=" * 60)


def main():
    """
    评估主流程入口

    可选参数:
        --limit N   只运行前 N 条测试用例（快速验证）
    """
    parser = argparse.ArgumentParser(description="RAGAS RAG 评估")
    parser.add_argument("--limit", type=int, default=None, help="只运行前 N 条测试用例")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("           RAGAS RAG 评估开始")
    print("=" * 60)

    # 1. 加载测试数据集
    test_data = load_test_dataset(ragas_config.test_dataset_path)
    print(f"加载测试数据集: {ragas_config.test_dataset_path}")
    print(f"问答对数量: {len(test_data)}")

    # 如果指定了 --limit，截取前 N 条
    if args.limit and args.limit > 0:
        test_data = test_data[:args.limit]
        print(f"[限制模式] 只运行前 {args.limit} 条")

    # 2. 运行 RAG 查询，收集评估数据
    print(f"\n开始运行 RAG 查询流程（共 {len(test_data)} 条）...")
    samples = build_evaluation_samples(test_data)

    # 统计查询结果
    success_count = sum(1 for s in samples if s.get("response") and not s["response"].startswith("[查询异常]"))
    empty_context_count = sum(1 for s in samples if not s.get("retrieved_contexts"))
    print(f"\n查询完成: 成功 {success_count}/{len(samples)}，无上下文 {empty_context_count} 条")

    # 3. 构建评估指标
    metrics = build_metrics()

    # 4. 运行 RAGAS 评估
    result = run_evaluation(samples, metrics)

    # 5. 输出报告
    print_summary(result, len(samples))
    save_report(result, samples, ragas_config.report_output_path)

    print(f"\n详细报告已保存至: {ragas_config.report_output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
