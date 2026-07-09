# RAGAS 评估配置
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RagasConfig:
    """RAGAS 评估框架配置"""
    test_dataset_path: str          # 测试数据集 JSON 路径
    report_output_path: str         # 评估报告输出路径
    report_output_dir: str          # 评估报告输出目录
    enable_response_relevancy: bool # 是否启用 ResponseRelevancy 指标（需加载 BGE-M3，耗时较长）
    enable_context_recall: bool     # 是否启用 LLMContextRecall 指标（需要 ground_truth）
    enable_factual_correctness: bool # 是否启用 FactualCorrectness 指标（需要 ground_truth）


ragas_config = RagasConfig(
    test_dataset_path=os.getenv("RAGAS_TEST_DATASET_PATH", "app/eval/test_dataset.json"),
    report_output_path=os.getenv("RAGAS_REPORT_PATH", "output/ragas_evaluation_report.json"),
    report_output_dir=os.getenv("RAGAS_REPORT_DIR", "output"),
    enable_response_relevancy=os.getenv("RAGAS_ENABLE_RESPONSE_RELEVANCY", "True").lower() == "true",
    enable_context_recall=os.getenv("RAGAS_ENABLE_CONTEXT_RECALL", "True").lower() == "true",
    enable_factual_correctness=os.getenv("RAGAS_ENABLE_FACTUAL_CORRECTNESS", "True").lower() == "true",
)
