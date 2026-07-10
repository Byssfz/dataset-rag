import sys

from app.clients.neo4j_utils import is_neo4j_configured, search_kg_facts
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


def node_search_kg(state: QueryGraphState):
    """
    节点功能：从 Neo4j 知识图谱中召回结构化事实。

    返回的 kg_docs 会在 node_rerank 中和本地向量结果、网页搜索结果一起合并，
    再交给 reranker 做统一精排。
    """
    function_name = sys._getframe().f_code.co_name
    print("---KG知识图谱检索 开始处理---")
    add_running_task(state["session_id"], function_name, state.get("is_stream"))

    try:
        if not is_neo4j_configured():
            logger.warning("未配置 Neo4j，跳过知识图谱检索")
            return {"kg_docs": []}

        item_names = state.get("item_names") or []
        query = state.get("rewritten_query") or state.get("original_query") or ""
        kg_docs = search_kg_facts(item_names=item_names, query=query, limit=8)
        logger.info(f"KG知识图谱检索完成，数量={len(kg_docs)}，结果={kg_docs}")
        return {"kg_docs": kg_docs}
    except Exception as e:
        logger.error(f"KG知识图谱检索失败：{e}", exc_info=True)
        return {"kg_docs": []}
    finally:
        add_done_task(state["session_id"], function_name, state.get("is_stream"))
        print("---KG知识图谱检索 处理结束---")
