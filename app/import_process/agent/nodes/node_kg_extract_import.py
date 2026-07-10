import json
import sys
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

from app.clients.neo4j_utils import delete_item_graph, is_neo4j_configured, upsert_kg_triples
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.lm_utils import get_llm_client
from app.utils.task_utils import add_done_task, add_running_task


MAX_CHUNKS_FOR_KG = 30
MAX_CHUNK_CONTENT_CHARS = 1800
MAX_TRIPLES_PER_CHUNK = 8


def _clean_json_text(text: str) -> str:
    content = (text or "").strip()
    if content.startswith("```json"):
        content = content.replace("```json", "", 1).strip()
    if content.startswith("```"):
        content = content.replace("```", "", 1).strip()
    if content.endswith("```"):
        content = content[:-3].strip()
    return content


def _parse_triples_response(content: str) -> List[Dict[str, Any]]:
    try:
        data = json.loads(_clean_json_text(content))
    except json.JSONDecodeError:
        logger.warning(f"KG抽取结果不是合法JSON，原始内容：{content}")
        return []

    if isinstance(data, dict):
        triples = data.get("triples", [])
    elif isinstance(data, list):
        triples = data
    else:
        triples = []
    if not isinstance(triples, list):
        return []
    return [item for item in triples if isinstance(item, dict)]


def _extract_chunk_triples(chunk: Dict[str, Any], item_name: str, file_title: str) -> List[Dict[str, Any]]:
    content = (chunk.get("content") or "")[:MAX_CHUNK_CONTENT_CHARS]
    title = chunk.get("title") or ""
    if not content.strip():
        return []

    prompt = load_prompt(
        "kg_extract",
        item_name=item_name,
        file_title=file_title,
        title=title,
        content=content,
        max_triples=MAX_TRIPLES_PER_CHUNK,
    )
    llm = get_llm_client(json_mode=True)
    response = llm.invoke([HumanMessage(content=prompt)])
    triples = _parse_triples_response(response.content)

    normalized = []
    for triple in triples[:MAX_TRIPLES_PER_CHUNK]:
        subject = (triple.get("subject") or item_name or "").strip()
        obj = (triple.get("object") or "").strip()
        if not subject or not obj:
            continue
        normalized.append({
            "subject": subject,
            "subject_type": triple.get("subject_type") or "概念",
            "predicate": triple.get("predicate") or "RELATED_TO",
            "object": obj,
            "object_type": triple.get("object_type") or "概念",
            "evidence": triple.get("evidence") or "",
            "confidence": triple.get("confidence") or 0.7,
            "chunk_id": chunk.get("chunk_id") or "",
            "file_title": chunk.get("file_title") or file_title,
            "title": title,
            "item_name": item_name,
        })
    return normalized


def node_kg_extract_import(state: ImportGraphState) -> ImportGraphState:
    """
    从导入后的 chunks 中抽取知识图谱三元组，并写入 Neo4j。

    该节点放在 Milvus 入库之后执行，因为 Milvus 会回填 chunk_id，KG 需要这个 id 做证据溯源。
    未配置 Neo4j 时自动跳过，不影响原有向量 RAG 导入链路。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
    add_running_task(state["task_id"], function_name)

    try:
        if not is_neo4j_configured():
            logger.warning("未配置 Neo4j，跳过知识图谱导入节点")
            state["kg_triples"] = []
            return state

        chunks = state.get("chunks") or []
        item_name = state.get("item_name") or (chunks[0].get("item_name") if chunks else "")
        file_title = state.get("file_title") or (chunks[0].get("file_title") if chunks else "")
        if not chunks or not item_name:
            logger.warning("缺少 chunks 或 item_name，跳过知识图谱导入")
            state["kg_triples"] = []
            return state

        all_triples: List[Dict[str, Any]] = []
        for chunk in chunks[:MAX_CHUNKS_FOR_KG]:
            chunk_triples = _extract_chunk_triples(chunk, item_name, file_title)
            all_triples.extend(chunk_triples)

        delete_item_graph(item_name)
        written_count = upsert_kg_triples(item_name, all_triples)
        state["kg_triples"] = all_triples
        logger.info(f"知识图谱抽取和导入完成，抽取数量={len(all_triples)}，写入数量={written_count}")
    except Exception as e:
        logger.error(f">>> [{function_name}]知识图谱导入发生异常：{e}", exc_info=True)
        raise
    finally:
        logger.info(f">>> [{function_name}]执行结束，现在的状态为：{state}")
        add_done_task(state["task_id"], function_name)

    return state
