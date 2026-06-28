from langgraph.graph import StateGraph, END

from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.agent.state import QueryGraphState

builder = StateGraph(QueryGraphState)

# 节点添加完毕！！
builder.add_node("node_item_name_confirm",node_item_name_confirm)
builder.add_node("node_search_embedding",node_search_embedding)
builder.add_node("node_search_embedding_hyde",node_search_embedding_hyde)
builder.add_node("node_web_search_mcp",node_web_search_mcp)
builder.add_node("node_rrf",node_rrf)
builder.add_node("node_rerank",node_rerank)
builder.add_node("node_answer_output",node_answer_output)

# 添加边
builder.set_entry_point("node_item_name_confirm")

# node_item_name_confirm 可能出现，没有明确的主体 item_name 我们会提前结束返回用户提示，让他明确内容！！
# node_item_name_confirm -> (answer: str  # 最终生成的答案) -》答案生成 给前端反馈  || 多路召回
# 条件边！！！ conditional_edges
# 并行执行：当返回元组 (node1, node2, node3) 时，LangGraph会并行执行这三个节点，提高效率
def route_after_node_item_name_confirm(state: QueryGraphState):
    if state['answer']:
        return "node_answer_output"
    return "node_search_embedding","node_search_embedding_hyde","node_web_search_mcp"

builder.add_conditional_edges("node_item_name_confirm"
                              , route_after_node_item_name_confirm,
                              {
                                  "node_answer_output":"node_answer_output",
                                  "node_search_embedding":"node_search_embedding",
                                  "node_search_embedding_hyde":"node_search_embedding_hyde",
                                  "node_web_search_mcp":"node_web_search_mcp"
                              })
#这里都先到rrf但是网络搜索不参与粗排，与图不同，逻辑差不多
#这里可能会有并发修改问题，不能都直接返回state，改了啥就修改啥
builder.add_edge("node_search_embedding","node_rrf")
builder.add_edge("node_search_embedding_hyde","node_rrf")
builder.add_edge("node_web_search_mcp","node_rrf")

#最后一起到rerank，是为了保证同时到达，并发
builder.add_edge("node_rrf","node_rerank")

builder.add_edge("node_rerank","node_answer_output")
builder.add_edge("node_answer_output",END)

query_app = builder.compile()