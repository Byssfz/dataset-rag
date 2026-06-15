import sys
from pathlib import Path

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import add_running_task, add_done_task


def step_1_validate_paths(state: ImportGraphState):
    """
    进行路径校验！ pdf_path失效 直接异常处理!
                local_dir 没有，给与默认值
    :param state:
    :return:
    """
    logger.debug(f">>> [step_1_validate_paths]在md转pdf下，开始进行文件格式校验！！")
    pdf_path = state['pdf_path']
    local_dir = state['local_dir']
    # 常规的非空校验 （站在字符串的角度）
    if not pdf_path:
        logger.error("step_1_validate_paths检查发现没有输入文件，无法继续解析！！")
        raise ValueError("step_1_validate_paths检查发现没有输入文件，无法继续解析！！")
    if not local_dir:
        # 给与一个输出的默认值
        local_dir = str(PROJECT_ROOT / "output")
        logger.info(f"step_1_validate_paths检查发现local_dir没有赋值，给与默认值：{local_dir}！")
    # 进行文件存在校验
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)

    if not pdf_path_obj.exists():
        logger.error(f"[step_1_validate_paths检查发现pdf_path不存在，请检查输入文件路径是否正确！！")
        raise FileNotFoundError(f"[step_1_validate_paths]检查发现pdf_path不存在，请检查输入文件路径是否正确！！")
    if not local_dir_obj.exists():
        logger.error(f"[step_1_validate_paths检查发现local_dir不存在，主动创建对应的文件夹！！！")
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj, local_dir_obj


def step_2_upload_and_poll(pdf_path_obj):
    pass


def step_3_download_and_extract(zip_url, local_dir_obj, stem):
    pass


def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    1. 调用 MinerU (magic-pdf) 工具。
    2. 将 PDF 转换成 Markdown 格式。
    3. 将结果保存到 state["md_content"]。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
    add_running_task(state['task_id'],function_name)

    try:
        # 2.进行参数校验 （local_dir -》 给与默认值 | local_file_path完成字面意思的校验 -》 深入校验校验的文件是否真的存在）
        # 参数：state local_file_path | local_dir
        # 返回：校验后的文件和输出文件夹 Path对象
        pdf_path_obj, local_dir_obj = step_1_validate_paths(state)
        # 3.调用minerU进行pdf的解析（local_file_path）返回一个下载文件的地址 xx.zip url地址
        # 参数：要解析的pdf文件路径  返回值：要下载的zip文件地址
        zip_url = step_2_upload_and_poll(pdf_path_obj)
        # 4.下载zip包，并且解析和提取 （local_dir）
        # 参数：1.要下载的地址 2. local_dir_obj 解压的文件夹  3. 文件名 二狗子 (二狗子.pdf)
        # 返回值：解压后md文件的真实路径
        md_path = step_3_download_and_extract(zip_url,local_dir_obj,pdf_path_obj.stem)
        state["md_path"]=md_path
        state["local_dir"]=local_dir_obj
        with open(md_path, "r", encoding="utf-8") as f:
            state["md_content"] = f.read()
    except Exception as e:
        # 处理异常
        logger.error(f">>> [{function_name}]使用minerU解析发生了异常，异常信息：{e}")
        raise # 终止工作流
    finally:
        logger.info(f">>> [{function_name}]开始结束了！现在的状态为：{state}")
        add_done_task(state['task_id'], function_name)
    return state