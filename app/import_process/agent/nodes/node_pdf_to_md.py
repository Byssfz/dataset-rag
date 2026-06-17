import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import requests

from app.conf.mineru_config import mineru_config
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState, create_default_state
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
    """
      将pdf文件使用minerU解析，并且获取md对应的下载的url地址！！
      :param pdf_path_obj: 上传解析pdf文件的 path对象
      :return: str -> url , minerU解析后md文件zip压缩包的下载地址
      """
    # 1. 申请上传解析的地址
    # 前置准备和参数： url api | token | 准备固定格式的请求头
    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": f"{pdf_path_obj.name}"}
        ],
        "model_version": "vlm"
    }
    response = requests.post(url, headers=header, json=data,timeout=30)
    print(response)
    # 结果处理 请求http状态码不是200 或者 返回结果的状态码 不是0 请求失败！
    if response.status_code != 200 or response.json()['code'] != 0:
        logger.error(f"[step_2_upload_and_poll]请求minerU解析接口失败，请检查输入文件路径是否正确！！")
        raise RuntimeError(f"[step_2_upload_and_poll]请求minerU解析接口失败，请检查输入文件路径是否正确！！")
    uploaded_url = response.json()['data']['file_urls'][0] # 上这个地址上传文件
    batch_id = response.json()['data']['batch_id'] # 处理id，后续根据这个id获取结果！

    # 2. 将文件上传到对应的解析地址
    # 使用Put请求，将pdf_path_obj文件传递到uploaded_url地址即可！
    # 注意： 不能直接使用put! 这块很大概率报错！ 原因：电脑开了各种代理，put的请求头，添加一些额外的参数头！将文件真的转存到第三方的文件存储服务器！
    # 文件存储服务器检查都比较严格！ 拒绝存储！报错！ get post 宽进宽出  put严进严出！
    #这里用一个httpsession来关闭代理
    http_session = requests.Session()
    http_session.trust_env = False # 1.禁止走代理 2. 复用请求对象
    try:
        with open(pdf_path_obj,'rb') as f:
            file_data = f.read()
        upload_response = http_session.put(uploaded_url,data=file_data)
        if upload_response.status_code != 200:
            logger.error(f"[step_2_upload_and_poll]上传文件到minerU失败，请检查输入文件路径是否正确！！")
            raise RuntimeError(f"[step_2_upload_and_poll]上传文件到minerU失败，请检查输入文件路径是否正确！！")
    except Exception as e:
        logger.error(f"[step_2_upload_and_poll]上传文件到minerU失败，请检查输入文件路径是否正确！！")
        raise RuntimeError(f"[step_2_upload_and_poll]上传文件到minerU失败，请检查输入文件路径是否正确！！")
    finally:
        http_session.close()


    # 3. 轮询获取解析结果
    # 循环获取！确保获取到结果，再先后执行！！
    # 设计一个循环，3秒获取一次！ 最多等待10分钟600 -> 600页pdf
    url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    timeout_seconds = 600  # 1s -> 1页pdf
    poll_interval = 3  #间隔时间是3秒
    start_time = time.time() # 进去起始时间

    while True:
        if time.time()-start_time > timeout_seconds:
            logger.error("[step_2_upload_and_poll]请求minerU解析接口超时，请检查输入文件路径是否正确！！")
            raise TimeoutError("[step_2_upload_and_poll]请求minerU解析接口超时，请检查输入文件路径是否正确！！")
        res=requests.get(url,headers=header)
        if res.status_code !=200:
            # 5xx系列我们会给与机会，直到timeout
            # http的状态码 1xx 2xx 3xx 4xx  【5xx】
            if 500 <= res.status_code < 600:
                time.sleep(poll_interval)
                continue
            raise RuntimeError(f"[step_2_upload_and_poll]请求minerU解析接口失败，返回的状态码{res.status_code}！！")
        json_data = res.json()  # 获取本次结果
        if json_data['code'] != 0:
            raise RuntimeError(f"[step_2_upload_and_poll]请求minerU解析接口失败，返回的错误:{json_data['code']}信息{json_data['msg']}！！")
        extract_result=json_data['data']['extract_result'][0]
        if extract_result ['state'] == 'done':
            full_zip_url=extract_result['full_zip_url']
            logger.info(f"已经完成pdf的解析，耗时：{time.time() - start_time}s,解析结果：{full_zip_url}")
            return full_zip_url
        else:
            time.sleep(poll_interval)

def step_3_download_and_extract(zip_url, local_dir_obj, stem):
    response=requests.get(zip_url,timeout=120)
    if response.status_code != 200:
        logger.error(f"[step_3_download_and_extract]下载文件失败，请检查输入文件路径是否正确！！")
        raise RuntimeError(f"[step_3_download_and_extract]下载文件失败，请检查输入文件路径是否正确！！")
    zip_save_path=local_dir_obj/f"{stem}_result.zip"
    with open(zip_save_path,"wb") as f:
        f.write(response.content)
    logger.info(f"[step_3_download_and_extract]下载文件成功，保存位置：{zip_save_path}")
    extract_target_dir = local_dir_obj / stem
    if extract_target_dir.exists():
        shutil.rmtree(extract_target_dir)
    extract_target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_save_path, 'r') as zip_file_object:
        zip_file_object.extractall(extract_target_dir)
        # 5. 返回md文件的地址
        # 解压以后得文件的文件名 可能叫 文件.md 低版本 也可能叫 full.md
    md_file_list = list(extract_target_dir.rglob("*.md"))

    if not md_file_list:
        logger.error(f"[step_3_download_and_extract]没有找到md文件，请检查输入文件路径是否正确！！")
        raise RuntimeError(f"[step_3_download_and_extract]没有找到md文件，请检查输入文件路径是否正确！！")

    target_md_file = None  # 存储最终md文件
    # 检查有没有原文件名的md
    for md_file in md_file_list:
        # stem 文件名 二狗子
        if md_file.name == stem + ".md":
            target_md_file = md_file
            break
    # 检查有没有full.md (第一次没有找到，才找full)
    if not target_md_file:
        for md_file in md_file_list:
            # stem 文件名 二狗子
            if md_file.name.lower() == "full.md":
                target_md_file = md_file
                break
    # 实在没有我就获取第一个就行
    if not target_md_file:
        target_md_file = md_file_list[0]

    # md文件名  二狗子.md  full.md  不知道.md
    # 统一改成  原文件名（stem）.md
    # 不是原名字的时候，我才重命名
    if target_md_file.stem != stem:
        # 进行重命名
        # target_md_file.with_name(f"{stem}.md") 修改path对象 （不涉及文件操作） 返回结果是修改后path对象
        # target_md_file.rename(target_md_file.with_name(f"{stem}.md")) 修改磁盘中的文件名称（修改名称了） return 新的路径path
        target_md_file = target_md_file.rename(target_md_file.with_name(f"{stem}.md"))

    # 最终的md文件获取绝对路径，并且返回字符串类型
    final_md_str_path = str(target_md_file.resolve())
    logger.info(f"[step_3_download_and_extract]完成md解压，最终存储md路径为：{final_md_str_path}")
    return final_md_str_path
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

if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")