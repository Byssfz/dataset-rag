import re
import sys

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task


def step_1_get_content(state:ImportGraphState) -> Tuple[str, Path, Path]:
    """
    提取内容
    :param state:
    :return:
    """
    # 1. 获取md的地址 md_path
    md_file_path = state["md_path"]
    if not md_file_path:
        raise ValueError("md_path不能为空！")

    md_path_obj = Path(md_file_path)
    if not md_path_obj.exists():
        raise FileNotFoundError(f"md_path:{md_file_path} 文件不存在！")

    # 要读取md_content
    if not state['md_content']:
        # 没有，再读取！ 有，证明是pdf节点解析过来的，已经给md_content进行赋值了！
        with md_path_obj.open("r", encoding="utf-8") as f:
            md_content = f.read()
        state['md_content'] = md_content

    # 图片文件夹obj
    # 注意：自己传入的md -》 你的图片文件夹也必须交 images
    images_dir_obj = md_path_obj.parent / "images"
    return md_content , md_path_obj, images_dir_obj


def find_image_in_md_content(md_content, image_file, context_length: int = 100):
    """
    从md_content识别图片的上下文！
    约定上下文长度100
    :param md_content:
    :param image_file:
    :param context_length：默认截取长度
    :return:
    """

    """
    # 你好啊
    我很好，还有7行代码今天就结束了！小伙伴们坚持好！谢谢！
    哈哈
    哈
    嘿嘿
    【start】 ![二大爷](/xxx/xx/zhaoweifeng.jpgxxx)【end】
    啦啦啦啦
    巴巴爸爸
    ![二大爷](/xxx/xx/zhaoweifeng.jpgxxx)
    嘿嘿额

    file_name zhaoweifeng.jpg
    """
    # 定义正则表达式  .*  .*?
    pattern = re.compile(r"!\[.*?\]\(.*?" + image_file + ".*?\)")

    results = []  # 存储图片多处使用，上下文不同 ！ 本次暴力处理，获取第一个！
    # 查询符合位置
    for item in pattern.finditer(md_content):
        start, end = item.span()  # span获取匹配对象的起始和终止的位置
        # 截取上文
        pre_text = md_content[max(start - context_length, 0):start]  # 考虑前面有没有context_length 没有从0开始
        post_text = md_content[end:min(end + context_length, len(md_content))]  # 考虑后面有没有context_length 没有就到长度
        # 截取下文
        results.append((pre_text, post_text))
    # 截取位置前后的内容
    if results:
        logger.info(f"图片：{image_file} ,在{md_content[:100]}中使用了：{len(results)}次，截取第一个上下文：{results[0]}")
        return results[0]

def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
    add_running_task(state['task_id'], function_name)
    return state