import os
import sys

# 动态将项目根目录加入环境路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.constant import DIR_MAPPING
from FileProcess.file_extract import WosFieldExtractor
from FileProcess.file_merge import TextFilesMerger
from visualize.analysis import GraphAnalysisEngine


class PipelineRunner:
    """
    自动化工作流管理器：提取 -> 合并 -> 分析
    将 run_process.py 中原来的 start_pipeline() 函数解耦为类，
    内部分别调用 WosFieldExtractor、TextFilesMerger、GraphAnalysisEngine。
    """

    def __init__(self, source_dir, base_extract_dir, max_workers=10):
        """
        初始化工作流管理器
        :param source_dir: 原始 txt 文件存放的根目录
        :param base_extract_dir: 提取结果输出的根目录
        :param max_workers: 全局线程池大小
        """
        self.source_dir = source_dir
        self.base_extract_dir = base_extract_dir
        self.max_workers = max_workers

    def run(self, target, top_k=300, top_freq_k=100, threshold=0.7):
        """
        运行单一目标字段的完整工作流：提取 -> 合并 -> 可视化分析
        :param target: 提取目标字段 (如 DE, AU, ID, C1)
        :param top_k: 保留出现频率最高的前 top_k 个词构建图谱
        :param top_freq_k: 语义合并后保留的关键词数
        :param threshold: 语义相似度阈值
        """
        target = target.upper()
        sub_dir_name = DIR_MAPPING.get(target, target.lower())
        target_out_dir = os.path.join(self.base_extract_dir, sub_dir_name)

        print(f"\n{'=' * 20}")
        print(f">>> 开始处理任务目标: {target} <<<")
        print(f"{'=' * 20}")

        # --- 步骤 1: 提取数据 (WosFieldExtractor) ---
        print(f"\n[Step 1] 正在提取 {target} 数据...")
        extractor = WosFieldExtractor(
            source_dir=self.source_dir,
            base_output_dir=self.base_extract_dir,
            target_name=target,
            max_workers=self.max_workers
        )
        extractor.extract()

        # --- 步骤 2: 合并文件 (TextFilesMerger) ---
        merged_filename = f"merged_{sub_dir_name}.txt"
        merged_filepath = os.path.join(target_out_dir, merged_filename)
        print(f"\n[Step 2] 正在合并文件至: {merged_filepath}")
        merger = TextFilesMerger(
            source_dir=target_out_dir,
            output_name=merged_filename
        )
        merger.merge()

        # --- 步骤 3: 运行可视化分析 (GraphAnalysisEngine) ---
        png_filepath = os.path.join(target_out_dir, f"merged_{sub_dir_name}_map.png")
        print(f"\n[Step 3] 正在启动可视化分析...")
        engine = GraphAnalysisEngine(max_workers=self.max_workers)
        engine.run(
            file_path=merged_filepath,
            output_png=png_filepath,
            target=target,
            top_k=top_k,
            threshold=threshold
        )

        print("\n" + "=" * 50)
        print(f"[{target}] 工作流执行完毕！")
        print(f"最终结果图谱: {png_filepath}")
        print("=" * 50)

    def run_all(self, target_list, top_k=300, top_freq_k=100, threshold=0.7):
        """
        批量运行多个目标字段的完整工作流
        :param target_list: 目标字段列表 (如 ["DE", "AU", "ID", "C1"])
        """
        for target in target_list:
            self.run(target, top_k=top_k, top_freq_k=top_freq_k, threshold=threshold)


if __name__ == "__main__":
    # ================= 配置区域 =================
    SOURCE_DIR = r"./data"
    BASE_OUTPUT_DIR = r"./extract"
    TARGET_LIST = ["DE", "AU", "ID", "C1"]

    TOP_K = 300
    TOP_FREQ_K = 100
    THRESHOLD = 0.7
    MAX_WORKERS = 100  # 线程池最大线程数
    # ==========================================

    # 初始化管理器并启动
    pipeline = PipelineRunner(SOURCE_DIR, BASE_OUTPUT_DIR, MAX_WORKERS)
    pipeline.run_all(TARGET_LIST, top_k=TOP_K, top_freq_k=TOP_FREQ_K, threshold=THRESHOLD)
