"""
test_run_process.py - 针对 run_process.py 中所用到的每个类的单元测试

测试目标：
1. TestWosFieldExtractor        - 测试 WosFieldExtractor 的字段提取功能
2. TestTextFilesMerger          - 测试 TextFilesMerger 的文件合并功能
3. TestGraphAnalysisEngine      - 测试 GraphAnalysisEngine 的图构建与合并功能
4. TestPipelineRunner           - 测试 PipelineRunner 的端到端完整流水线 (mock 数据)
5. TestRealDataAuthorPipeline   - 使用 data/ 目录真实 WoS 数据，提取 AU 并绘制作者图谱
6. TestRealDataKeywordPipeline  - 使用 data/ 目录真实 WoS 数据，提取 DE 并绘制关键词图谱
                                  (含语义合并，每步展示调用的类与文件路径)
"""

import os
import sys
import unittest
import tempfile
import shutil

# 确保项目根目录在导入路径中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from FileProcess.file_extract import WosFieldExtractor
from FileProcess.file_merge import TextFilesMerger
from visualize.analysis import GraphAnalysisEngine
from entity.graph import WeightedGraph
from run_process import PipelineRunner


# ============================================================
# Mock 数据：模拟 Web of Science 格式的原始文件
# ============================================================
MOCK_WOS_FILE_1 = """\
PT J
AU Smith, J
   Wang, L
DE Machine learning; Artificial Intelligence; Deep learning
ID Neural Networks; Data Mining
C1 [Smith, J] Univ ABC, Dept Chem, New York, USA.
   [Wang, L] Tsinghua Univ, Dept CS, Beijing, Peoples R China.
CR Author1, 2020, V1, P1, DOI 10.1000/test.001
   Author2, 2019, V2, P2, DOI 10.1000/test.002
ER

PT J
AU Doe, A
   Li, B
DE Artificial Intelligence; Natural Language Processing
ID Data Mining; Computer Vision
C1 [Doe, A] MIT, Cambridge, USA.
   [Li, B] Peking Univ, Beijing, Peoples R China.
CR Author3, 2021, V3, P3, DOI 10.1000/test.003
ER
"""

MOCK_WOS_FILE_2 = """\
PT J
AU Johnson, K
DE Deep learning; Reinforcement Learning
ID Neural Networks; Optimization
C1 [Johnson, K] Stanford Univ, Stanford, USA.
ER
"""


class TestWosFieldExtractor(unittest.TestCase):
    """测试 WosFieldExtractor 类的字段提取功能"""

    def setUp(self):
        """每个测试前创建临时目录和 mock 数据"""
        self.temp_base = tempfile.mkdtemp()
        self.source_dir = os.path.join(self.temp_base, "raw_data")
        self.extract_dir = os.path.join(self.temp_base, "extract")
        os.makedirs(self.source_dir)

        # 写入 mock 文件
        with open(os.path.join(self.source_dir, "file1.txt"), "w", encoding="utf-8") as f:
            f.write(MOCK_WOS_FILE_1)
        with open(os.path.join(self.source_dir, "file2.txt"), "w", encoding="utf-8") as f:
            f.write(MOCK_WOS_FILE_2)

    def tearDown(self):
        """每个测试后清理临时目录"""
        shutil.rmtree(self.temp_base)

    def test_extract_de_keywords(self):
        """测试提取 DE (关键词) 字段"""
        extractor = WosFieldExtractor(
            source_dir=self.source_dir,
            base_output_dir=self.extract_dir,
            target_name="DE",
            max_workers=2
        )
        extractor.extract()

        # 验证输出目录已创建
        output_dir = os.path.join(self.extract_dir, "keywords")
        self.assertTrue(os.path.exists(output_dir), "关键词输出目录未创建")

        # 验证输出文件存在且有内容
        output_files = [f for f in os.listdir(output_dir) if f.endswith(".txt")]
        self.assertGreater(len(output_files), 0, "未生成提取结果文件")

        # 读取并验证内容包含关键词
        all_content = ""
        for fname in output_files:
            with open(os.path.join(output_dir, fname), "r", encoding="utf-8") as f:
                all_content += f.read()
        # 应该包含方括号格式的关键词
        self.assertIn("[", all_content, "提取内容缺少方括号格式")
        print("  -> DE 关键词提取测试通过！")

    def test_extract_au_authors(self):
        """测试提取 AU (作者) 字段"""
        extractor = WosFieldExtractor(
            source_dir=self.source_dir,
            base_output_dir=self.extract_dir,
            target_name="AU",
            max_workers=2
        )
        extractor.extract()

        output_dir = os.path.join(self.extract_dir, "authors")
        self.assertTrue(os.path.exists(output_dir), "作者输出目录未创建")

        output_files = [f for f in os.listdir(output_dir) if f.endswith(".txt")]
        self.assertGreater(len(output_files), 0, "未生成作者提取结果文件")

        all_content = ""
        for fname in output_files:
            with open(os.path.join(output_dir, fname), "r", encoding="utf-8") as f:
                all_content += f.read()
        # 应该能找到作者名
        self.assertIn("Smith", all_content, "提取内容缺少作者信息")
        print("  -> AU 作者提取测试通过！")

    def test_extract_c1_country(self):
        """测试提取 C1 (国家/地区) 字段"""
        extractor = WosFieldExtractor(
            source_dir=self.source_dir,
            base_output_dir=self.extract_dir,
            target_name="C1",
            max_workers=2
        )
        extractor.extract()

        output_dir = os.path.join(self.extract_dir, "country")
        self.assertTrue(os.path.exists(output_dir), "国家输出目录未创建")

        output_files = [f for f in os.listdir(output_dir) if f.endswith(".txt")]
        self.assertGreater(len(output_files), 0, "未生成国家提取结果文件")
        print("  -> C1 国家提取测试通过！")

    def test_extract_nonexistent_source_dir(self):
        """测试源目录不存在时不崩溃"""
        extractor = WosFieldExtractor(
            source_dir="/nonexistent/path",
            base_output_dir=self.extract_dir,
            target_name="DE",
            max_workers=2
        )
        # 不应抛出异常
        extractor.extract()
        print("  -> 源目录不存在容错测试通过！")


class TestTextFilesMerger(unittest.TestCase):
    """测试 TextFilesMerger 类的文件合并功能"""

    def setUp(self):
        """创建包含多个待合并文件的临时目录"""
        self.temp_dir = tempfile.mkdtemp()

        # 创建模拟的提取结果文件
        with open(os.path.join(self.temp_dir, "file1.txt"), "w", encoding="utf-8") as f:
            f.write("1. [Machine learning], [Artificial Intelligence]\n")
            f.write("2. [Deep learning], [Neural Networks]\n")

        with open(os.path.join(self.temp_dir, "file2.txt"), "w", encoding="utf-8") as f:
            f.write("1. [Natural Language Processing]\n")
            f.write("2. [Computer Vision], [Data Mining]\n")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_merge_creates_output_file(self):
        """测试合并后输出文件是否生成"""
        output_name = "merged_output.txt"
        merger = TextFilesMerger(source_dir=self.temp_dir, output_name=output_name)
        merger.merge()

        output_path = os.path.join(self.temp_dir, output_name)
        self.assertTrue(os.path.exists(output_path), "合并输出文件未生成")
        print("  -> 合并文件生成测试通过！")

    def test_merge_content_is_renumbered(self):
        """测试合并后内容是否重新编号"""
        output_name = "merged_output.txt"
        merger = TextFilesMerger(source_dir=self.temp_dir, output_name=output_name)
        merger.merge()

        output_path = os.path.join(self.temp_dir, output_name)
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 应该有4行（2个文件各2行）
        self.assertEqual(len(lines), 4, f"合并后行数不正确，期望4行，实际{len(lines)}行")

        # 验证重新编号 1, 2, 3, 4
        for i, line in enumerate(lines, 1):
            self.assertTrue(line.startswith(f"{i}."), f"第{i}行编号不正确: {line.strip()}")
        print("  -> 合并内容重新编号测试通过！")

    def test_merge_excludes_output_file(self):
        """测试合并过程中排除输出文件本身"""
        output_name = "merged_output.txt"
        # 先创建一个同名文件
        with open(os.path.join(self.temp_dir, output_name), "w") as f:
            f.write("should be excluded")

        merger = TextFilesMerger(source_dir=self.temp_dir, output_name=output_name)
        merger.merge()

        output_path = os.path.join(self.temp_dir, output_name)
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 不应包含 "should be excluded"
        self.assertNotIn("should be excluded", content, "输出文件自身未被排除")
        print("  -> 排除输出文件自身测试通过！")

    def test_merge_nonexistent_dir(self):
        """测试目录不存在时不崩溃"""
        merger = TextFilesMerger(source_dir="/nonexistent/dir", output_name="out.txt")
        # 不应抛出异常
        merger.merge()
        print("  -> 目录不存在容错测试通过！")


class TestGraphAnalysisEngine(unittest.TestCase):
    """测试 GraphAnalysisEngine 及其依赖的 WeightedGraph 的图构建与合并功能"""

    def test_weighted_graph_build_from_list(self):
        """测试 WeightedGraph 从关键词列表构建图"""
        graph = WeightedGraph("test")
        graph.build_from_list(["apple", "banana", "cherry"])

        # 3个节点
        self.assertEqual(len(graph.nodes), 3, "节点数不正确")
        # 3条边 (C(3,2) = 3)
        self.assertEqual(len(graph.edges), 3, "边数不正确")
        print("  -> 图构建测试通过！")

    def test_weighted_graph_merge(self):
        """测试两个图合并后边权重正确"""
        g1 = WeightedGraph("1")
        g1.build_from_list(["apple", "banana", "cherry"])

        g2 = WeightedGraph("2")
        g2.build_from_list(["apple", "banana"])

        g1.merge_from(g2)

        # apple-banana 边权重应为 2
        key = tuple(sorted(("apple", "banana")))
        self.assertEqual(g1.edges.get(key), 2, "合并后 apple-banana 边权重不正确")
        print("  -> 图合并测试通过！")

    def test_weighted_graph_no_self_loop(self):
        """测试不会产生自环"""
        graph = WeightedGraph("test")
        graph.add_edge("a", "a", weight=5)
        self.assertEqual(len(graph.edges), 0, "自环应被忽略")
        print("  -> 无自环测试通过！")

    def test_weighted_graph_single_node(self):
        """测试单节点列表"""
        graph = WeightedGraph("test")
        graph.build_from_list(["only_one"])
        self.assertEqual(len(graph.nodes), 1, "单节点数不正确")
        self.assertEqual(len(graph.edges), 0, "单节点不应有边")
        print("  -> 单节点测试通过！")

    def test_engine_build_subgraph(self):
        """测试 GraphAnalysisEngine 的子图构建方法"""
        engine = GraphAnalysisEngine(max_workers=2)

        # 模拟一行提取数据
        line = "1. [apple], [banana], [cherry]"
        valid_set = {"apple", "banana", "cherry"}
        mapping = {}
        results = [None]

        engine._build_single_subgraph(line, valid_set, mapping, results, 0)

        self.assertIsNotNone(results[0], "子图构建结果不应为 None")
        self.assertEqual(len(results[0].nodes), 3, "子图节点数不正确")
        print("  -> 子图构建测试通过！")

    def test_engine_build_subgraph_with_mapping(self):
        """测试带映射关系的子图构建"""
        engine = GraphAnalysisEngine(max_workers=2)

        line = "1. [ml], [ai], [dl]"
        valid_set = {"ml", "ai", "dl"}
        mapping = {"ml": "machine learning", "ai": "artificial intelligence", "dl": "deep learning"}
        results = [None]

        engine._build_single_subgraph(line, valid_set, mapping, results, 0)

        self.assertIsNotNone(results[0], "映射后子图不应为 None")
        # 验证映射后的节点名
        self.assertIn("machine learning", results[0].nodes, "映射未生效")
        print("  -> 带映射的子图构建测试通过！")

    def test_engine_build_subgraph_empty_line(self):
        """测试空行不构建子图"""
        engine = GraphAnalysisEngine(max_workers=2)
        results = [None]
        engine._build_single_subgraph("", set(), {}, results, 0)
        self.assertIsNone(results[0], "空行应返回 None")
        print("  -> 空行子图测试通过！")


class TestPipelineRunner(unittest.TestCase):
    """测试 PipelineRunner 类的端到端流水线功能"""

    def setUp(self):
        """创建临时环境"""
        self.temp_base = tempfile.mkdtemp()
        self.source_dir = os.path.join(self.temp_base, "raw_data")
        self.extract_dir = os.path.join(self.temp_base, "extract")
        os.makedirs(self.source_dir)

        # 写入 mock 数据
        with open(os.path.join(self.source_dir, "file1.txt"), "w", encoding="utf-8") as f:
            f.write(MOCK_WOS_FILE_1)
        with open(os.path.join(self.source_dir, "file2.txt"), "w", encoding="utf-8") as f:
            f.write(MOCK_WOS_FILE_2)

    def tearDown(self):
        shutil.rmtree(self.temp_base)

    def test_pipeline_runner_init(self):
        """测试 PipelineRunner 初始化参数"""
        runner = PipelineRunner(
            source_dir=self.source_dir,
            base_extract_dir=self.extract_dir,
            max_workers=2
        )
        self.assertEqual(runner.source_dir, self.source_dir)
        self.assertEqual(runner.base_extract_dir, self.extract_dir)
        self.assertEqual(runner.max_workers, 2)
        print("  -> PipelineRunner 初始化测试通过！")

    def test_pipeline_run_au(self):
        """端到端测试：AU 字段的完整流水线（提取→合并→分析）
        使用 AU 类型避免加载语义合并 ML 模型，加速测试。
        """
        runner = PipelineRunner(
            source_dir=self.source_dir,
            base_extract_dir=self.extract_dir,
            max_workers=2
        )
        runner.run("AU", top_k=5, threshold=0.7)

        # 验证合并文件生成
        merged_path = os.path.join(self.extract_dir, "authors", "merged_authors.txt")
        self.assertTrue(os.path.exists(merged_path), f"合并文件 {merged_path} 未生成")

        with open(merged_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertGreater(len(content), 0, "合并文件内容为空")

        # 验证可视化图谱生成
        png_path = os.path.join(self.extract_dir, "authors", "merged_authors_map.png")
        self.assertTrue(os.path.exists(png_path), f"可视化图谱 {png_path} 未生成")
        print("  -> AU 端到端流水线测试通过！")

    def test_pipeline_run_all(self):
        """测试 run_all 批量运行（仅测试 AU，避免 ML 模型加载）"""
        runner = PipelineRunner(
            source_dir=self.source_dir,
            base_extract_dir=self.extract_dir,
            max_workers=2
        )
        runner.run_all(["AU"], top_k=5, threshold=0.7)

        merged_path = os.path.join(self.extract_dir, "authors", "merged_authors.txt")
        self.assertTrue(os.path.exists(merged_path), "run_all 未生成合并文件")
        print("  -> run_all 批量运行测试通过！")


class TestRealDataAuthorPipeline(unittest.TestCase):
    """
    使用 data/ 目录下的真实 WoS 数据，完整运行 AU (作者) 字段的
    提取 -> 合并 -> 绘制作者共现图谱 全流程。

    输出结果保存在项目 extract/authors/ 目录下：
      - merged_authors.txt      : 合并后的作者数据
      - merged_authors_map.png  : 作者共现网络图谱
    """

    # 使用项目真实路径
    REAL_SOURCE_DIR = os.path.join(PROJECT_ROOT, "data")
    REAL_EXTRACT_DIR = os.path.join(PROJECT_ROOT, "extract")
    AUTHORS_DIR = os.path.join(REAL_EXTRACT_DIR, "authors")

    def test_step1_extract_au_from_real_data(self):
        """步骤1: 从真实 WoS 数据中提取 AU 字段"""
        print("\n" + "=" * 60)
        print("  [真实数据] Step 1: 提取 AU 字段")
        print("=" * 60)

        # 确认真实数据目录存在且有文件
        self.assertTrue(
            os.path.exists(self.REAL_SOURCE_DIR),
            f"真实数据目录不存在: {self.REAL_SOURCE_DIR}"
        )
        txt_files = [f for f in os.listdir(self.REAL_SOURCE_DIR) if f.endswith(".txt")]
        self.assertGreater(len(txt_files), 0, "data/ 目录下没有 .txt 文件")
        print(f"  发现 {len(txt_files)} 个原始 WoS 数据文件")

        # 执行提取
        extractor = WosFieldExtractor(
            source_dir=self.REAL_SOURCE_DIR,
            base_output_dir=self.REAL_EXTRACT_DIR,
            target_name="AU",
            max_workers=100
        )
        extractor.extract()

        # 验证提取结果
        self.assertTrue(os.path.exists(self.AUTHORS_DIR), "authors 输出目录未创建")
        extracted_files = [f for f in os.listdir(self.AUTHORS_DIR) if f.endswith(".txt")]
        self.assertGreater(len(extracted_files), 0, "未提取到任何作者数据文件")
        print(f"  -> 成功提取 {len(extracted_files)} 个作者数据文件")

    def test_step2_merge_au_files(self):
        """步骤2: 合并所有提取的作者文件"""
        print("\n" + "=" * 60)
        print("  [真实数据] Step 2: 合并作者文件")
        print("=" * 60)

        # 先确保 step1 的结果存在（若单独运行此测试则先提取）
        if not os.path.exists(self.AUTHORS_DIR):
            self.test_step1_extract_au_from_real_data()

        merged_filename = "merged_authors.txt"
        merger = TextFilesMerger(
            source_dir=self.AUTHORS_DIR,
            output_name=merged_filename
        )
        merger.merge()

        # 验证合并结果
        merged_path = os.path.join(self.AUTHORS_DIR, merged_filename)
        self.assertTrue(os.path.exists(merged_path), f"合并文件未生成: {merged_path}")

        with open(merged_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertGreater(len(lines), 0, "合并文件为空")
        print(f"  -> 合并完成，共 {len(lines)} 条作者记录")

        # 抽样检查格式：每行应以数字编号开头
        for line in lines[:5]:
            self.assertRegex(line.strip(), r"^\d+\.", f"行格式不正确: {line.strip()}")

    def test_step3_draw_author_graph(self):
        """步骤3: 基于合并后的作者数据绘制共现网络图谱"""
        print("\n" + "=" * 60)
        print("  [真实数据] Step 3: 绘制作者共现图谱")
        print("=" * 60)

        merged_path = os.path.join(self.AUTHORS_DIR, "merged_authors.txt")

        # 先确保合并文件存在
        if not os.path.exists(merged_path):
            self.test_step2_merge_au_files()

        png_path = os.path.join(self.AUTHORS_DIR, "merged_authors_map.png")

        engine = GraphAnalysisEngine(max_workers=100)
        engine.run(
            file_path=merged_path,
            output_png=png_path,
            target="AU",
            top_k=300,
            threshold=0.7
        )

        # 验证图谱文件生成
        self.assertTrue(os.path.exists(png_path), f"作者图谱未生成: {png_path}")

        # 验证 PNG 文件大小合理（至少 10KB，非空图片）
        file_size = os.path.getsize(png_path)
        self.assertGreater(file_size, 10 * 1024, f"图谱文件过小 ({file_size} bytes)，可能生成异常")
        print(f"  -> 作者图谱生成成功: {png_path}")
        print(f"     文件大小: {file_size / 1024:.1f} KB")

    def test_full_au_pipeline_with_runner(self):
        """完整端到端测试: 使用 PipelineRunner 一次性完成 AU 的提取→合并→绘图"""
        print("\n" + "=" * 60)
        print("  [真实数据] 完整 AU Pipeline 端到端测试")
        print("=" * 60)

        runner = PipelineRunner(
            source_dir=self.REAL_SOURCE_DIR,
            base_extract_dir=self.REAL_EXTRACT_DIR,
            max_workers=100
        )
        runner.run("AU", top_k=300, threshold=0.7)

        # 验证最终产物
        merged_path = os.path.join(self.AUTHORS_DIR, "merged_authors.txt")
        png_path = os.path.join(self.AUTHORS_DIR, "merged_authors_map.png")

        self.assertTrue(os.path.exists(merged_path), "合并文件未生成")
        self.assertTrue(os.path.exists(png_path), "作者图谱未生成")

        with open(merged_path, "r", encoding="utf-8") as f:
            line_count = len(f.readlines())
        file_size_kb = os.path.getsize(png_path) / 1024

        print(f"  -> 合并文件: {merged_path} ({line_count} 条记录)")
        print(f"  -> 作者图谱: {png_path} ({file_size_kb:.1f} KB)")
        print("  -> 完整 AU Pipeline 端到端测试通过！")


class TestRealDataKeywordPipeline(unittest.TestCase):
    """
    使用 data/ 目录下的真实 WoS 数据，完整运行 DE (关键词) 字段的
    提取 -> 合并 -> 语义聚类 -> 绘制关键词共现图谱 全流程。

    与 AU 不同，DE 字段在 Step 3 会触发语义合并 (SentenceTransformer)，
    将语义相近的关键词聚类为同一代表词，再构建共现图谱。

    每一步打印调用的类、源文件路径、输入/输出文件路径，方便 review。

    调用链路:
    ┌──────────────────────────────────────────────────────────────────┐
    │ Step 1: WosFieldExtractor     (FileProcess/file_extract.py)     │
    │         从 data/*.txt 提取 DE 字段                               │
    │         输出 → extract/keywords/LLM *.txt                       │
    │                                                                  │
    │ Step 2: TextFilesMerger       (FileProcess/file_merge.py)       │
    │         合并所有提取文件并重新编号                                  │
    │         输出 → extract/keywords/merged_keywords.txt              │
    │                                                                  │
    │ Step 3: GraphAnalysisEngine   (visualize/analysis.py)           │
    │   3a. KeywordProcessor.get_top_k_raw (utils/processor.py)       │
    │       多线程统计词频，返回 top_k 高频关键词                        │
    │   3b. KeywordProcessor.semantic_merge (utils/processor.py)      │
    │       加载 SentenceTransformer 模型，语义聚类相似关键词             │
    │       返回 valid_set + mapping_dict                              │
    │   3c. _build_single_subgraph  (visualize/analysis.py)           │
    │       逐行构建 WeightedGraph 子图 (entity/graph.py)              │
    │   3d. WeightedGraph.merge_from (entity/graph.py)                │
    │       并行归约合并所有子图为一张总图                                │
    │   3e. WeightedGraph.visualize  (entity/graph.py)                │
    │       NetworkX + Matplotlib 绘制共现图谱                          │
    │       输出 → extract/keywords/merged_keywords_map.png            │
    └──────────────────────────────────────────────────────────────────┘
    """

    REAL_SOURCE_DIR = os.path.join(PROJECT_ROOT, "data")
    REAL_EXTRACT_DIR = os.path.join(PROJECT_ROOT, "extract")
    KEYWORDS_DIR = os.path.join(REAL_EXTRACT_DIR, "keywords")

    def _print_step_header(self, step_num, title, cls_name, source_file):
        """统一打印步骤头信息"""
        print(f"\n{'━' * 70}")
        print(f"  Step {step_num}: {title}")
        print(f"  调用类   : {cls_name}")
        print(f"  源文件   : {source_file}")
        print(f"{'━' * 70}")

    def _print_files(self, label, file_list, limit=10):
        """打印文件列表（最多显示 limit 个）"""
        print(f"  {label} ({len(file_list)} 个文件):")
        for f in file_list[:limit]:
            print(f"    - {f}")
        if len(file_list) > limit:
            print(f"    ... 省略 {len(file_list) - limit} 个文件")

    # ------------------------------------------------------------------
    # Step 1: 提取 DE 字段
    # ------------------------------------------------------------------
    def test_step1_extract_de_from_real_data(self):
        """
        Step 1: WosFieldExtractor 从真实 WoS 数据中提取 DE (关键词) 字段

        调用类: WosFieldExtractor  (FileProcess/file_extract.py)
        输入: data/*.txt  (48 个原始 WoS 数据文件)
        输出: extract/keywords/*.txt  (每个原始文件对应一个提取结果)
        """
        self._print_step_header(
            1, "提取 DE (关键词) 字段",
            "WosFieldExtractor", "FileProcess/file_extract.py"
        )

        # 显示输入数据
        self.assertTrue(os.path.exists(self.REAL_SOURCE_DIR),
                        f"数据目录不存在: {self.REAL_SOURCE_DIR}")
        source_files = sorted([f for f in os.listdir(self.REAL_SOURCE_DIR)
                               if f.endswith(".txt")])
        self.assertGreater(len(source_files), 0, "data/ 下没有 .txt 文件")

        print(f"  输入目录 : {self.REAL_SOURCE_DIR}")
        self._print_files("输入文件", source_files)
        print(f"  输出目录 : {self.KEYWORDS_DIR}")

        # 执行提取
        extractor = WosFieldExtractor(
            source_dir=self.REAL_SOURCE_DIR,
            base_output_dir=self.REAL_EXTRACT_DIR,
            target_name="DE",
            max_workers=100
        )
        print(f"\n  >>> extractor = WosFieldExtractor("
              f"source_dir='{self.REAL_SOURCE_DIR}', "
              f"base_output_dir='{self.REAL_EXTRACT_DIR}', "
              f"target_name='DE', max_workers=100)")
        print(f"  >>> extractor.extract()\n")
        extractor.extract()

        # 验证输出
        self.assertTrue(os.path.exists(self.KEYWORDS_DIR), "keywords 目录未创建")
        extracted_files = sorted([f for f in os.listdir(self.KEYWORDS_DIR)
                                  if f.endswith(".txt") and not f.startswith("merged")])
        self.assertGreater(len(extracted_files), 0, "未提取到任何关键词文件")
        self._print_files("输出文件", extracted_files)

        # 抽样显示一个提取文件的前5行
        sample_file = os.path.join(self.KEYWORDS_DIR, extracted_files[0])
        with open(sample_file, "r", encoding="utf-8") as f:
            sample_lines = f.readlines()[:5]
        print(f"\n  抽样 [{extracted_files[0]}] 前 {len(sample_lines)} 行:")
        for line in sample_lines:
            print(f"    {line.rstrip()}")

        print(f"\n  ✅ Step 1 完成: 成功提取 {len(extracted_files)} 个关键词文件")

    # ------------------------------------------------------------------
    # Step 2: 合并文件
    # ------------------------------------------------------------------
    def test_step2_merge_de_files(self):
        """
        Step 2: TextFilesMerger 合并所有提取的关键词文件

        调用类: TextFilesMerger  (FileProcess/file_merge.py)
        输入: extract/keywords/*.txt  (排除 merged_*.txt)
        输出: extract/keywords/merged_keywords.txt
        """
        self._print_step_header(
            2, "合并关键词文件",
            "TextFilesMerger", "FileProcess/file_merge.py"
        )

        # 确保 step1 结果存在
        if not os.path.exists(self.KEYWORDS_DIR):
            self.test_step1_extract_de_from_real_data()

        input_files = sorted([f for f in os.listdir(self.KEYWORDS_DIR)
                              if f.endswith(".txt") and not f.startswith("merged")])
        merged_filename = "merged_keywords.txt"
        merged_path = os.path.join(self.KEYWORDS_DIR, merged_filename)

        print(f"  输入目录 : {self.KEYWORDS_DIR}")
        self._print_files("待合并文件", input_files)
        print(f"  输出文件 : {merged_path}")

        # 执行合并
        merger = TextFilesMerger(
            source_dir=self.KEYWORDS_DIR,
            output_name=merged_filename
        )
        print(f"\n  >>> merger = TextFilesMerger("
              f"source_dir='{self.KEYWORDS_DIR}', "
              f"output_name='{merged_filename}')")
        print(f"  >>> merger.merge()\n")
        merger.merge()

        # 验证
        self.assertTrue(os.path.exists(merged_path), f"合并文件未生成: {merged_path}")
        with open(merged_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertGreater(len(lines), 0, "合并文件为空")

        # 抽样检查格式
        for line in lines[:3]:
            self.assertRegex(line.strip(), r"^\d+\.", f"行格式不正确: {line.strip()}")

        # 显示合并结果摘要
        print(f"\n  合并结果: {len(lines)} 条关键词记录")
        print(f"  文件大小: {os.path.getsize(merged_path) / 1024:.1f} KB")
        print(f"  前 5 行预览:")
        for line in lines[:5]:
            print(f"    {line.rstrip()}")
        print(f"  ...")
        for line in lines[-3:]:
            print(f"    {line.rstrip()}")

        print(f"\n  ✅ Step 2 完成: 合并产出 {merged_path}")

    # ------------------------------------------------------------------
    # Step 3: 可视化分析（含语义合并）
    # ------------------------------------------------------------------
    def test_step3_draw_keyword_graph(self):
        """
        Step 3: GraphAnalysisEngine 对合并后的关键词数据进行可视化分析

        调用类及调用链:
          GraphAnalysisEngine          (visualize/analysis.py)
            └─ KeywordProcessor        (utils/processor.py)
                ├─ .get_top_k_raw()    → 多线程统计词频，返回 top_k
                └─ .semantic_merge()   → SentenceTransformer 语义聚类
            └─ WeightedGraph           (entity/graph.py)
                ├─ .build_from_list()  → 构建子图
                ├─ .merge_from()       → 并行归约合并
                └─ .visualize()        → NetworkX + Matplotlib 绘图

        输入: extract/keywords/merged_keywords.txt
        输出: extract/keywords/merged_keywords_map.png
        """
        self._print_step_header(
            3, "可视化分析 (含语义合并)",
            "GraphAnalysisEngine", "visualize/analysis.py"
        )

        merged_path = os.path.join(self.KEYWORDS_DIR, "merged_keywords.txt")
        if not os.path.exists(merged_path):
            self.test_step2_merge_de_files()

        png_path = os.path.join(self.KEYWORDS_DIR, "merged_keywords_map.png")

        print(f"  输入文件 : {merged_path}")
        print(f"  输出图谱 : {png_path}")
        print()
        print(f"  内部调用链路:")
        print(f"  ┌─ GraphAnalysisEngine.__init__(max_workers=100)")
        print(f"  │    └─ self.processor = KeywordProcessor()    [utils/processor.py]")
        print(f"  │")
        print(f"  ├─ GraphAnalysisEngine.run()                   [visualize/analysis.py]")
        print(f"  │    ├─ 3a. processor.get_top_k_raw()          [utils/processor.py]")
        print(f"  │    │       → 多线程解析每行 [keyword] 并统计词频")
        print(f"  │    │       → 返回 top_k=300 高频关键词")
        print(f"  │    │")
        print(f"  │    ├─ 3b. processor.semantic_merge()          [utils/processor.py]")
        print(f"  │    │       → 加载 SentenceTransformer ('all-MiniLM-L6-v2')")
        print(f"  │    │       → 编码所有关键词为向量")
        print(f"  │    │       → 计算余弦相似度矩阵")
        print(f"  │    │       → 阈值=0.7 语义聚类，返回 valid_set + mapping")
        print(f"  │    │")
        print(f"  │    ├─ 3c. _build_single_subgraph() × N 行    [visualize/analysis.py]")
        print(f"  │    │       → parse_line() 解析关键词            [utils/processor.py]")
        print(f"  │    │       → 过滤 valid_set + mapping 映射")
        print(f"  │    │       → WeightedGraph.build_from_list()   [entity/graph.py]")
        print(f"  │    │")
        print(f"  │    ├─ 3d. 并行归约 merge_from()                [entity/graph.py]")
        print(f"  │    │       → 两两合并子图直到剩余 1 张总图")
        print(f"  │    │")
        print(f"  │    └─ 3e. WeightedGraph.visualize()            [entity/graph.py]")
        print(f"  │           → NetworkX 图 + Louvain 社区发现")
        print(f"  │           → Matplotlib 绘制 + 保存 PNG")
        print(f"  └─ 完成")

        # 执行
        engine = GraphAnalysisEngine(max_workers=100)
        print(f"\n  >>> engine = GraphAnalysisEngine(max_workers=100)")
        print(f"  >>> engine.run(file_path='{merged_path}', "
              f"output_png='{png_path}', target='DE', top_k=300, threshold=0.7)\n")
        engine.run(
            file_path=merged_path,
            output_png=png_path,
            target="DE",
            top_k=300,
            threshold=0.7
        )

        # 验证
        self.assertTrue(os.path.exists(png_path), f"关键词图谱未生成: {png_path}")
        file_size = os.path.getsize(png_path)
        self.assertGreater(file_size, 10 * 1024,
                           f"图谱文件过小 ({file_size} bytes)，可能生成异常")

        print(f"\n  ✅ Step 3 完成: 关键词图谱已生成")
        print(f"     文件路径: {png_path}")
        print(f"     文件大小: {file_size / 1024:.1f} KB")

    # ------------------------------------------------------------------
    # 完整端到端 (使用 PipelineRunner 一键执行)
    # ------------------------------------------------------------------
    def test_full_de_pipeline_with_runner(self):
        """
        完整端到端: 使用 PipelineRunner 一次性完成 DE 的提取→合并→绘图

        调用类: PipelineRunner  (run_process.py)
        内部依次调用:
          → WosFieldExtractor.extract()     [FileProcess/file_extract.py]
          → TextFilesMerger.merge()         [FileProcess/file_merge.py]
          → GraphAnalysisEngine.run()       [visualize/analysis.py]
            → KeywordProcessor              [utils/processor.py]
            → WeightedGraph                 [entity/graph.py]
        """
        print(f"\n{'━' * 70}")
        print(f"  完整端到端: PipelineRunner 运行 DE Pipeline")
        print(f"  调用类   : PipelineRunner  (run_process.py)")
        print(f"{'━' * 70}")
        print(f"  输入数据 : {self.REAL_SOURCE_DIR}")
        print(f"  输出目录 : {self.REAL_EXTRACT_DIR}")
        print()
        print(f"  PipelineRunner.run('DE') 内部调用顺序:")
        print(f"    1. WosFieldExtractor.extract()     → FileProcess/file_extract.py")
        print(f"    2. TextFilesMerger.merge()          → FileProcess/file_merge.py")
        print(f"    3. GraphAnalysisEngine.run()        → visualize/analysis.py")
        print(f"       ├─ KeywordProcessor.get_top_k_raw()  → utils/processor.py")
        print(f"       ├─ KeywordProcessor.semantic_merge()  → utils/processor.py")
        print(f"       ├─ WeightedGraph.build_from_list()   → entity/graph.py")
        print(f"       ├─ WeightedGraph.merge_from()        → entity/graph.py")
        print(f"       └─ WeightedGraph.visualize()         → entity/graph.py")

        runner = PipelineRunner(
            source_dir=self.REAL_SOURCE_DIR,
            base_extract_dir=self.REAL_EXTRACT_DIR,
            max_workers=100
        )
        print(f"\n  >>> runner = PipelineRunner(source_dir='{self.REAL_SOURCE_DIR}', "
              f"base_extract_dir='{self.REAL_EXTRACT_DIR}', max_workers=100)")
        print(f"  >>> runner.run('DE', top_k=300, threshold=0.7)\n")
        runner.run("DE", top_k=300, threshold=0.7)

        # 验证最终产物
        merged_path = os.path.join(self.KEYWORDS_DIR, "merged_keywords.txt")
        png_path = os.path.join(self.KEYWORDS_DIR, "merged_keywords_map.png")

        self.assertTrue(os.path.exists(merged_path), "合并文件未生成")
        self.assertTrue(os.path.exists(png_path), "关键词图谱未生成")

        with open(merged_path, "r", encoding="utf-8") as f:
            line_count = len(f.readlines())
        file_size_kb = os.path.getsize(png_path) / 1024

        print(f"\n  {'─' * 50}")
        print(f"  最终产物清单:")
        print(f"    [提取结果]  {self.KEYWORDS_DIR}/*.txt")
        print(f"    [合并文件]  {merged_path}  ({line_count} 条记录)")
        print(f"    [关键词图谱] {png_path}  ({file_size_kb:.1f} KB)")
        print(f"  {'─' * 50}")
        print(f"  ✅ 完整 DE Pipeline 端到端测试通过！")


if __name__ == "__main__":
    print("=" * 60)
    print("  run_process.py 重构后的单元测试")
    print("=" * 60)
    unittest.main(verbosity=2)
