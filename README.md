# Statistical-significance-test|显著性检验测试
This is a statistical analysis script designed for multi-algorithm comparison experiments. Its core functionality includes the Friedman overall significance test, the Iman-Davenport correction, the Wilcoxon signed-rank test, as well as the Nemenyi post-hoc test and critical difference diagram visualization.

The script takes a "multi-dataset × multi-algorithm" score matrix as input, first assessing whether there are significant overall differences between algorithms. It then identifies the differences between the target algorithm and each baseline method, outputting average ranks, significance results, and graphical files. This script is suitable for reproducible clustering algorithm experiments, method comparisons, and result reporting in research papers.

这是一个可用于多算法对比实验的统计分析脚本，核心功能包括 Friedman 总体显著性检验、Iman-Davenport 修正、 Wilcoxon 配对符号秩检验，以及 Nemenyi 事后检验与临界差异图可视化。

脚本以“多数据集 × 多算法”的评分矩阵为输入，先判断算法整体差异是否显著，再定位目标算法与各基线方法之间的差异，并输出平均秩、显著性结果和图形文件，适合用于聚类算法实验的论文复现、方法对比与结果汇报。

The Nemenyi post-hoc test visualization diagram.
Nemenyi可视化效果图：
<img width="1008" height="373" alt="image" src="https://github.com/user-attachments/assets/d9af0bfa-ea1b-40d5-9017-8cbfddd64c3a" />


