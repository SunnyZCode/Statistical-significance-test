# Statistical Significance Test | 显著性检验测试

This script performs statistical analysis for multi-algorithm comparison experiments. Its core functionalities include the **Friedman overall significance test**, the **Iman-Davenport correction**, the **Wilcoxon signed-rank test**, as well as the **Nemenyi post-hoc test** and **critical difference (CD) diagram visualization**.

The script takes a "multi-dataset × multi-algorithm" score matrix as input. It first evaluates whether there are significant overall differences between algorithms, then identifies the differences between a target algorithm and each baseline method. The outputs include **average ranks**, **significance results**, and **graphical visualizations**. This tool is suitable for reproducible clustering experiments, method comparisons, and result reporting in research papers.

---

本脚本用于多算法对比实验的统计分析。核心功能包括 **Friedman 总体显著性检验**、**Iman-Davenport 修正**、**Wilcoxon 配对符号秩检验**，以及 **Nemenyi 事后检验** 和 **临界差异图（CD 图）可视化**。

脚本以“多数据集 × 多算法”的评分矩阵作为输入，先评估算法整体差异是否显著，再定位目标算法与各基线方法之间的差异。输出结果包括 **平均秩**、**显著性检验结果** 和 **图形可视化文件**。该工具适用于聚类算法实验复现、方法对比及论文结果汇报。

---

### Example Visualization | Nemenyi 可视化效果图

<img width="500" alt="Nemenyi Critical Difference Diagram" src="https://github.com/user-attachments/assets/d9af0bfa-ea1b-40d5-9017-8cbfddd64c3a" />


If you find this repository useful for your research, please consider citing our paper:

```bibtex
@article{EGBDPM,
  title = {EGBDPM: Efficient granular ball density peaks clustering for manifold data},
  journal = {Neurocomputing},
  volume = {682},
  pages = {133427},
  year = {2026},
  issn = {0925-2312},
  doi = {https://doi.org/10.1016/j.neucom.2026.133427},
  url = {https://www.sciencedirect.com/science/article/pii/S0925231226008246},
  author = {Xingguo Zhang and Li Xu and Weikuan Jia}
}
