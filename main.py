"""
Friedman 显著性检验 + Wilcoxon 配对检验 + Nemenyi 可视化
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, f, rankdata, wilcoxon, studentized_range
import importlib

def mean_ranks_by_dataset(scores: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
	"""
	对每个数据集（每一行）进行秩转换，再对全部数据集求平均秩。
	返回 shape=(n_algorithms, ) 的平均秩，值越小排名越靠前。
	"""
	# rankdata 默认越小秩越小，所以"越大越好"时取负号
	transformed = -scores if higher_is_better else scores
	ranks = np.apply_along_axis(rankdata, 1, transformed, method="average")
	return ranks.mean(axis=0)

def cd_judgement(
	avg_ranks: np.ndarray,
	algorithm_names: list[str],
	n_datasets: int,
	alpha: float = 0.05,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
	"""
	基于 Nemenyi 的 CD进行判断。
	返回：CD 值、两两比较表、用于绘图的“p值风格矩阵”（显著=0，不显著=1）。
	"""
	k = len(algorithm_names)
	# Demsar(2006) 使用的 q_alpha 需将 studentized_range 的值除以 sqrt(2)
	q_alpha = studentized_range.isf(alpha, k, np.inf) / np.sqrt(2)
	cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n_datasets))

	rows = []
	# critical_difference_diagram 要求的是 p 值矩阵：< alpha 视为显著

	p_like_matrix = pd.DataFrame(1.0, index=algorithm_names, columns=algorithm_names)
	np.fill_diagonal(p_like_matrix.values, 1.0)
	for i in range(k):
		for j in range(i + 1, k):
			diff = abs(avg_ranks[i] - avg_ranks[j])
			sig = diff > cd
			p_like_matrix.iloc[i, j] = 0.0 if sig else 1.0
			p_like_matrix.iloc[j, i] = 0.0 if sig else 1.0
			rows.append(
				{
					"Alg_i": algorithm_names[i],
					"Alg_j": algorithm_names[j],
					"|RankDiff|": diff,
					"CD": cd,
					"Significant(基于CD)": "是" if sig else "否",
				}
			)

	pair_df = pd.DataFrame(rows).sort_values("|RankDiff|", ascending=False)
	return cd, pair_df, p_like_matrix


def friedman_test(
	scores: np.ndarray,
	algorithm_names: list[str] | None = None,
	higher_is_better: bool = True,
) -> dict:
	"""
	对每个算法在全部数据集上的结果做 Friedman 检验。
	"""
	n_datasets, n_algorithms = scores.shape
	if algorithm_names is None:
		algorithm_names = [f"Alg{i+1}" for i in range(n_algorithms)]
	if len(algorithm_names) != n_algorithms:
		raise ValueError("algorithm_names 长度与算法列数不一致")

	groups = [scores[:, i] for i in range(n_algorithms)]
	chi2_stat, p_value = friedmanchisquare(*groups)

	# Iman-Davenport 修正（更常用于ML比较）
	ff = ((n_datasets - 1) * chi2_stat) / (n_datasets * (n_algorithms - 1) - chi2_stat)
	df1 = n_algorithms - 1
	df2 = (n_algorithms - 1) * (n_datasets - 1)
	p_ff = 1 - f.cdf(ff, df1, df2)

	avg_ranks = mean_ranks_by_dataset(scores, higher_is_better=higher_is_better)
	rank_table = pd.DataFrame(
		{
			"Algorithm": algorithm_names,
			"MeanRank(越小越好)": avg_ranks,
			"MeanScore": scores.mean(axis=0),
			"StdScore": scores.std(axis=0, ddof=1),
		}
	).sort_values("MeanRank(越小越好)", ascending=True)

	return {
		"chi2": chi2_stat,
		"p": p_value,
		"iman_davenport_F": ff,
		"iman_davenport_p": p_ff,
		"rank_table": rank_table,
	}


def nemenyi_posthoc_and_plot(scores: np.ndarray, algorithm_names: list[str], save_path: str = "nemenyi_plot.png") -> pd.DataFrame | None:
	"""
	Nemenyi 事后检验并可视化。
	"""

	sp = importlib.import_module("scikit_posthocs")

	df = pd.DataFrame(scores, columns=algorithm_names)
	pvals = sp.posthoc_nemenyi_friedman(df)
	
	
	# 计算平均秩用于绘图
	ranks = np.apply_along_axis(rankdata, 1, -scores, method="average")
	avg_ranks = ranks.mean(axis=0)
	cd, pair_df, sig_matrix_cd = cd_judgement(avg_ranks, algorithm_names, n_datasets=scores.shape[0], alpha=0.05)

	print(f"CD(α=0.05) = {cd:.3f}")

		
	# 将平均秩转换为带标签的 Series
	ranks_series = pd.Series(avg_ranks, index=algorithm_names)
		
	fig, ax = plt.subplots(figsize=(10, 3))
	sp.critical_difference_diagram(
			ranks=ranks_series,
			sig_matrix=sig_matrix_cd,
			ax=ax,
			label_fmt_left='{label}',
			label_fmt_right='{label}',
			label_props={'fontsize': 12, 'fontweight': 'bold'},   #算法名称加粗
			text_h_margin=0.1,  # 减小标签与横线的垂直距离
			elbow_props={'color': 'k', 'linewidth': 1},
			crossbar_props={'color': 'red', 'linewidth': 1.5}
	)
	plt.tight_layout()
	plt.savefig(save_path, dpi=300, bbox_inches='tight')
	print(f"\n[可视化已保存] {save_path}")
	plt.close()
	return pvals


def pairwise_comparison_with_target(
	scores: np.ndarray,
	algorithm_names: list[str],
	target_algorithm: str,
) -> pd.DataFrame:
	"""
	将目标算法与其他所有算法进行配对 Wilcoxon signed-rank 检验。
	scores: shape = (N, K), 行=数据集, 列=算法
	返回包含算法名、统计量和 p 值的 DataFrame
	"""
	if target_algorithm not in algorithm_names:
		raise ValueError(f"目标算法 '{target_algorithm}' 不在算法列表中")
	
	target_idx = algorithm_names.index(target_algorithm)
	target_scores = scores[:, target_idx]
	
	results = []
	for i, alg_name in enumerate(algorithm_names):
		if i == target_idx:
			continue  # 跳过与自己的比较
		
		other_scores = scores[:, i]
		# Wilcoxon signed-rank test (配对样本)
		stat, p_value = wilcoxon(target_scores, other_scores, alternative='two-sided')
		results.append({
			"Algorithm": alg_name,
			"Statistic": stat,
			"p-value": p_value,
			"Significant(α=0.05)": "是" if p_value < 0.05 else "否",
		})
	
	return pd.DataFrame(results).sort_values("p-value")


def main():
	
	algorithm_names = [
		"DPC",
		"DPC-KNN",
		"SNNDPC",
		"FastDP",
		"GDBDSCAN",
		"GB-DP",
		"AGB-DP",
		"GBSC",
		"GBCT",
		"LGS-DPC",
	]

	scores = np.array(
	[
    [0.867, 0.788, 0.971, 0.883, 0.983, 0.979, 0.992, 0.792, 0.988, 0.992],
	[0.922, 0.879, 0.804, 0.920, 0.587, 0.571, 0.560, 0.901, 0.810, 1.000],
	[0.652, 0.655, 1.000, 0.656, 1.000, 0.554, 0.641, 0.473, 1.000, 1.000],
	[0.997, 0.999, 0.994, 0.996, 0.866, 0.696, 0.525, 0.540, 0.997, 0.999],
	[0.612, 0.612, 0.696, 0.614, 0.859, 0.447, 0.515, 0.839, 0.949, 1.000],
	[0.612, 0.504, 0.603, 0.663, 1.000, 0.524, 0.393, 1.000, 1.000, 1.000],
	[0.498, 0.510, 0.496, 0.536, 0.624, 0.398, 0.756, 0.698, 0.813, 1.000],
		
    [0.555, 0.634, 0.386, 0.584, 0.723, 0.634, 0.683, 0.495, 0.614, 0.752],
    [0.553, 0.867, 0.924, 0.887, 0.767, 0.833, 0.853, 0.847, 0.813, 0.927],
    [0.510, 0.620, 0.500, 0.567, 0.539, 0.505, 0.510, 0.505, 0.505, 0.558],
    [0.900, 0.819, 0.876, 0.907, 0.467, 0.908, 0.547, 0.557, 0.638, 0.910],
    [0.696, 0.696, 0.680, 0.739, 0.617, 0.694, 0.696, 0.545, 0.555, 0.706],
    [0.620, 0.584, 0.696, 0.900, 0.900, 0.650, 0.624, 0.863, 0.895, 0.886],
    [0.602, 0.676, 0.568, 0.601, 0.508, 0.492, 0.548, 0.545, 0.356, 0.683],
    [0.701, 0.689, 0.550, 0.733, 0.330, 0.673, 0.605, 0.736, 0.322, 0.738],
    [0.678, 0.677, 0.677, 0.677, 0.714, 0.677, 0.703, 0.677, 0.678, 0.731],
    [0.751, 0.627, 0.744, 0.757, 0.306, 0.704, 0.650, 0.572, 0.358, 0.797],
	
	],dtype=float,)


	# 1. Friedman 检验
	result = friedman_test(scores, algorithm_names=algorithm_names, higher_is_better=True)

	print("Friedman 检验结果")
	print(f"chi2 statistic     = {result['chi2']:.6f}")   # Friedman 统计量
	print(f"p-value            = {result['p']:.6g}")     # Friedman p 值
	print(f"Iman-Davenport F   = {result['iman_davenport_F']:.6f}")   # Iman-Davenport 统计量
	print(f"Iman-Davenport p   = {result['iman_davenport_p']:.6g}")   # Iman-Davenport p 值
	print(result["rank_table"].to_string(index=False))

	alpha = 0.05
	if result["p"] < alpha:
		print(f"\n结论：p < {alpha}，拒绝原假设，不同算法之间存在显著差异。")
	else:
		print(f"\n结论：p >= {alpha}，未发现算法间显著差异（但继续进行配对检验）。")
	

	# 2. LGS-DPC 与其他算法的配对 Wilcoxon 检验
	print("\n" + "=" * 60)
	print("LGS-DPC 与其他算法的配对 Wilcoxon 符号秩检验")

	# 以 LGS-DPC 作为目标算法进行配对比较
	pairwise_results = pairwise_comparison_with_target(scores, algorithm_names, "LGS-DPC")   
	print(pairwise_results.to_string(index=False))
	

	# 3. Nemenyi 事后检验及可视化
	print("\n" + "=" * 60)
	print("Nemenyi 事后检验及可视化")
	nemenyi_pvals = nemenyi_posthoc_and_plot(scores, algorithm_names, save_path="/icislab/volume1/zxg188/算法/nemenyi_plot.png")


if __name__ == "__main__":
	main()

