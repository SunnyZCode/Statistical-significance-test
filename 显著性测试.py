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

    [0.543, 0.413, 0.811, 0.552, 0.866, 0.867, 0.936, 0.243, 0.899, 0.927],  # D1
    [0.645, 0.542, 0.415, 0.637, 0.892, 0.175, 0.116, 0.57, 0.246, 1.000],    # D2
    [0.577, 0.614, 1.000, 0.575, 0.899, 0.534, 0.591, 0.125, 1.000, 1.000],   # D3
    [0.986, 0.995, 0.971, 0.980, 0.937, 0.641, 0.521, 0.277, 1.000, 0.995],   # D4
    [0.625, 0.623, 0.763, 0.689, 0.922, 0.516, 0.607, 0.912, 0.954, 1.000],   # D5
    [0.706, 0.662, 0.761, 0.716, 1.000, 0.694, 0.020, 0.989, 1.000, 1.000],   # D6
    [0.157, 0.238, 0.378, 0.275, 0.858, 0.088, 0.807, 0.744, 0.908, 1.000],   # D7
    [0.835, 0.804, 0.899, 0.804, 0.967, 0.746, 0.881, 0.732, 0.991, 1.000],   # D8

    [0.536, 0.327, 0.885, 0.586, 0.928, 0.917, 0.967, 0.323, 0.950, 0.967],  # D1
    [0.706, 0.569, 0.368, 0.696, 0.950, -0.013, -0.090, 0.619, 0.256, 1.000], # D2
    [0.416, 0.464, 1.000, 0.410, 1.000, 0.348, 0.434, 0.072, 1.000, 1.000],  # D3
    [0.403, 0.402, 0.578, 0.491, 0.890, 0.264, 0.306, 0.215, 1.000, 1.000],  # D4
    [0.429, 0.393, 0.532, 0.486, 0.889, 0.465, 0.358, 0.819, 1.000, 1.000],   # D5
    [0.069, 0.104, 0.144, 0.181, 1.000, -0.021, 0.011, 1.000, 1.000, 1.000],  # D6
    [0.768, 0.641, 0.844, 0.736, 0.713, 0.594, 0.675, 0.587, 0.734, 1.000],   # D7
    [0.990, 0.992, 0.985, 0.991, 0.988, 0.476, 0.703, 0.737, 0.997, 0.997],   # D8

    [0.864, 0.787, 0.968, 0.880, 0.982, 0.977, 0.991, 0.323, 0.986, 0.991],  
    [0.907, 0.862, 0.787, 0.904, 0.568, 0.570, 0.359, 0.619, 0.654, 1.000],  
    [0.646, 0.592, 1.000, 0.651, 1.000, 0.549, 0.633, 0.072, 1.000, 1.000],  
    [0.605, 0.602, 0.728, 0.665, 0.781, 0.271, 0.372, 0.215, 1.000, 1.000], 
    [0.621, 0.477, 0.549, 0.522, 0.794, 0.387, 0.458, 0.819, 0.918, 1.000],  
    [0.454, 0.402, 0.403, 0.500, 1.000, 0.391, 0.368, 1.000, 1.000, 1.000],  
    [0.815, 0.681, 0.757, 0.728, 0.533, 0.632, 0.627, 0.587, 0.817, 1.000],  
    [0.998, 0.998, 0.993, 0.994, 0.985, 0.558, 0.675, 0.737, 0.996, 0.999], 
	
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

    [0.668, 0.721, 0.487, 0.675, 0.721, 0.728, 0.714, 0.531, 0.572, 0.730],  
    [0.653, 0.744, 0.786, 0.763, 0.717, 0.740, 0.754, 0.714, 0.728, 0.796], 
    [0.000, 0.001, 0.001, 0.016, 0.083, 0.058, 0.058, 0.002, 0.045, 0.011],  
    [0.714, 0.619, 0.664, 0.713, 0.197, 0.712, 0.557, 0.361, 0.529, 0.715],  
    [0.137, 0.137, 0.117, 0.192, 0.064, 0.134, 0.137, 0.002, 0.015, 0.152],  
    [0.013, 0.001, 0.030, 0.001, 0.001, 0.001, 0.001, 0.003, 0.008, 0.008],  
    [0.533, 0.523, 0.458, 0.539, 0.496, 0.436, 0.507, 0.459, 0.318, 0.542],  
    [0.595, 0.616, 0.523, 0.637, 0.233, 0.610, 0.568, 0.632, 0.270, 0.632],  
    [0.235, 0.235, 0.235, 0.235, 0.251, 0.235, 0.236, 0.235, 0.112, 0.309],  
    [0.740, 0.588, 0.739, 0.732, 0.437, 0.714, 0.497, 0.597, 0.454, 0.741],  

    [0.410, 0.630, -0.050, 0.473, 0.547, 0.434, 0.559, 0.315, 0.407, 0.749],  
    [0.453, 0.644, 0.799, 0.718, 0.696, 0.636, 0.663, 0.605, 0.612, 0.802],  
    [-0.004, 0.010, 0.005, 0.001, 0.001, -0.004, -0.005, -0.005, -0.005, 0.009],
    [0.724, 0.424, -0.676, 0.747, 0.037, 0.721, 0.382, 0.298, 0.450, 0.754],  
    [0.151, 0.151, 0.082, 0.226, 0.051, 0.151, 0.162, 0.010, 0.005, 0.167],  
    [0.028, 0.133, -0.073, 0.010, 0.001, -0.006, -0.002, -0.010, -0.010, -0.023], 
    [0.337, 0.431, 0.316, 0.324, 0.352, 0.265, 0.420, 0.327, 0.100, 0.434],  
    [0.541, 0.512, 0.516, 0.520, 0.082, 0.512, 0.517, 0.588, 0.092, 0.556],  
    [0.125, 0.125, 0.125, 0.125, 0.211, 0.125, 0.216, 0.125, 0.126, 0.212],  
    [0.627, 0.581, 0.613, 0.630, 0.205, 0.562, 0.348, 0.426, 0.190, 0.643], 

    [0.434, 0.401, 0.434, 0.434, 0.626, 0.434, 0.705, 0.385, 0.435, 0.730],  
    [0.487, 0.830, 0.901, 0.884, 0.744, 0.822, 0.846, 0.841, 0.798, 0.927],  
    [0.489, 0.407, 0.497, 0.427, 0.407, 0.499, 0.481, 0.431, 0.336, 0.558],  
    [0.898, 0.516, 0.876, 0.913, 0.402, 0.905, 0.470, 0.508, 0.553, 0.907], 
    [0.669, 0.669, 0.611, 0.724, 0.609, 0.669, 0.684, 0.365, 0.388, 0.681],  
    [0.487, 0.588, 0.423, 0.481, 0.474, 0.450, 0.463, 0.489, 0.472, 0.470], 
    [0.620, 0.697, 0.571, 0.622, 0.329, 0.479, 0.642, 0.458, 0.303, 0.578],  
    [0.650, 0.646, 0.629, 0.704, 0.217, 0.650, 0.689, 0.635, 0.230, 0.570],  
    [0.630, 0.630, 0.630, 0.630, 0.701, 0.630, 0.687, 0.630, 0.606, 0.703],  
    [0.710, 0.690, 0.720, 0.714, 0.260, 0.676, 0.392, 0.529, 0.317, 0.789],  ],dtype=float,)


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

