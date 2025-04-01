"""
Contain some omics data preprocess functions
"""
import pandas as pd
import numpy as np


# def separate_B(B_df_single):
#     """
#     Separate the DNA methylation dataframe into subsets according to their targeting chromosomes

#     Parameters:
#         B_df_single(DataFrame) -- a dataframe that contains the single DNA methylation matrix

#     Return:
#         B_df_list(list) -- a list with 23 subset dataframe
#         B_dim(list) -- the dims of each chromosome
#     """
#     # anno = pd.read_csv('./anno/B_anno.csv', dtype={'CHR': str}, index_col=0)
#     anno = pd.read_csv('./anno/B_anno.csv', sep='\t', dtype={'chrom': str}, index_col=0)
#     anno = anno.rename(columns={"chrom": "CHR"})  # 匹配你代码里后面用的字段
#     anno_contain = anno.loc[B_df_single.index, :]
#     print('Separating B.tsv according the targeting chromosome...')
#     B_df_list, B_dim_list = [], []
#     ch_id = list(range(1, 23))
#     ch_id.append('X')
#     for ch in ch_id:
#         # ch_index = anno_contain[anno_contain.CHR == str(ch)].index
#         ch_index = anno_contain[anno_contain["chrom"] == str(ch)].index
#         ch_df = B_df_single.loc[ch_index, :]
#         B_df_list.append(ch_df)
#         B_dim_list.append(len(ch_df))

#     return B_df_list, B_dim_list

def separate_B(B_df_single):
    """
    Separate the DNA methylation dataframe into subsets according to their targeting chromosomes

    Parameters:
        B_df_single (DataFrame): DNA methylation beta value matrix (probes as rows)

    Returns:
        B_df_list (list of np.ndarray): list of arrays separated by chromosome
        B_dim_list (list of int): dimension (number of probes) per chromosome
    """

    # 读取注释文件
    anno = pd.read_csv('./anno/B_anno.csv', sep='\t', index_col=0)
    
    # 标准化列名，确保包含 'CHR' 字段
    if 'chrom' in anno.columns:
        anno = anno.rename(columns={"chrom": "CHR"})
    
    if 'CHR' not in anno.columns:
        raise ValueError("B_anno.csv must contain a column named 'CHR' or 'chrom'.")

    # 输出用于调试
    print(f"Number of probes in B.tsv: {B_df_single.shape[0]}")
    print(f"Number of probes in anno: {anno.shape[0]}")

    # 只保留存在于 B_df_single 的 probe 注释
    anno_contain = anno.loc[B_df_single.index.intersection(anno.index)]
    print(f"Number of matching probes: {anno_contain.shape[0]}")

    # 初始化输出
    B_df_list = []
    B_dim_list = []

    # 构建染色体列表 ['chr1', ..., 'chr22', 'chrX']
    ch_id = ['chr' + str(i) for i in range(1, 23)] + ['chrX']

    print('Separating B.tsv according to the targeting chromosome...')

    for ch in ch_id:
        ch_index = anno_contain[anno_contain['CHR'] == ch].index
        ch_df = B_df_single.loc[ch_index]

        # 替换空字符串为空值，再填充为 0，最后转为 float32 数组
        ch_df = ch_df.replace('', np.nan).fillna(0.0)
        ch_array = ch_df.to_numpy().astype(np.float32)

        B_df_list.append(ch_array)
        B_dim_list.append(ch_array.shape[0])

    return B_df_list, B_dim_list