import pandas as pd
import arxiv
import time

# --- 配置参数 ---
SEARCH_QUERY = "cat:cs.LG OR cat:cs.CL OR cat:cs.AI OR cat:cs.CV"
MAX_RESULTS = 50 # 建议把总量也稍微提高一点，比如 100
OUTPUT_FILE = '../data/raw_papers.csv' # 输出文件的路径

def fetch_arxiv_abstracts(query, max_results):
    """
    从arXiv获取论文摘要.
    """
    print(f"正在从arXiv搜索 '{query}'...")
    
    # 使用arxiv库进行搜索
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate # 按提交日期排序，获取最新的
    )

    papers_data = []
    for result in search.results():
        paper_id = result.entry_id.split('/')[-1] # 获取论文ID，例如 '2307.15043v2'
        
        # 清理摘要文本，替换换行符和多余空格
        clean_abstract = result.summary.replace('\n', ' ').strip()
        
        papers_data.append({
            'paper_id': paper_id,
            'title': result.title,
            'clean_abstract': clean_abstract
        })
        print(f"  - 已获取: {paper_id} - {result.title[:50]}...")
        # arXiv API 礼貌性延迟，防止请求过于频繁
        time.sleep(0.5) 
        
    print(f"成功获取 {len(papers_data)} 篇论文摘要。")
    return pd.DataFrame(papers_data)

def main():
    """
    主函数
    """
    # 获取数据
    df_papers = fetch_arxiv_abstracts(SEARCH_QUERY, MAX_RESULTS)

    # 保存为CSV文件
    if not df_papers.empty:
        df_papers.to_csv(OUTPUT_FILE, index=False)
        print(f"数据已成功保存到: {OUTPUT_FILE}")
    else:
        print("未能获取任何数据。")

if __name__ == "__main__":
    main()