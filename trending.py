import os
import re
import string
from datetime import datetime
from collections import defaultdict, Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import bigrams, trigrams
import nltk

# 首次运行需要下载nltk数据
nltk.download('punkt')
nltk.download('stopwords')

# 配置参数
DIR_PATH = "./"
CUSTOM_STOPWORDS = ["arxiv", "paper", "show", "using"]
MIN_WORD_LENGTH = 3  
PHRASE_MIN_FREQ = 3  # 词组最低出现频次
OUTPUT_DIR = "./phrase_clouds"

def extract_phrases(tokens):
    """提取高频词组"""
    # 生成候选词组
    bi_grams = list(bigrams(tokens))
    tri_grams = list(trigrams(tokens))
    
    # 统计词组频率
    phrase_counter = Counter()
    for gram in bi_grams + tri_grams:
        phrase = ' '.join(gram)
        phrase_counter[phrase] += 1
    
    # 过滤低频词组
    return [p for p, c in phrase_counter.items() if c >= PHRASE_MIN_FREQ]

def process_text(text):
    """改进的文本处理流程"""
    # 移除Markdown语法
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'[#!*\-\[\]\(\)]', ' ', text)
    
    # 基础处理
    text = text.lower()
    tokens = word_tokenize(text)
    
    # 停用词过滤
    stop_words = set(stopwords.words('english') + CUSTOM_STOPWORDS
    tokens = [word for word in tokens if (
        word not in stop_words and
        word not in string.punctuation and
        len(word) >= MIN_WORD_LENGTH and
        not word.isdigit()
    )]
    
    # 提取高频词组
    phrases = extract_phrases(tokens)
    
    # 合并词和词组
    return tokens + phrases

def analyze_file(file_path):
    """文件分析流程"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取正文内容
    main_content = re.split(r'\n## ', content)[1:] if '##' in content else content
    processed = process_text(' '.join(main_content))
    
    return Counter(processed)

def generate_phrase_cloud(freq_dict, filename):
    """生成词组词云"""
    wc = WordCloud(
        width=2000,
        height=1500,
        background_color='white',
        colormap='nipy_spectral',
        max_words=150,
        collocations=False,  # 重要！禁用默认的词组生成
        prefer_horizontal=0.8  # 更适合显示词组
    ).generate_from_frequencies(freq_dict)
    
    plt.figure(figsize=(25, 18))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight', dpi=300)
    plt.close()

def process_files():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    monthly_stats = defaultdict(Counter)

    for filename in os.listdir(DIR_PATH):
        if not filename.endswith('.md'):
            continue
        
        # 解析日期
        try:
            date_part = '-'.join(filename.split('-')[:3])
            date = datetime.strptime(date_part, "%Y-%m-%d")
            month_key = date.strftime("%Y-%m")
        except Exception as e:
            print(f"跳过文件 {filename}: {str(e)}")
            continue
        
        # 处理文件
        file_path = os.path.join(DIR_PATH, filename)
        counter = analyze_file(file_path)
        monthly_stats[month_key] += counter
    
    # 生成词云
    for month, counter in monthly_stats.items():
        if counter:
            # 过滤单字符和纯数字
            filtered = {k:v for k,v in counter.items() 
                       if len(k) > 2 and not k.isdigit()}
            generate_phrase_cloud(filtered, f"{month}_phrase_cloud.png")

if __name__ == "__main__":
    process_files()
    print(f"处理完成，词云保存至：{OUTPUT_DIR}")
