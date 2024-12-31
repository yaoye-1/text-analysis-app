import streamlit as st
import requests
from bs4 import BeautifulSoup
import jieba
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pyecharts.charts import WordCloud, Bar, Line, Pie, Scatter, Radar, HeatMap
from pyecharts import options as opts
import string
import re
import pandas as pd
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号


# 对文本分词
def word_segmentation(text):
    words = jieba.lcut(text)
    words = [word for word in words if len(word) > 1 and not word.isspace()]
    return words


# 请求URL抓取文本内容
def fetch_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # 移除脚本和样式元素
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()

    # 获取纯文本
    text = soup.get_text()
    return text


# 清理文本，去除标点符号和其他不需要的字符
def clean_text(text):
    punctuation_to_remove = string.punctuation + "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟📐｢｣、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟⚗︎〾〿–—‘’‛“”„‟…‧﹏."
    cleaned_text = re.sub(r'[{}]+'.format(re.escape(punctuation_to_remove)), '', text)
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text


# 对文本分词, 统计词频
def count_word_frequency(words):
    counter = Counter(words)
    return counter

def plot_line(word_freq):
    line = (
        Line()
        .add_xaxis(list(word_freq.keys()))
        .add_yaxis("频率", list(word_freq.values()))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="词频变化趋势"),
            yaxis_opts=opts.AxisOpts(name="频率"),
            xaxis_opts=opts.AxisOpts(name="词汇"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
            datazoom_opts=[opts.DataZoomOpts()],
        )
    )
    return line
# 使用Matplotlib绘制柱状图
def plot_bar_with_matplotlib(word_freq):
    fig, ax = plt.subplots(figsize=(10, 7))

    # 如果word_freq是字典，则先排序再取前20项
    sorted_items = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:20]
    words, freqs = zip(*sorted_items) if sorted_items else ([], [])

    # 创建条形图
    ax.barh(words, freqs, color='skyblue')
    ax.invert_yaxis()  # labels read top-to-bottom

    # 明确设置y轴的标签
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)

    ax.set_xlabel('频率')
    ax.set_title('词频排名前20')

    # 调整布局以防止标签被裁剪
    plt.tight_layout()

    # 旋转标签以适应空间
    plt.xticks(rotation=45)

    st.pyplot(fig)

# 使用Seaborn绘制热力图
def plot_heatmap_with_seaborn(word_freq):
    data = pd.DataFrame(list(word_freq.items()), columns=['词汇', '频率'])
    data = data.pivot_table(index='词汇', values='频率', aggfunc='sum').fillna(0)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(data, annot=True, fmt=".1f", cmap="YlGnBu")
    ax.set_title('词频热力图')
    st.pyplot(fig)


# 使用Pyecharts绘制词云
def plot_wordcloud_with_pyecharts(word_freq):
    word_cloud = (
        WordCloud()
        .add(series_name="词汇频率", data_pair=word_freq.items(), word_size_range=[20, 100])
        .set_global_opts(
            title_opts=opts.TitleOpts(title="词频分布词云"),
            tooltip_opts=opts.TooltipOpts(is_show=True),
        )
    )
    return word_cloud


# 使用Pyecharts绘制饼图
def plot_pie_with_pyecharts(word_freq):
    pie = (
        Pie()
        .add("", list(word_freq.items())[:20])
        .set_global_opts(
            title_opts=opts.TitleOpts(title="词频比例"),
            legend_opts=opts.LegendOpts(orient="vertical", pos_top="15%", pos_left="2%"),
        )
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c} ({d}%)"))
    )
    return pie


# 使用Pyecharts绘制散点图
def plot_scatter_with_pyecharts(word_freq):
    scatter = (
        Scatter()
        .add_xaxis(list(word_freq.keys()))
        .add_yaxis("频率", list(word_freq.values()))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="词频散点图"),
            yaxis_opts=opts.AxisOpts(name="频率"),
            xaxis_opts=opts.AxisOpts(name="词汇"),
            toolbox_opts=opts.ToolboxOpts(is_show=True),
        )
    )
    return scatter


# 使用Pyecharts绘制雷达图
def plot_radar_with_pyecharts(word_freq):
    top_words = list(word_freq.items())[:6]
    if not top_words:
        return None

    schema = [{"name": word, "max": max(word_freq.values())} for word, _ in top_words]

    radar = (
        Radar()
        .add_schema(schema)
        .add("频率", [[freq for _, freq in top_words]])
        .set_global_opts(title_opts=opts.TitleOpts(title="词频雷达图"))
    )
    return radar


# 使用Pyecharts绘制热力图
def plot_heatmap_with_pyecharts(word_freq):
    heatmap_data = [(i, 0, freq) for i, (word, freq) in enumerate(word_freq.items())]
    heatmap = (
        HeatMap()
        .add_xaxis(list(word_freq.keys()))
        .add_yaxis("频率", ["频率"], heatmap_data, label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(title="词频热力图"),
            visualmap_opts=opts.VisualMapOpts(),
        )
    )
    return heatmap


# Streamlit应用标题
st.title("文章URL词频分析")

# 用户输入URL的文本框
url = st.text_input('请输入文章的URL:', '')

# 构建Streamlit侧边栏进行图型筛选
chart_type = st.sidebar.selectbox(
    "选择图表类型",
    ["词云", "柱状图", "折线图", "饼图", "散点图", "雷达图", "热力图"]
)

# 交互过滤低频词
min_occurrences = st.sidebar.slider('最小出现次数:', 1, 100, 1)

if url:
    text_content = fetch_text_from_url(url)
    if text_content:
        cleaned_text = clean_text(text_content)
        words = word_segmentation(cleaned_text)
        word_freq = count_word_frequency(words)

        # 确保返回的是Counter对象
        filtered_words = Counter({word: freq for word, freq in word_freq.items() if freq >= min_occurrences})

        # 根据用户选择显示不同类型的图表
        chart_functions = {
            "词云": plot_wordcloud_with_pyecharts,
            "柱状图": plot_bar_with_matplotlib,
            "折线图":plot_line ,  # 如果需要，可以添加折线图绘制逻辑
            "饼图": plot_pie_with_pyecharts,
            "散点图": plot_scatter_with_pyecharts,
            "雷达图": plot_radar_with_pyecharts,
            "热力图": plot_heatmap_with_pyecharts
        }

        if chart_type in chart_functions:
            chart_func = chart_functions[chart_type]
            chart = chart_func(filtered_words)
            if chart is not None:
                if isinstance(chart, str):  # 如果是HTML字符串
                    st.components.v1.html(chart, height=650)
                else:
                    st.components.v1.html(chart.render_embed(), height=650)

        # 展示词频排名前20的词汇
        top_20_words = word_freq.most_common(20)
        st.write("Top 20 Words:")
        st.table(top_20_words)