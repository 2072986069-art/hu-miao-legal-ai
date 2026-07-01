from pypdf import PdfReader
import os

base = r"D:\hu_miao_env\docs"
files = [
    "儿童信息规定.pdf",
    "家庭促进法.pdf",
    "强制报告意见.pdf",
    "网络保护规定.pdf",
    "未成年人保护法.pdf",
    "学校保护规定.pdf",
    "预防犯罪法.pdf"
]

for f in files:
    path = os.path.join(base, f)
    if not os.path.exists(path):
        print(f"{f}: 文件不存在")
        continue
    reader = PdfReader(path)
    page_count = len(reader.pages)
    first_text = reader.pages[0].extract_text() if reader.pages else ""
    print(f"{f}: 页数={page_count}, 首页文字={first_text[:80] if first_text else '无文字'}")