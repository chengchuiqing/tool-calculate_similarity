import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document

nltk.download('punkt')


def read_docx(filename):
    doc = Document(filename)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip() != '']
    return paragraphs


def write_file(filename, content):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)


def calculate_similarity(paragraphs1, paragraphs2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(paragraphs1 + paragraphs2)

    similarity_matrix = cosine_similarity(tfidf_matrix[:len(paragraphs1)], tfidf_matrix[len(paragraphs1):])
    return similarity_matrix


def main():
    threshold = 0.5  # 相似度阈值，按需更改

    # 读取文件内容.注意源文本和带查重文本要写反了，否则打印内容也是相反的
    paragraphs_a = read_docx('A.docx')  # 原文本
    paragraphs_b = read_docx('B.docx')  # 待查重文本

    # 计算相似性
    similarity_matrix = calculate_similarity(paragraphs_a, paragraphs_b)

    # 输出相似性矩阵
    for i, row in enumerate(similarity_matrix):
        for j, similarity in enumerate(row):
            print(f"原文第{i + 1}段落 和 待查重文本第{j + 1}段落 的相似度: {similarity:.4f}")

    # 输出相似的内容
    similar_content = []
    for i, row in enumerate(similarity_matrix):
        for j, similarity in enumerate(row):
            if similarity > threshold:
                similar_content.append(
                    f"原文第{i + 1}段落:\n{paragraphs_a[i]}\n\n和\n\n待查重文本第{j + 1}段落:\n{paragraphs_b[j]}\n\n相似度: {similarity:.4f}\n\n{'=' * 50}\n\n")

    write_file('result.txt', ''.join(similar_content))


if __name__ == "__main__":
    main()
